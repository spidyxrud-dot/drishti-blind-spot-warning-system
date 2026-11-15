"""Threaded OpenCV capture, YOLO tracking, and image-space zone analysis."""
from __future__ import annotations

import time
import math
import threading
import zlib
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Full, Empty
from typing import Optional

import cv2
import numpy as np

CLASSES = {0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
VEHICLES = {"Car", "Bus", "Truck"}
TWO_WHEELERS = {"Bicycle", "Motorcycle"}
PERFORMANCE_PROFILES = {"Ultra Fast": 320, "Fast": 416, "Balanced": 512, "Detail": 640}
_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_INFERENCE_LOCK = threading.Lock()


def scan_available_cameras(max_probe: int = 6) -> list[int]:
    """Probes connected camera hardware indices and returns verified active camera IDs."""
    import sys
    available = []
    for i in range(max_probe):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available if available else [0]


@dataclass
class Detection:
    name: str
    confidence: float
    box: tuple[int, int, int, int]
    center: tuple[int, int]
    in_zone: bool
    track_id: int
    zone_name: str = ""
    velocity: tuple[float, float] = (0.0, 0.0)
    speed_px: float = 0.0
    motion_state: str = "STEADY"  # "APPROACHING", "RECEDING", "STEADY"


class SourceError(RuntimeError):
    """An actionable camera or video source error."""


class ModelError(RuntimeError):
    """Model load or inference startup error."""


def _get_model(weights: Path, imgsz: int, device: str, status_callback) -> object:
    """Load and warm YOLO once per process, then reuse it across source restarts."""
    key = (str(weights) if weights.exists() else "yolov8n.pt", device)
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is not None:
            return model
        from ultralytics import YOLO
        model = YOLO(key[0])
        try:
            model.fuse()
        except Exception:
            pass
        status_callback(f"Preparing YOLOv8n on {device.upper()} (first run only)")
        model.predict(np.zeros((320, 320, 3), dtype=np.uint8), imgsz=imgsz,
                      verbose=False, device=device)
        _MODEL_CACHE[key] = model
        return model


class DetectionWorker:
    def __init__(self, source: int | str | Path, confidence: float,
                 zone: dict[str, list[tuple[float, float]]] | list[tuple[float, float]],
                 output: Queue,
                 *, loop_video: bool = False,
                 zone_mode: str = "both",
                 performance_mode: str = "Fast") -> None:
        self.source = source
        self.confidence = confidence
        if isinstance(zone, dict):
            self.zones = {k: list(v) for k, v in zone.items()}
            self.zone = list(zone.get(zone_mode, next(iter(zone.values()), [])))
        else:
            self.zones = {"active": list(zone)}
            self.zone = list(zone)
        self.zone_mode = zone_mode
        self.output = output
        self.loop_video = loop_video
        self.performance_mode = performance_mode if performance_mode in PERFORMANCE_PROFILES else "Fast"
        self.imgsz = PERFORMANCE_PROFILES[self.performance_mode]
        self.night_mode = False
        self._prev_centers: dict[int, tuple[int, int, int]] = {}
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.error: Optional[str] = None
        self.is_camera = isinstance(source, int)
        self.input_size: tuple[int, int] = (0, 0)
        self._fallback_tracks: dict[int, tuple[str, tuple[int, int, int, int], int]] = {}
        self._next_fallback_id = -1
        self._frame_index = 0
        self._loop_index = 0

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name="video-inference", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.pause_event.clear()

    def pause(self) -> None:
        self.pause_event.set()

    def resume(self) -> None:
        self.pause_event.clear()

    def _send(self, item: dict) -> None:
        try:
            self.output.put_nowait(item)
        except Full:
            try:
                old = self.output.get_nowait()
                if old.get("kind") != "frame" and item.get("kind") == "frame":
                    self.output.put_nowait(old)
                    return
            except Empty:
                pass
            try:
                self.output.put_nowait(item)
            except Full:
                pass

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        union = area_a + area_b - intersection
        return intersection / union if union else 0.0

    def _fallback_id(self, name: str, box: tuple[int, int, int, int], used: set[int]) -> int:
        best_id, best_iou = None, 0.15
        for track_id, (old_name, old_box, old_frame) in self._fallback_tracks.items():
            if track_id in used or old_name != name or self._frame_index - old_frame > 12:
                continue
            overlap = self._iou(box, old_box)
            if overlap > best_iou:
                best_id, best_iou = track_id, overlap
        if best_id is None:
            best_id = self._next_fallback_id
            self._next_fallback_id -= 1
        self._fallback_tracks[best_id] = (name, box, self._frame_index)
        return best_id

    def _detect(self, frame, model) -> list[Detection]:
        height, width = frame.shape[:2]
        active_zones: dict[str, np.ndarray] = {}
        for zname, zpts in self.zones.items():
            if self.zone_mode in ("both", "Both Sides", "all") or self.zone_mode.lower() in zname.lower():
                pts = np.array([[int(x * width), int(y * height)] for x, y in zpts], dtype=np.int32)
                if len(pts) >= 3:
                    active_zones[zname] = pts

        with _MODEL_INFERENCE_LOCK:
            result = model.track(frame, persist=True, tracker="bytetrack.yaml",
                                 conf=float(self.confidence), classes=list(CLASSES),
                                 imgsz=self.imgsz, verbose=False)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        explicit_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
        found: list[Detection] = []
        used: set[int] = set()
        for index, row in enumerate(boxes):
            class_id = int(row.cls[0])
            if class_id not in CLASSES:
                continue
            x1, y1, x2, y2 = map(int, row.xyxy[0].cpu().tolist())
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            in_zone = False
            zone_name = ""
            for zname, zpx in active_zones.items():
                # Match the documented rule: zone membership is based on box centre only.
                if cv2.pointPolygonTest(zpx, center, False) >= 0:
                    in_zone = True
                    zone_name = zname
                    break

            raw_id = explicit_ids[index] if index < len(explicit_ids) else None
            track_id = int(raw_id) if raw_id is not None else self._fallback_id(CLASSES[class_id], (x1, y1, x2, y2), used)
            used.add(track_id)
            self._fallback_tracks[track_id] = (CLASSES[class_id], (x1, y1, x2, y2), self._frame_index)

            # Calculate motion velocity and approach state
            velocity = (0.0, 0.0)
            speed_px = 0.0
            motion_state = "STEADY"
            if track_id in self._prev_centers:
                px, py, pframe = self._prev_centers[track_id]
                dt = max(1, self._frame_index - pframe)
                vx = (center[0] - px) / dt
                vy = (center[1] - py) / dt
                velocity = (vx, vy)
                speed_px = math.hypot(vx, vy)
                if vy > 2.0:
                    motion_state = "APPROACHING"
                elif vy < -2.0:
                    motion_state = "RECEDING"
                else:
                    motion_state = "STEADY"
            self._prev_centers[track_id] = (center[0], center[1], self._frame_index)

            found.append(Detection(CLASSES[class_id], float(row.conf[0]), (x1, y1, x2, y2),
                                   center, in_zone, track_id, zone_name,
                                   velocity, speed_px, motion_state))
        self._fallback_tracks = {key: value for key, value in self._fallback_tracks.items() if self._frame_index - value[2] <= 30}
        self._prev_centers = {key: value for key, value in self._prev_centers.items() if self._frame_index - value[2] <= 45}
        return found

    @staticmethod
    def _apply_clahe(img: np.ndarray) -> np.ndarray:
        """Applies adaptive histogram equalization and gamma enhancement for night & rain vision."""
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
            # Subtle gamma boost
            inv_gamma = 1.0 / 1.15
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
            return cv2.LUT(enhanced, table)
        except Exception:
            return img

    def _run(self) -> None:
        cap = None
        try:
            if self.is_camera:
                import sys
                cap = (cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                       if sys.platform.startswith("win") else cv2.VideoCapture(self.source))
            else:
                cap = cv2.VideoCapture(self.source)
            if not cap.isOpened() and self.is_camera and sys.platform.startswith("win"):
                cap.release()
                cap = cv2.VideoCapture(self.source, cv2.CAP_MSMF)
            if not cap.isOpened():
                if self.is_camera:
                    raise SourceError(
                        f"Camera {self.source} could not open. Check Windows camera permission, "
                        "close apps that may already be using it, then try another camera index."
                    )
                raise SourceError(
                    "The selected video could not be opened. Confirm the file still exists, "
                    "uses MP4, AVI, MOV, or WebM, and can play in another media player."
                )
            if self.is_camera:
                # Request a low-latency preview size before the first frame is read.
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            source_fps = source_fps if 1 <= source_fps <= 120 else 0.0
            self._send({"kind": "status", "text": "Camera connected" if self.is_camera else "Video ready", "size": (width, height)})
            preview_ok, first_frame = cap.read()
            if not preview_ok:
                if self.is_camera:
                    raise SourceError(f"Camera {self.source} opened but did not return a frame. Try another camera index or close apps using it.")
                raise SourceError("The selected video opened but did not return a frame.")
            self._send({"kind": "preview", "frame": first_frame})
            if self.stop_event.is_set():
                return
            self._send({"kind": "status", "text": "Loading YOLOv8n tracking model"})
            import os
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch.set_num_threads(max(1, min(6, (os.cpu_count() or 4) // 2)))
            cv2.setNumThreads(1)
            weights = Path(__file__).resolve().parent / "yolov8n.pt"
            try:
                model = _get_model(
                    weights, self.imgsz, device,
                    lambda text: self._send({"kind": "status", "text": text}),
                )
                with _MODEL_INFERENCE_LOCK:
                    if getattr(model, "predictor", None) is not None:
                        for tracker in getattr(model.predictor, "trackers", []):
                            tracker.reset()
            except Exception as exc:
                detail = ("Place yolov8n.pt beside main.py or install the project dependencies "
                          "with `python -m pip install -r requirements.txt`. If the weights are "
                          "not present, an internet connection is needed for the first download.\n\n")
                raise ModelError(detail + str(exc)) from exc
            if self.stop_event.is_set():
                return
            self._send({"kind": "status", "text": f"YOLOv8n ready on {device.upper()}"})
            previous_fingerprint = None
            inference_fps = 0.0
            frames_skipped_total = 0
            pending_frame = first_frame
            while not self.stop_event.is_set():
                if self.pause_event.is_set():
                    time.sleep(.05)
                    continue
                frame_started = time.perf_counter()
                if pending_frame is not None:
                    frame = pending_frame
                    pending_frame = None
                    ok = True
                else:
                    ok, frame = cap.read()
                if not ok:
                    if self.loop_video and not self.is_camera and not self.stop_event.is_set():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        previous_fingerprint = None
                        self._loop_index += 1
                        self._fallback_tracks.clear()
                        self._send({"kind": "looped"})
                        continue
                    if not self.is_camera:
                        self._send({"kind": "ended"})
                    break
                fingerprint = zlib.crc32(frame)
                if fingerprint == previous_fingerprint:
                    continue
                previous_fingerprint = fingerprint
                self._frame_index += 1
                infer_started = time.perf_counter()
                infer_frame = self._apply_clahe(frame) if self.night_mode else frame
                detections = self._detect(infer_frame, model)
                infer_ms = (time.perf_counter() - infer_started) * 1000
                instantaneous = 1000.0 / max(infer_ms, 1.0)
                inference_fps = instantaneous if not inference_fps else inference_fps * .75 + instantaneous * .25
                self._send({"kind": "frame", "frame": infer_frame if self.night_mode else frame, "detections": detections,
                            "zones": self.zones, "zone_mode": self.zone_mode, "night_mode": self.night_mode,
                            "zone": list(self.zone), "size": (frame.shape[1], frame.shape[0]),
                            "source": "Camera" if self.is_camera else "Video",
                            "inference_fps": inference_fps, "infer_ms": infer_ms,
                            "source_fps": source_fps, "frame_index": self._frame_index,
                            "frames_skipped": frames_skipped_total,
                            "loop_index": self._loop_index})
                if source_fps and not self.is_camera:
                    elapsed = time.perf_counter() - frame_started
                    # Drop stale decoded frames to keep playback responsive on slower CPUs.
                    # Cap the work per loop so a long stall cannot block stop/pause handling.
                    frames_behind = min(30, max(0, int(elapsed * source_fps) - 1))
                    for _ in range(frames_behind):
                        if not cap.grab():
                            break
                        frames_skipped_total += 1
                    remaining = 1.0 / source_fps - elapsed
                    if remaining > 0:
                        time.sleep(remaining)
        except SourceError as exc:
            self.error = str(exc)
            self._send({"kind": "error", "text": self.error, "source": "camera" if self.is_camera else "video"})
        except ModelError as exc:
            self.error = str(exc)
            self._send({"kind": "error", "text": self.error, "source": "model"})
        except Exception as exc:
            self.error = str(exc)
            self._send({"kind": "error", "text": self.error, "source": "model" if "model" in str(exc).lower() or "yolo" in str(exc).lower() else "processing"})
        finally:
            if cap is not None:
                cap.release()
            self._send({"kind": "finished"})
