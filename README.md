# DRISHTI · Intelligent Blind-Spot Warning System for Heavy Vehicles

A futuristic, high-performance desktop application for road-user tracking and heavy vehicle blind-spot warning. Built using YOLOv8n, ByteTrack, OpenCV, and CustomTkinter, inspired by ISRO aerospace telemetry and dedicated to **Safer Roads, Stronger India**.

## Safety scope

This is an educational demo and algorithm prototype, not a certified vehicle safety product. Warning zones are image-space screen-coordinate overlays. They are not calibrated to specific physical truck dimensions, camera pose, road distance, or real blind spots. Do not use the application to make live driving or safety decisions.

## Included features

- **Aerospace & Defense Inspired HUD**: High-contrast dark dashboard with DRISHTI & ISRO branding, Indian tricolor motifs, live ticking clock, and system telemetry.
- **Multi-Tab Architecture**: Seamless switching between **Live Monitor**, **Analytics**, **History**, **Settings**, and **About** views.
- **Dual-Zone Detection**: Independent simultaneous evaluation of **Left Blind Zone** (mirror area) and **Right Blind Zone** (lane traffic), plus single-zone modes.
- **Interactive Zone Sliders**: Real-time adjustment of **Zone Width** (lateral spread) and **Zone Sensitivity** (confidence thresholding).
- **Interactive Polygon Dragging**: Visual vertex dragging directly on the live camera canvas, with point addition and deletion.
- **HUD Alert Cards & Visual Badges**: Instant glowing red alert banner, high-visibility `⚠️ BLIND SPOT ALERT!` in-zone HUD pill badges, and bounding box tagging.
- **Category Metrics with Deltas**: Real-time tracking of **Vehicles**, **Pedestrians**, and **Two-Wheelers** with live increment badges.
- **Connected Event Timeline**: Chronological vertical timeline linking detected safety events, entries, and clearings.
- **Embedded Analytics**: Built-in 30-day class distribution bar charts and daily trend graphs.
- **Full Historical Telemetry**: SQLite-backed history table with event filters and one-click CSV export.
- **Customizable Video & Camera Input**: Support for local webcams 0–5, video files (MP4, AVI, MOV, WebM), and one-click looping DRISHTI road demo.
- **CPU performance path**: Uses a reduced Fast inference size, warmed-up/fused YOLO layers where supported, limited OpenCV threading, a short camera buffer, and stale-frame dropping for recorded video. Skipped video frames are surfaced in the live performance badge; faster processing can miss brief events.
- **Optimized local inference**: Uses fused YOLO layers where supported, a warmed-up model at the selected resolution, limited OpenCV threading, a short camera buffer, and stale-frame dropping for recorded video to reduce lag.
- **Event-specific voice warning**: On a new zone entry, the app can speak the detected class and the left/right image-space side. Windows speech support uses `pyttsx3`; the visual alert and warning tone remain available if speech is unavailable. Mute controls both audio outputs.
- **Transparent performance display**: The video badge separates model inference FPS from displayed UI FPS and reports the number of recorded-video frames skipped to catch up.
- **Reproducible model evaluation tool**: `evaluate_model.py` runs Ultralytics validation against a labeled dataset YAML and saves precision, recall, mAP50, and mAP50-95 to JSON, plus the standard plots.

## Run on Windows

Double-click `run.bat` or run:

```bat
run.bat --demo
```

This launches the local Python runtime under `.python` with CPU PyTorch and YOLOv8n weights into the looping DRISHTI road demo.

## Landing page

Open `landing/index.html` in a browser or double-click `run_landing.bat`. The responsive static page follows the supplied cream, navy, serif-led visual reference and uses the bundled demo clip. Its “Copy run command” button copies `run.bat --demo`. The generated `support.js` and vendor React runtime from the design export were not copied; the page uses small project-owned HTML, CSS, and JavaScript files instead.

### Fresh setup and recovery

Use Python 3.10–3.12. If the bundled `.python` runtime is absent or incomplete, create a local environment and install the listed packages:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --demo
```

The model weights are `yolov8n.pt` beside `main.py`. If they are missing, the first model load needs internet access to download them; otherwise, place the weights file there. For a camera error, grant camera access in Windows, close other apps using the webcam, and try another listed camera. If `pyttsx3` or a Windows speech voice is unavailable, install requirements again; the visual alert and tone continue to work.

### Measuring detection quality

The bundled NASA street timelapse is useful for showing the interface, but its night lighting and motion blur make it unsuitable for an accuracy score. The included COCO128 configuration downloads its labeled 128-image sample on first use. COCO128 comes from COCO train2017, so it is **not a held-out test set** and its score is not a trustworthy estimate of generalization. For a proper measured result, use a representative, separately held-out, labeled validation split in Ultralytics format with the standard 80 COCO class IDs/names (including person 0, bicycle 1, car 2, motorcycle 3, bus 5, and truck 7), then run:

```powershell
.\.venv\Scripts\python.exe evaluate_model.py path\to\dataset.yaml --profile Balanced --device cpu --output evaluation_report.json
```

The smoke baseline can be reproduced with `python evaluate_model.py evaluation_coco128.yaml --profile Balanced --device cpu --output evaluation_report.json`. The command generates a local report and plots. Reports record precision, recall, mAP50, mAP50-95, supported-class metrics, profile, image size, threshold, dataset path, limitations, and plot output directory. Results apply only to that labeled dataset and configuration; they do not certify the warning zones or system for vehicle safety.

## Demo video credit

The bundled `assets/demo_road.webm` is **Street Timelapse**, credited to Megan Willy (NASA/Goddard Space Flight Center), from [NASA Scientific Visualization Studio item 10397](https://svs.gsfc.nasa.gov/10397). See [assets/ATTRIBUTION.md](assets/ATTRIBUTION.md). The clip is a timelapse and is included only to demonstrate the video and detection workflow.

## Controls

1. **DRISHTI Road Demo** explicitly starts the bundled road clip, applies the right-side polygon preset, loops playback, and uses Detail inference so distant traffic is more likely to register.
2. **Start** uses the selected camera unless a video has been uploaded. Use **Pause**, **Stop**, and **Restart** as needed.
3. Select **Upload** for MP4, AVI, MOV, or WebM files. The file dialog validates the source when playback starts.
4. Select one of the polygon presets or edit it directly on the displayed image. Edits apply to the current session.
5. Adjust confidence, audio volume, or mute. The warning stays visible if audio is unavailable.
6. Export the event timeline as CSV or open Analytics for class and day-by-day summaries.
7. Read the video badge as separate AI inference FPS and displayed UI FPS. The skipped-frame count shows how often recorded playback has jumped ahead to avoid accumulating lag.

## Files

```text
main.py                 Dashboard, zone editing, event timeline, and analytics
detector.py             OpenCV capture, YOLO tracking, and polygon checks
database.py             SQLite history, summaries, and CSV export
audio_alert.py          Computer-speaker alert tone
evaluate_model.py       Labeled-dataset YOLO validation and JSON report
evaluation_coco128.yaml COCO128 validation configuration (smoke baseline)
evaluation_report.json  Reproducible run summary and stated limitations
assets/demo_road.webm   Bundled road-video demo
run.bat                 Windows launch shortcut
```

Event history is stored in `data/detections.db`. Existing legacy per-frame rows are left intact; the new distinct event timeline uses the `detection_events` table.
