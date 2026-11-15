"""Run a reproducible YOLO validation pass on a labeled Ultralytics dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from detector import PERFORMANCE_PROFILES

PROJECT_COCO_CLASSES = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Ultralytics dataset YAML with a labeled validation split")
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("yolov8n.pt"))
    parser.add_argument("--profile", choices=PERFORMANCE_PROFILES, default="Balanced")
    parser.add_argument("--confidence", type=float, default=0.001,
                        help="Validation threshold; 0.001 retains low-confidence predictions for AP calculation")
    parser.add_argument("--device", default="cpu", help="cpu, mps, 0, etc. Defaults to CPU")
    parser.add_argument("--output", type=Path, default=Path("evaluation_report.json"))
    args = parser.parse_args()

    if not args.dataset.is_file():
        parser.error(f"Dataset YAML was not found: {args.dataset}")
    if not args.model.is_file():
        parser.error(f"Model weights were not found: {args.model}")
    if not 0 < args.confidence < 1:
        parser.error("--confidence must be between 0 and 1")

    model = YOLO(str(args.model))
    metrics = model.val(
        data=str(args.dataset), imgsz=PERFORMANCE_PROFILES[args.profile],
        conf=args.confidence, device=args.device, plots=True, verbose=False,
    )
    box = metrics.box
    project_per_class = {}
    for metric_index, class_id in enumerate(box.ap_class_index):
        class_id = int(class_id)
        if class_id not in PROJECT_COCO_CLASSES:
            continue
        precision, recall, map50, map50_95 = box.class_result(metric_index)
        project_per_class[str(metrics.names[class_id])] = {
            "precision": float(precision), "recall": float(recall),
            "map50": float(map50), "map50_95": float(map50_95),
        }
    present_scores = list(project_per_class.values())
    project_mean = {
        key: sum(item[key] for item in present_scores) / len(present_scores) if present_scores else 0.0
        for key in ("precision", "recall", "map50", "map50_95")
    }
    dataset_limitations = ["Metrics describe this labeled dataset, split, image size, and confidence setting only.",
                           "This benchmark does not validate calibrated heavy-vehicle blind zones or safe driving use."]
    if "coco128" in args.dataset.name.lower():
        dataset_limitations.append(
            "COCO128 is a 128-image subset of COCO train2017, not a held-out test set; the pretrained model may have seen these images during training. Treat this as a pipeline smoke baseline, not a reliable generalization estimate."
        )
    report = {
        "dataset_yaml": str(args.dataset.resolve()),
        "dataset_split": "Validation split configured by the supplied YAML; verify it is held out before treating metrics as generalization results.",
        "model": str(args.model.resolve()),
        "profile": args.profile,
        "image_size": PERFORMANCE_PROFILES[args.profile],
        "confidence_threshold": args.confidence,
        "device": args.device,
        "reported_project_coco_class_ids": PROJECT_COCO_CLASSES,
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "project_classes_present": len(present_scores),
        "project_class_mean": project_mean,
        "project_per_class": project_per_class,
        "ultralytics_output_dir": str(metrics.save_dir),
        "limitations": dataset_limitations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
