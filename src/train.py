"""
Fine-tune a YOLO detection model on the stitch dataset produced by data_prep.py.

Swap --model to a *-seg.pt checkpoint later if you decide you need pixel masks
instead of boxes -- the rest of the pipeline (row_reconstruction.py) only
relies on box centers + class labels, so it doesn't need to change.
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

# below this recall, flag the class as weak in the per-class report
WEAK_RECALL_THRESHOLD = 0.5


def build_per_class_report(metrics, weak_recall_threshold: float = WEAK_RECALL_THRESHOLD) -> str:
    """Per-class precision/recall/mAP, plus which classes need labeling attention.

    model.val()'s default __str__ dumps raw PR-curve arrays (1000 points per
    class) which is unreadable -- this builds one line per class instead, and
    separates "weak recall" (evaluated, but missed too many) from "no val
    data" (can't be assessed yet -- usually the rarest/compound stitches).
    """
    all_names = metrics.names  # full class list from the data yaml, index -> name
    rows = metrics.summary()  # one dict per class that had val instances
    by_name = {row["Class"]: row for row in rows}

    overall = metrics.results_dict
    lines = [
        f"{'all':<12}{'':>8}{'':>11}"
        f"{overall['metrics/precision(B)']:>11.3f}{overall['metrics/recall(B)']:>9.3f}"
        f"{overall['metrics/mAP50(B)']:>9.3f}{overall['metrics/mAP50-95(B)']:>10.3f}",
        "",
        f"{'class':<12}{'images':>8}{'instances':>11}{'precision':>11}{'recall':>9}{'mAP50':>9}{'mAP50-95':>10}",
    ]
    weak, missing = [], []
    for idx in sorted(all_names):
        name = all_names[idx]
        row = by_name.get(name)
        if row is None:
            lines.append(f"{name:<12}{'-':>8}{'0':>11}{'-':>11}{'-':>9}{'-':>9}{'-':>10}")
            missing.append(name)
            continue
        lines.append(
            f"{name:<12}{row['Images']:>8}{row['Instances']:>11}"
            f"{row['Box-P']:>11.3f}{row['Box-R']:>9.3f}{row['mAP50']:>9.3f}{row['mAP50-95']:>10.3f}"
        )
        if row["Box-R"] < weak_recall_threshold:
            weak.append(name)

    if missing:
        lines.append(f"\nNo val instances yet (can't assess): {', '.join(missing)}")
    if weak:
        lines.append(f"Weak recall (<{weak_recall_threshold:.0%}): {', '.join(weak)}")
    if missing or weak:
        lines.append("-> prioritize these classes when correcting model-assisted predictions.")

    return "\n".join(lines)


def report_and_save(metrics, weak_recall_threshold: float = WEAK_RECALL_THRESHOLD):
    report = build_per_class_report(metrics, weak_recall_threshold)
    print("\n" + report)

    save_dir = Path(metrics.save_dir)
    report_path = save_dir / "per_class_report.txt"
    report_path.write_text(report + "\n")

    overall = metrics.results_dict
    summary_path = save_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "precision": overall["metrics/precision(B)"],
        "recall": overall["metrics/recall(B)"],
        "mAP50": overall["metrics/mAP50(B)"],
        "mAP50-95": overall["metrics/mAP50-95(B)"],
    }, indent=2))

    all_names = metrics.names
    by_name = {row["Class"]: row for row in metrics.summary()}
    class_metrics_path = save_dir / "class_metrics.json"
    class_metrics_path.write_text(json.dumps({
        name: (
            {"precision": row["Box-P"], "recall": row["Box-R"], "mAP50": row["mAP50"], "mAP50-95": row["mAP50-95"]}
            if (row := by_name.get(name)) else None
        )
        for name in all_names.values()
    }, indent=2))

    print(f"\nSaved report to {report_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved per-class metrics to {class_metrics_path}")


def main(data_yaml: str, model_name: str, epochs: int, imgsz: int, name: str | None):
    model = YOLO(model_name)  # loads pretrained COCO weights as a starting point
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        name=name,  # e.g. "round1_yolov8s" -- keeps runs/detect/ self-describing across active-learning rounds
        # crochet swatches have lots of fine, repetitive texture -- keep
        # augmentation modest so the model doesn't blur adjacent stitches
        mosaic=0.3,
        degrees=10,
        flipud=0.0,   # a flipped swatch usually isn't a valid stitch orientation
        fliplr=0.5,
        patience=30,
    )
    metrics = model.val(name=f"{name}_val" if name else None)
    report_and_save(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="path to stitches.yaml")
    parser.add_argument("--model", type=str, default="yolov8s.pt",
                         help="starting checkpoint, e.g. yolov8s.pt or yolov8s-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960,
                         help="higher than the YOLO default (640) since stitches are small, dense objects")
    parser.add_argument("--name", type=str, default=None,
                         help="run name, e.g. round1_yolov8s -- saved under runs/detect/<name>/. "
                              "Omit to fall back to Ultralytics' auto-incrementing train/train2/...")
    args = parser.parse_args()
    main(args.data, args.model, args.epochs, args.imgsz, args.name)
