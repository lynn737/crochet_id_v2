"""
Run a trained model on a batch of new, unlabeled images and export the
predictions as a CVAT "YOLO 1.1" package -- the same layout stitch_labelled/
came in as -- so you can import it as pre-annotations and correct boxes
instead of hand-tracing from scratch.

Output layout:
    out/
      obj.names
      obj.data
      obj_train_data/*.<ext>   (copies of the input images)
      obj_train_data/*.txt     (predicted boxes, YOLO-normalized)
      train.txt

Import into CVAT: create a task from obj_train_data/*.<ext>, then import
annotations from this folder using the "YOLO 1.1" format.
"""
import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

IMG_EXTS = (".png", ".jpg", ".jpeg")


def _ordered_names(stitches_csv: Path) -> list[str]:
    rows = {}
    with stitches_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            rows[int(row["class_id"])] = row["abbreviation"].strip()
    return [rows[i] for i in sorted(rows)]


def predict_for_review(
    weights: Path,
    img_dir: Path,
    stitches_csv: Path,
    out_dir: Path,
    conf: float,
    imgsz: int,
):
    images = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"No images found in {img_dir}")

    names = _ordered_names(stitches_csv)

    data_dir = out_dir / "obj_train_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    results = model.predict(source=[str(p) for p in images], conf=conf, imgsz=imgsz, verbose=False)

    class_counts = Counter()
    for img_path, result in zip(images, results):
        shutil.copy(img_path, data_dir / img_path.name)

        lines = []
        for box in result.boxes:
            cls_id = int(box.cls.item())
            xc, yc, w, h = box.xywhn[0].tolist()
            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            class_counts[names[cls_id]] += 1

        (data_dir / f"{img_path.stem}.txt").write_text("\n".join(lines))

    (out_dir / "obj.names").write_text("\n".join(names) + "\n")
    (out_dir / "obj.data").write_text(
        f"classes = {len(names)}\ntrain = data/train.txt\nnames = data/obj.names\nbackup = backup/\n"
    )
    (out_dir / "train.txt").write_text(
        "\n".join(f"data/obj_train_data/{p.name}" for p in images) + "\n"
    )

    print(f"Predicted {sum(class_counts.values())} boxes across {len(images)} images -> {out_dir}")
    print("\npredictions per class:")
    for name in names:
        print(f"  {name:<12}{class_counts.get(name, 0)}")
    zero_pred = [n for n in names if class_counts.get(n, 0) == 0]
    if zero_pred:
        print(f"\nNo predictions at all for: {', '.join(zero_pred)}")
        print("-> these classes need the most correction attention (or aren't in this image batch).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True, help="trained model, e.g. runs/detect/train/weights/best.pt")
    parser.add_argument("--images", type=Path, required=True, help="folder of new, unlabeled images")
    parser.add_argument("--stitches-csv", type=Path, default=Path("stitches.csv"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold for predictions to include")
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()
    predict_for_review(args.weights, args.images, args.stitches_csv, args.out, args.conf, args.imgsz)
