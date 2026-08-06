"""
Convert per-image JSON stitch annotations into YOLO-format labels and
produce the data.yaml Ultralytics needs for training.

Input layout:
    raw_images/*.jpg
    raw_annotations/*.json   (same stem as the matching image)

Each JSON looks like:
{
  "image": "swatch_001.jpg",
  "width": 1024,
  "height": 768,
  "stitches": [
    {"class": "sc", "bbox": [x_min, y_min, x_max, y_max]},
    ...
  ]
}

Output layout (YOLO expects this exact structure):
    out/
      images/train/*.jpg
      images/val/*.jpg
      labels/train/*.txt
      labels/val/*.txt
      stitches.yaml
"""
import argparse
import json
import random
import shutil
from pathlib import Path

# Fixed class order -> class index mapping. Extend as your label set grows.
STITCH_CLASSES = [
    "ch",      # chain
    "sc",      # single crochet
    "hdc",     # half double crochet
    "dc",      # double crochet
    "tr",      # treble
    "sc_inc", "sc_dec",
    "dc_inc", "dc_dec",
    "fpdc", "bpdc",  # front/back post double crochet
]


def bbox_to_yolo(bbox, img_w, img_h):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2 / img_w
    y_center = (y_min + y_max) / 2 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return x_center, y_center, w, h


def convert(ann_dir: Path, img_dir: Path, out_dir: Path, val_frac: float = 0.15, seed: int = 0):
    ann_files = sorted(ann_dir.glob("*.json"))
    if not ann_files:
        raise SystemExit(f"No annotation JSON files found in {ann_dir}")

    random.seed(seed)
    random.shuffle(ann_files)
    n_val = max(1, int(len(ann_files) * val_frac))
    splits = {"val": ann_files[:n_val], "train": ann_files[n_val:]}

    class_to_idx = {c: i for i, c in enumerate(STITCH_CLASSES)}
    skipped_classes = set()

    for split, files in splits.items():
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        for ann_path in files:
            data = json.loads(ann_path.read_text())
            img_name = data["image"]
            src_img = img_dir / img_name
            if not src_img.exists():
                print(f"WARNING: missing image {src_img}, skipping")
                continue

            shutil.copy(src_img, out_dir / "images" / split / img_name)

            lines = []
            for st in data["stitches"]:
                cls_name = st["class"]
                if cls_name not in class_to_idx:
                    skipped_classes.add(cls_name)
                    continue
                cls_idx = class_to_idx[cls_name]
                xc, yc, w, h = bbox_to_yolo(st["bbox"], data["width"], data["height"])
                lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            label_path = out_dir / "labels" / split / (Path(img_name).stem + ".txt")
            label_path.write_text("\n".join(lines))

        print(f"{split}: {len(files)} images")

    if skipped_classes:
        print(f"NOTE: encountered classes not in STITCH_CLASSES, skipped: {sorted(skipped_classes)}")
        print("Add them to STITCH_CLASSES at the top of this script and rerun.")

    yaml_path = out_dir / "stitches.yaml"
    yaml_path.write_text(
        "path: {}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n{}\n".format(
            out_dir.resolve(),
            "\n".join(f"  {i}: {c}" for i, c in enumerate(STITCH_CLASSES)),
        )
    )
    print(f"Wrote {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-dir", type=Path, required=True)
    parser.add_argument("--img-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val-frac", type=float, default=0.15)
    args = parser.parse_args()
    convert(args.ann_dir, args.img_dir, args.out, args.val_frac)
