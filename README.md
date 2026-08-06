# Crochet Stitch ID + Counting Pipeline (sketch)

Goal: take a photo of a crochet swatch, detect every stitch, classify its type,
and reconstruct row-by-row instructions (e.g. "Row 1: 12 sc, Row 2: 6 dc, 2 inc...").

Pipeline stages:

1. `data_prep.py` — convert hand-labeled annotations (bounding boxes + class per
   stitch) into the YOLO format Ultralytics expects, and split into train/val.
2. `train.py` — fine-tune a YOLOv8/v11 detection (or -seg) model on your stitch
   classes.
3. `row_reconstruction.py` — turn a flat list of detections (box + class) into
   ordered rows using vertical clustering + horizontal sorting.
4. `inference.py` — glue it together: run the trained model on a new image,
   reconstruct rows, print pattern-style instructions.

## Why detection-first, not segmentation-first

Instructions need "what stitch, which row, what position" — not a pixel-exact
mask. Bounding-box centers are enough to reconstruct row order. Start here;
only move to `-seg` weights (drop-in swap, same Ultralytics API) if you later
need exact stitch geometry (e.g. to catch increases/decreases from shape
distortion rather than count anomalies).

## Data format expected by data_prep.py

A simple JSON per image:

```json
{
  "image": "swatch_001.jpg",
  "width": 1024,
  "height": 768,
  "stitches": [
    {"class": "sc", "bbox": [x_min, y_min, x_max, y_max]},
    {"class": "dc", "bbox": [x_min, y_min, x_max, y_max]}
  ]
}
```

Put these under `raw_annotations/*.json` alongside the source images in
`raw_images/`.

## Quickstart

```bash
pip install ultralytics --break-system-packages

python data_prep.py --ann-dir raw_annotations --img-dir raw_images --out data/
python train.py --data data/stitches.yaml --epochs 100 --model yolov8s.pt
python inference.py --weights runs/detect/train/weights/best.pt --image new_swatch.jpg
```
