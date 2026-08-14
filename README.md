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

Each labeling round is a CVAT "YOLO 1.1" export, stored under
`stitch_annotations/<round_name>/`:

```
stitch_annotations/round1/
  obj.names            one class name per line, in CVAT's class-index order
  obj.data
  obj_train_data/*.txt one label file per image, YOLO-normalized boxes
  train.txt
  swatch_images/*.png  the source photos (exported separately from CVAT)
```

`data_prep.py` matches each label file to its image by filename stem, remaps
CVAT's class ids onto the canonical ids in `stitches.csv` (the two can drift
independently as labeling and the class list evolve), and writes the
Ultralytics-ready split to `data/` (gitignored, regenerate anytime).

## Quickstart

```bash
pip install -r requirements.txt

python src/data_prep.py \
  --export-dir stitch_annotations/round1 \
  --img-dir stitch_annotations/round1/swatch_images \
  --stitches-csv stitches.csv \
  --out data
python src/train.py --data data/stitches.yaml --epochs 100 --model yolov8s.pt --name round1_yolov8s
python src/inference.py --weights runs/detect/round1_yolov8s/weights/best.pt --image new_swatch.jpg
```

For model-assisted labeling of a new round (run the current model on new,
unlabeled photos to get corrected-not-hand-traced starting annotations):

```bash
python src/predict_for_review.py \
  --weights runs/detect/round1_yolov8s/weights/best.pt \
  --images stitch_annotations/round2/swatch_images \
  --out stitch_annotations/round2/predicted
```
