"""
Fine-tune a YOLO detection model on the stitch dataset produced by data_prep.py.

Swap --model to a *-seg.pt checkpoint later if you decide you need pixel masks
instead of boxes -- the rest of the pipeline (row_reconstruction.py) only
relies on box centers + class labels, so it doesn't need to change.
"""
import argparse
from ultralytics import YOLO


def main(data_yaml: str, model_name: str, epochs: int, imgsz: int):
    model = YOLO(model_name)  # loads pretrained COCO weights as a starting point
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        # crochet swatches have lots of fine, repetitive texture -- keep
        # augmentation modest so the model doesn't blur adjacent stitches
        mosaic=0.3,
        degrees=10,
        flipud=0.0,   # a flipped swatch usually isn't a valid stitch orientation
        fliplr=0.5,
        patience=30,
    )
    metrics = model.val()
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="path to stitches.yaml")
    parser.add_argument("--model", type=str, default="yolov8s.pt",
                         help="starting checkpoint, e.g. yolov8s.pt or yolov8s-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960,
                         help="higher than the YOLO default (640) since stitches are small, dense objects")
    args = parser.parse_args()
    main(args.data, args.model, args.epochs, args.imgsz)
