"""
End-to-end: load a trained YOLO checkpoint, run it on a new swatch photo,
and print reconstructed row-by-row instructions.
"""
import argparse
from ultralytics import YOLO
from row_reconstruction import Detection, reconstruct_instructions


def run(weights: str, image: str, conf: float, alternate_direction: bool):
    model = YOLO(weights)
    results = model.predict(source=image, conf=conf, verbose=False)[0]

    names = results.names  # class index -> name, from the model itself
    detections = []
    for box in results.boxes:
        cls_idx = int(box.cls.item())
        x_center, y_center = box.xywh[0][:2].tolist()
        detections.append(
            Detection(
                cls_name=names[cls_idx],
                x_center=x_center,
                y_center=y_center,
                confidence=float(box.conf.item()),
            )
        )

    print(f"Detected {len(detections)} stitches\n")
    for line in reconstruct_instructions(detections, alternate_direction=alternate_direction):
        print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--conf", type=float, default=0.35,
                         help="confidence threshold -- tune per how dense/noisy your swatches are")
    parser.add_argument("--no-alternate", action="store_true",
                         help="disable alternating row direction (use if swatches aren't turned rows)")
    args = parser.parse_args()
    run(args.weights, args.image, args.conf, alternate_direction=not args.no_alternate)
