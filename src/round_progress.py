"""
Bar chart of overall mAP50 across training rounds, reading each round's
summary.json (written by train.py's report_and_save).

Usage:
    python3 src/round_progress.py \
      --run round1_yolov8s:round1 \
      --run round1round2_yolov8s:"round1+round2" \
      --out runs/round_progress.png

Each --run is "<runs/detect subdir>:<x-axis label>" -- the label defaults to
the subdir name if omitted.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_map50(run_dir: Path) -> float:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(
            f"{summary_path} not found -- this run wasn't trained with the "
            "current train.py (which writes summary.json), or the run name is wrong."
        )
    return json.loads(summary_path.read_text())["mAP50"]


def plot(runs: list[tuple[str, str]], detect_dir: Path, out_path: Path):
    labels, values = [], []
    for run_name, label in runs:
        values.append(load_map50(detect_dir / run_name))
        labels.append(label)

    fig, ax = plt.subplots(figsize=(1.5 * len(labels) + 2, 5))
    bars = ax.bar(labels, values, color="#4C72B0")
    ax.set_ylabel("mAP50")
    ax.set_ylim(0, 1)
    ax.set_title("Overall mAP50 by round")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")
    for label, v in zip(labels, values):
        print(f"  {label:<20} mAP50={v:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", action="append", required=True, dest="runs",
        help='runs/detect subdir for a round, optionally "<subdir>:<label>". Repeatable, in round order.',
    )
    parser.add_argument("--detect-dir", type=Path, default=Path("runs/detect"))
    parser.add_argument("--out", type=Path, default=Path("runs/round_progress.png"))
    args = parser.parse_args()

    parsed = []
    for r in args.runs:
        if ":" in r:
            name, label = r.split(":", 1)
        else:
            name, label = r, r
        parsed.append((name, label))

    plot(parsed, args.detect_dir, args.out)
