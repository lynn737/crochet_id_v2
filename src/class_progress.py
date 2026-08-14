"""
Line graph of per-class mAP50 across rounds, one line per stitch class, one
dot per round. Reads class_metrics.json (written by train.py's
report_and_save) from each named run under runs/detect/.

Usage:
    python3 src/class_progress.py \
      --run run_1_5_val:round1-5 \
      --run run_1_6_val:round1-6 \
      --out runs/class_progress.png

Each --run is "<runs/detect subdir>:<x-axis label>" -- the label defaults to
the subdir name if omitted. Runs must be given in round order.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_class_map50(run_dir: Path) -> dict[str, float | None]:
    path = run_dir / "class_metrics.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found -- this run predates class_metrics.json "
            "(needs report_and_save from the current train.py) or the run name is wrong."
        )
    data = json.loads(path.read_text())
    return {name: (row["mAP50"] if row else None) for name, row in data.items()}


def plot(runs: list[tuple[str, str]], detect_dir: Path, out_path: Path):
    labels = [label for _, label in runs]
    per_run = [load_class_map50(detect_dir / name) for name, _ in runs]
    class_names = list(per_run[0].keys())

    fig, ax = plt.subplots(figsize=(1.3 * len(labels) + 3, 6))
    cmap = plt.get_cmap("tab20")
    for i, cls in enumerate(class_names):
        ys = [run.get(cls) for run in per_run]
        xs = [x for x, y in zip(labels, ys) if y is not None]
        ys_present = [y for y in ys if y is not None]
        ax.plot(xs, ys_present, marker="o", label=cls, color=cmap(i % 20))

    ax.set_ylabel("mAP50")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class mAP50 by round")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")
    for cls in class_names:
        vals = ", ".join(
            f"{label}={run[cls]:.3f}" if run.get(cls) is not None else f"{label}=n/a"
            for run, (_, label) in zip(per_run, runs)
        )
        print(f"  {cls:<8} {vals}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", action="append", required=True, dest="runs",
        help='runs/detect subdir for a round, optionally "<subdir>:<label>". Repeatable, in round order.',
    )
    parser.add_argument("--detect-dir", type=Path, default=Path("runs/detect"))
    parser.add_argument("--out", type=Path, default=Path("runs/class_progress.png"))
    args = parser.parse_args()

    parsed = []
    for r in args.runs:
        if ":" in r:
            name, label = r.split(":", 1)
        else:
            name, label = r, r
        parsed.append((name, label))

    plot(parsed, args.detect_dir, args.out)
