"""
Convert one or more CVAT "YOLO 1.1" export rounds into the layout
Ultralytics expects, and split into train/val.

Input layout, per round (e.g. stitch_annotations/round1/):
    obj.names               one class name per line, in class-index order
    obj_train_data/*.txt    one label file per image, YOLO-normalized boxes:
                             "<class_id> <x_center> <y_center> <w> <h>" per line
    swatch_images/*.png     source photos, filename stem matches the label
                             file stem (not bundled in the CVAT export itself)

Output layout (YOLO expects this exact structure):
    out/
      images/train/*.png
      images/val/*.png
      labels/train/*.txt
      labels/val/*.txt
      stitches.yaml

Class indices are local to each round's own obj.names (CVAT assigns them by
label-creation order within that task, so they drift between rounds) and get
remapped to the canonical ids in stitches.csv by matching on name, not index.

Rounds can reuse the same filename for different photos (e.g. both round1
and round2 have a "bpdc.png"), so output filenames are prefixed with the
round dir's name to keep them distinct.
"""
import argparse
import csv
import random
import shutil
from pathlib import Path

IMG_EXTS = (".png", ".jpg", ".jpeg")


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def load_canonical_classes(stitches_csv: Path) -> dict[str, int]:
    """normalized name/abbreviation -> canonical class id, from stitches.csv."""
    lookup = {}
    with stitches_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            cls_id = int(row["class_id"])
            lookup[_normalize(row["name"])] = cls_id
            lookup[_normalize(row["abbreviation"])] = cls_id
    return lookup


def load_export_class_map(obj_names: Path, canonical: dict[str, int]) -> dict[int, int]:
    """CVAT export class id -> canonical class id, matched by name."""
    export_names = [n.strip() for n in obj_names.read_text().splitlines() if n.strip()]
    mapping = {}
    unmatched = []
    for export_id, name in enumerate(export_names):
        cls_id = canonical.get(_normalize(name))
        if cls_id is None:
            unmatched.append(name)
        else:
            mapping[export_id] = cls_id
    if unmatched:
        raise SystemExit(
            f"{obj_names} entries not found in stitches.csv: {unmatched}\n"
            "Add them to stitches.csv (or fix the name) and rerun."
        )
    return mapping


def stratified_split(
    items: list[dict], val_frac: float, seed: int, names: list[str]
) -> tuple[list[dict], list[dict]]:
    """Greedy split that guarantees every class appears in both train and val
    when it has enough images to allow that, instead of leaving coverage to
    chance the way a plain random shuffle does. A plain shuffle is exactly
    what caused e.g. 8 of 16 classes to have zero val images in round3 --
    picking rare classes first and choosing which image covers them avoids
    that instead of hoping a random draw happens to include them.
    """
    rng = random.Random(seed)
    n = len(items)
    target_val = max(1, round(n * val_frac))

    # Shuffle first: class_to_items preserves insertion order, and ties in
    # coverage_score below resolve to the *first* candidate. Without this,
    # items are still grouped in round-collection order (round1 first), so
    # every tie would be won by round1 images -- systematically skewing
    # val toward round1 regardless of val_frac.
    items = list(items)
    rng.shuffle(items)

    class_to_items: dict[int, list[int]] = {}
    for idx, item in enumerate(items):
        for c in item["classes"]:
            class_to_items.setdefault(c, []).append(idx)

    # An image that's the *only* one containing some class must stay in
    # train -- putting it in val would erase that class from training
    # entirely, which is worse than just leaving it out of val.
    protected = {idxs[0] for idxs in class_to_items.values() if len(idxs) == 1}
    singleton_classes = [c for c, idxs in class_to_items.items() if len(idxs) == 1]

    val_idx: set[int] = set()
    for c in sorted(class_to_items, key=lambda c: len(class_to_items[c])):
        candidates = [i for i in class_to_items[c] if i not in protected]
        if not candidates or any(i in val_idx for i in candidates):
            continue  # can't cover this class in val, or already covered

        def coverage_score(i):
            return sum(
                1 for cc in items[i]["classes"]
                if not any(j in val_idx for j in class_to_items.get(cc, []))
            )

        val_idx.add(max(candidates, key=coverage_score))

    remaining = [i for i in range(n) if i not in val_idx and i not in protected]
    rng.shuffle(remaining)
    while len(val_idx) < target_val and remaining:
        val_idx.add(remaining.pop())

    if singleton_classes:
        label = ", ".join(names[c] for c in sorted(singleton_classes))
        print(f"NOTE: only one labeled image for: {label} -- kept in train, can't be represented in val.")

    uncovered = {c for c in class_to_items if not any(i in val_idx for i in class_to_items[c])}
    unexplained = sorted(uncovered - set(singleton_classes))
    if unexplained:
        label = ", ".join(names[c] for c in unexplained)
        print(f"NOTE: still absent from val despite stratification: {label}")

    val = [items[i] for i in sorted(val_idx)]
    train = [items[i] for i in range(n) if i not in val_idx]
    return train, val


def pinned_split(items: list[dict], val_list: Path) -> tuple[list[dict], list[dict]]:
    """Split by an explicit, fixed list of "<round>:<image_stem>" val images
    instead of drawing one automatically. Everything not on the list goes to
    train. Use this once you've picked a val set you want to hold constant
    round over round, so mAP/recall comparisons stop floating with whatever
    a fresh stratified draw happens to include each time.
    """
    wanted = set()
    for line in val_list.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            wanted.add(tuple(line.split(":", 1)))

    val, train = [], []
    for item in items:
        key = (item["round"], item["img_path"].stem)
        (val if key in wanted else train).append(item)
        wanted.discard(key)

    if wanted:
        print(f"WARNING: val_list entries not found among collected images: {sorted(wanted)}")

    return train, val


def collect_round(round_dir: Path, canonical: dict[str, int]) -> list[tuple[Path, Path, dict[int, int]]]:
    """(image_path, label_path, export-class-id -> canonical-class-id map) for one round."""
    label_dir = round_dir / "obj_train_data"
    img_dir = round_dir / "swatch_images"
    label_files = sorted(label_dir.glob("*.txt"))
    if not label_files:
        raise SystemExit(f"No label files found in {label_dir}")

    class_map = load_export_class_map(round_dir / "obj.names", canonical)

    pairs = []
    for label_path in label_files:
        stem = label_path.stem
        img_path = next(
            (img_dir / f"{stem}{ext}" for ext in IMG_EXTS if (img_dir / f"{stem}{ext}").exists()),
            None,
        )
        if img_path is None:
            print(f"WARNING: [{round_dir.name}] no image found for label {label_path.name}, skipping")
            continue
        pairs.append((img_path, label_path, class_map))

    extra_images = {p.stem for p in img_dir.glob("*") if p.suffix.lower() in IMG_EXTS} - {
        lp.stem for lp in label_files
    }
    if extra_images:
        print(f"NOTE: [{round_dir.name}] images with no label file, skipped: {sorted(extra_images)}")

    return pairs


def convert(
    round_dirs: list[Path],
    stitches_csv: Path,
    out_dir: Path,
    val_frac: float = 0.15,
    seed: int = 0,
    val_list: Path | None = None,
):
    canonical = load_canonical_classes(stitches_csv)
    names = _ordered_names(stitches_csv)

    items = []  # {round, img_path, label_lines (already remapped), classes}
    for round_dir in round_dirs:
        for img_path, label_path, class_map in collect_round(round_dir, canonical):
            lines, classes = [], set()
            for line in label_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                export_cls, *coords = line.split()
                cls_id = class_map[int(export_cls)]
                lines.append(f"{cls_id} {' '.join(coords)}")
                classes.add(cls_id)
            items.append({"round": round_dir.name, "img_path": img_path, "label_lines": lines, "classes": classes})

    if not items:
        raise SystemExit("No image/label pairs matched, nothing to convert.")

    if val_list is not None:
        train_items, val_items = pinned_split(items, val_list)
    else:
        train_items, val_items = stratified_split(items, val_frac, seed, names)

    for split, split_items in [("train", train_items), ("val", val_items)]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        for item in split_items:
            out_stem = f"{item['round']}_{item['img_path'].stem}"
            shutil.copy(item["img_path"], out_dir / "images" / split / f"{out_stem}{item['img_path'].suffix}")
            (out_dir / "labels" / split / f"{out_stem}.txt").write_text("\n".join(item["label_lines"]))

        by_round = {}
        for item in split_items:
            by_round[item["round"]] = by_round.get(item["round"], 0) + 1
        breakdown = ", ".join(f"{r}={n}" for r, n in sorted(by_round.items()))
        print(f"{split}: {len(split_items)} images ({breakdown})")

    yaml_path = out_dir / "stitches.yaml"
    yaml_path.write_text(
        "path: {}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n{}\n".format(
            out_dir.resolve(),
            "\n".join(f"  {i}: {n}" for i, n in enumerate(names)),
        )
    )
    print(f"Wrote {yaml_path}")


def _ordered_names(stitches_csv: Path) -> list[str]:
    rows = {}
    with stitches_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            rows[int(row["class_id"])] = row["abbreviation"].strip()
    return [rows[i] for i in sorted(rows)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export-dir", type=Path, required=True, nargs="+",
        help="one or more round dirs (e.g. stitch_annotations/round1 stitch_annotations/round2), "
             "each with obj.names, obj_train_data/, swatch_images/",
    )
    parser.add_argument("--stitches-csv", type=Path, default=Path("stitches.csv"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val-frac", type=float, default=0.15,
                         help="ignored if --val-list is given")
    parser.add_argument("--val-list", type=Path, default=None,
                         help="file of '<round>:<image_stem>' lines -- a fixed val set, "
                              "overriding the automatic stratified split")
    args = parser.parse_args()
    convert(args.export_dir, args.stitches_csv, args.out, args.val_frac, val_list=args.val_list)
