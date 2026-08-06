"""
Turn a flat, unordered list of stitch detections (box + class) into
row-ordered, pattern-style instructions -- including geometric inference of
increases/decreases, without needing dedicated inc/dec detection classes.

Approach:
1. Cluster detections into rows by vertical (y) position.
2. Within each row, sort left-to-right (or right-to-left on alternating rows).
3. Flag "tight pairs": two same-class stitches sitting closer together (in x)
   than the row's typical stitch spacing -- these are candidates for a shared
   base (increase) or a converging top (decrease). We can't tell which from
   spacing alone, so direction is inferred from whether the row's total
   stitch count went up or down relative to the previous row.
4. Merge each tight pair into a single "<class>_inc" / "<class>_dec" token.
5. Run-length encode the resulting token sequence into pattern text.

This is a heuristic, not a learned model -- cheap to tune by eye, and keeps
inc/dec logic decoupled from the detector, which only needs to classify base
stitch type. If this heuristic produces noisy calls on a particular stitch
type (e.g. dense compound stitches where base spacing is hard to infer),
that's the signal to fall back to explicit inc/dec detection classes for
just that stitch type, rather than assuming this works universally.
"""
from dataclasses import dataclass


@dataclass
class Detection:
    cls_name: str
    x_center: float
    y_center: float
    confidence: float = 1.0


def cluster_rows(detections: list[Detection], row_gap_frac: float = 0.6) -> list[list[Detection]]:
    """Group detections into rows by y-center, using the median stitch height
    (approximated here via the spread of y-centers) as the clustering gap.
    """
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: d.y_center)
    y_vals = [d.y_center for d in sorted_dets]
    gaps = [y_vals[i + 1] - y_vals[i] for i in range(len(y_vals) - 1)]
    nonzero_gaps = [g for g in gaps if g > 0]
    threshold = (sorted(nonzero_gaps)[len(nonzero_gaps) // 2] if nonzero_gaps else 1) * (1 / row_gap_frac)

    rows: list[list[Detection]] = [[sorted_dets[0]]]
    for prev, curr in zip(sorted_dets, sorted_dets[1:]):
        if curr.y_center - prev.y_center > threshold:
            rows.append([])
        rows[-1].append(curr)

    return rows


def order_row(row: list[Detection], reverse: bool) -> list[Detection]:
    return sorted(row, key=lambda d: d.x_center, reverse=reverse)


def find_tight_pairs(ordered_row: list[Detection], tightness_frac: float = 0.5) -> list[tuple[int, int]]:
    """Return index-pairs of adjacent, same-class stitches whose x-spacing is
    notably smaller than the row's typical spacing -- inc/dec candidates.
    Needs at least 3 stitches in the row to establish a "typical" spacing.
    """
    if len(ordered_row) < 3:
        return []

    # Use absolute distance -- ordered_row may be sorted right-to-left on
    # alternating rows, which would otherwise make raw differences negative
    # and silently break the "< threshold" comparison below.
    spacings = [abs(ordered_row[i + 1].x_center - ordered_row[i].x_center) for i in range(len(ordered_row) - 1)]
    median_spacing = sorted(spacings)[len(spacings) // 2]
    if median_spacing <= 0:
        return []
    threshold = median_spacing * tightness_frac

    pairs = []
    i = 0
    while i < len(ordered_row) - 1:
        gap = abs(ordered_row[i + 1].x_center - ordered_row[i].x_center)
        same_class = ordered_row[i].cls_name == ordered_row[i + 1].cls_name
        if gap < threshold and same_class:
            pairs.append((i, i + 1))
            i += 2  # don't let a stitch participate in two pairs
        else:
            i += 1
    return pairs


def tokens_for_row(ordered_row: list[Detection], direction: str | None) -> list[str]:
    """Collapse tight pairs into a single inc/dec token; leave other stitches
    as-is. `direction` is None (no inc/dec inference this row -- not enough
    context yet), "inc", or "dec".
    """
    if not ordered_row:
        return []

    pairs = find_tight_pairs(ordered_row)
    pair_starts = {p[0] for p in pairs}

    tokens = []
    i = 0
    while i < len(ordered_row):
        if i in pair_starts and direction is not None:
            tokens.append(f"{ordered_row[i].cls_name}_{direction}")
            i += 2
        else:
            tokens.append(ordered_row[i].cls_name)
            i += 1
    return tokens


def run_length_encode(tokens: list[str]) -> str:
    if not tokens:
        return ""
    parts = []
    current = tokens[0]
    count = 1
    for tok in tokens[1:]:
        if tok == current:
            count += 1
        else:
            parts.append(f"{count} {current}")
            current = tok
            count = 1
    parts.append(f"{count} {current}")
    return ", ".join(parts)


def reconstruct_instructions(
    detections: list[Detection],
    alternate_direction: bool = True,
) -> list[str]:
    """
    alternate_direction=True mimics turned crochet rows (left-to-right, then
    right-to-left, ...). Set False for swatches always worked/photographed in
    one direction (e.g. flattened in-the-round pieces).

    Increase/decrease direction per row is inferred by comparing this row's
    raw stitch count (post pairing, pre-merge) to the previous row's final
    (merged) count -- more raw stitches than the previous row's total implies
    increases; fewer implies decreases. The very first row has no prior row
    to compare against, so tight pairs there are left un-merged.
    """
    rows = cluster_rows(detections)
    instructions = []
    prev_final_count: int | None = None

    for i, row in enumerate(rows):
        reverse = alternate_direction and (i % 2 == 1)
        ordered = order_row(row, reverse=reverse)

        direction = None
        if prev_final_count is not None:
            if len(ordered) > prev_final_count:
                direction = "inc"
            elif len(ordered) < prev_final_count:
                direction = "dec"

        tokens = tokens_for_row(ordered, direction)
        instructions.append(f"Row {i + 1}: {run_length_encode(tokens)} ({len(tokens)} sts)")
        prev_final_count = len(tokens)

    return instructions


if __name__ == "__main__":
    # Row 1: 3 sc, no prior row to compare against.
    # Row 2: 4 raw stitches with one tight pair -> more sts than row 1 (3)
    #        -> inferred as an increase, merged into "sc_inc".
    demo = [
        Detection("sc", x_center=10, y_center=10),
        Detection("sc", x_center=30, y_center=11),
        Detection("sc", x_center=50, y_center=9),

        Detection("sc", x_center=10, y_center=40),
        Detection("sc", x_center=29, y_center=41),   # tight pair with next
        Detection("sc", x_center=32, y_center=41),   # (shared-base increase)
        Detection("sc", x_center=52, y_center=39),
    ]
    for line in reconstruct_instructions(demo):
        print(line)