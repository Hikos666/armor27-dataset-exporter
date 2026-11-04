"""Schema v2 labels. Class and color are passed in; this module never parses material names."""

FLIP_INDEX = (3, 2, 1, 0)


def joint_id(color_id, class_id):
    return color_id * 9 + class_id


def decode_joint(value):
    return value // 9, value % 9


def bbox_from_points(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    low_x, high_x = min(xs), max(xs)
    low_y, high_y = min(ys), max(ys)
    return [(low_x + high_x) / 2, (low_y + high_y) / 2, high_x - low_x, high_y - low_y]


def horizontal_flip(points, visibility):
    """Mirror x and apply flip_idx so the order stays LT, LB, RB, RT in the new image."""
    mirrored = [(1.0 - x, y) for x, y in points]
    return [mirrored[i] for i in FLIP_INDEX], [visibility[i] for i in FLIP_INDEX]


def format_row(color_id, class_id, points, visibility):
    if len(points) != 4 or len(visibility) != 4:
        raise ValueError("a plate has four corners")
    fields = [str(joint_id(color_id, class_id))]
    fields.extend(f"{v:.8f}" for v in bbox_from_points(points))
    for (x, y), flag in zip(points, visibility):
        fields.extend((f"{x:.8f}", f"{y:.8f}", str(int(flag))))
    if len(fields) != 17:
        raise ValueError("label row must have 17 fields")
    return " ".join(fields)


def parse_row(line):
    parts = line.split()
    if len(parts) != 17:
        raise ValueError(f"expected 17 fields, got {len(parts)}")
    joint = int(parts[0])
    bbox = [float(v) for v in parts[1:5]]
    points, visibility = [], []
    for i in range(4):
        base = 5 + i * 3
        points.append((float(parts[base]), float(parts[base + 1])))
        visibility.append(int(parts[base + 2]))
    return {"joint_id": joint, "color_id": joint // 9, "class_id": joint % 9,
            "bbox": bbox, "points": points, "visibility": visibility}


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_cross(a, b, c, d):
    def side(v):
        if abs(v) <= 1e-12:
            return 0
        return 1 if v > 0 else -1
    return side(_cross(a, b, c)) * side(_cross(a, b, d)) < 0 and side(_cross(c, d, a)) * side(_cross(c, d, b)) < 0


def geometry_issues(points, visibility, minimum_area=1e-5):
    issues = []
    if any(v not in (0, 1, 2) for v in visibility):
        issues.append("bad_visibility")
    if any(not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) for x, y in points):
        issues.append("outside_unit_square")
    if _segments_cross(points[0], points[1], points[2], points[3]) or _segments_cross(points[1], points[2], points[3], points[0]):
        issues.append("self_intersection")
    area = abs(sum(_cross(points[0], points[i], points[(i + 1) % 4]) for i in range(1, 4))) / 2
    if area < minimum_area:
        issues.append("area_too_small")
    left_x = (points[0][0] + points[1][0]) / 2
    right_x = (points[3][0] + points[2][0]) / 2
    top_y = (points[0][1] + points[3][1]) / 2
    bottom_y = (points[1][1] + points[2][1]) / 2
    if left_x >= right_x:
        issues.append("left_right_swapped")
    if top_y >= bottom_y:
        issues.append("up_down_swapped")
    expected = bbox_from_points(points)
    return issues, area, expected


def needs_moire_span(in_band, in_band_count, sample_count, target, minimum_fraction):
    """True when this outside-band pose must be skipped to leave room for the moiré quota."""
    if target <= 0:
        return False
    need = int(minimum_fraction * target)
    if minimum_fraction * target > need:
        need += 1
    slots_left = target - sample_count
    return (not in_band) and (need - in_band_count) >= slots_left


def plate_count(unit, low, high, distribution):
    """Pick how many plates share one image. unit is in [0, 1)."""
    low, high = int(low), int(high)
    if high < low:
        low, high = high, low
    counts = list(range(low, high + 1))
    if distribution == "more":
        weights = [count - low + 1 for count in counts]
    elif distribution == "fewer":
        weights = [high - count + 1 for count in counts]
    else:
        weights = [1 for _ in counts]
    pick = float(unit) * sum(weights)
    for count, weight in zip(counts, weights):
        pick -= weight
        if pick <= 0:
            return count
    return counts[-1]


def choose_entry(entries, used_classes, index):
    """Cover each available class once before repeating. Purple is the caller's job to exclude."""
    if not entries:
        raise ValueError("no trainable entries")
    by_class = {}
    for entry in entries:
        by_class.setdefault(entry["class_id"], []).append(entry)
    missing = [class_id for class_id in sorted(by_class) if class_id not in used_classes]
    pool = by_class[missing[0]] if missing else entries
    return pool[index % len(pool)]


def split_plan(count, train_ratio, val_ratio, test_ratio):
    """Largest-remainder assignment of count images to train, val, and test."""
    weights = [float(train_ratio), float(val_ratio), float(test_ratio)]
    total = sum(weights)
    if count < 1:
        raise ValueError("count must be positive")
    if total <= 0:
        raise ValueError("split ratios must be positive")
    weights = [w / total for w in weights]
    raw = [count * w for w in weights]
    counts = [int(v) for v in raw]
    leftover = count - sum(counts)
    order = sorted(range(3), key=lambda i: (raw[i] - counts[i], -i), reverse=True)
    for i in range(leftover):
        counts[order[i]] += 1
    names = ("train", "val", "test")
    plan = []
    for name, n in zip(names, counts):
        plan.extend([name] * n)
    return plan


SPLIT_START = {"train": 1, "val": 101, "test": 201}


def sample_id(split, already):
    return f"{SPLIT_START[split] + already:06d}"
