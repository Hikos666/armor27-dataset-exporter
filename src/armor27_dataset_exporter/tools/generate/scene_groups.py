"""Pure metadata helpers for grouped ID swaps and conditional coverage audits."""

import math
from collections import Counter


def split_groups(group_ids, train, val, test, seed):
    import random
    import labels
    unique = list(dict.fromkeys(group_ids))
    shuffled = list(unique)
    random.Random(seed).shuffle(shuffled)
    assignments = labels.split_plan(len(unique), train, val, test)
    mapping = dict(zip(shuffled, assignments))
    return [mapping[group] for group in group_ids]


def buckets(points, width, height, angles, area, visibility, color):
    # Equivalent square side of the projected LED area; less sensitive to roll than bbox span.
    signed = sum(points[i][0] * points[(i+1) % 4][1] -
                 points[(i+1) % 4][0] * points[i][1] for i in range(4)) / 2
    scale = math.sqrt(abs(signed) * width * height)
    yaw = abs(angles[0])
    occlusion = max(0, 1 - area)
    return {
        "color_id": color, "size_px": scale,
        "size": "small" if scale < 60 else "medium" if scale < 160 else "large",
        "pose": "frontal" if yaw < 20 else "oblique" if yaw < 60 else "side",
        "occlusion": "none" if occlusion <= 0.02 and all(v == 2 for v in visibility)
                     else "light" if occlusion <= 0.2 else "partial",
        # Visibility and sampling settings alone do not establish blur/moire appearance.
        "image_quality": "unverified",
    }


def coverage(samples, classes):
    result = {str(c): {key: {} for key in ("color_id", "size", "pose", "occlusion", "image_quality")}
              for c in classes}
    for sample in samples:
        for plate in sample.get("plates", [sample]):
            row = result[str(plate["class_id"])]
            for key in row:
                value = str(plate.get("buckets", {}).get(key, "unrecorded"))
                row[key][value] = row[key].get(value, 0) + 1
    return result


def same_geometry(a, b, tolerance=1e-6):
    """Blender decomposes/rebuilds matrix_world in float32 when restoring a pose."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_geometry(a[k], b[k], tolerance) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same_geometry(x, y, tolerance) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=0, abs_tol=tolerance)
    return a == b


def missing_buckets(report):
    expected = {"color_id": ("0", "1", "3"), "size": ("small", "medium", "large"),
                "pose": ("frontal", "oblique", "side"),
                "occlusion": ("none", "light", "partial"),
                "image_quality": ("clear", "slight_blur", "visible_moire")}
    return {c: {key: [value for value in values if not buckets[key].get(value)]
                for key, values in expected.items()}
            for c, buckets in report.items()}
