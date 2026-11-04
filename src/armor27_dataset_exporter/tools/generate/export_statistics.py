"""Export one batch to UTF-8-BOM CSV tables and JSON, without Blender or Excel.

Labels are authoritative for object counts. Missing historical metadata stays null.
Only statistics/ is written; images, labels and the manifest are never modified.
"""

import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_dataset
import labels

CLASS_NAMES = ("哨兵", "1", "2", "3", "4", "5", "前哨", "基地", "未知")
COLOR_NAMES = ("蓝", "红", "紫", "无")
CATEGORIES = {
    "color_id": (0, 1, 3), "size": ("small", "medium", "large"),
    "pose": ("frontal", "oblique", "side"), "occlusion": ("none", "light", "partial"),
    "image_quality": ("clear", "slight_blur", "significant_blur", "visible_moire", "unverified"),
}
NUMERIC_BINS = {
    "distance_m": (0.1, 0.5, 1, 2, 4, 8.000001),
    "object_bottom_height_m": (0.05, 0.15, 0.25, 0.350001),
    "camera_height_m": (0.25, 0.4, 0.6, 0.800001),
    "projected_size_px": (0, 60, 160, 400, math.inf),
    "visible_area_fraction": (0, 0.3, 0.5, 0.8, 0.98, 1.000001),
    "center_x": (0, 0.333333, 0.666667, 1.000001),
    "center_y": (0, 0.333333, 0.666667, 1.000001),
}
OBJECT_FIELDS = [
    "image_id", "split", "scene_group", "plate_index", "label_valid", "metadata_present",
    "class_id", "class_name", "color_id", "color_name", "joint_id", "material", "base_material",
    "region", "distance_m", "object_bottom_height_m", "camera_height_m", "yaw_deg", "pitch_deg", "roll_deg",
    "visible_area_fraction", "visibility_combo", "visible_corners", "occluded_corners",
    "projected_size_px", "size", "pose", "occlusion", "image_quality", "center_x", "center_y",
    "bbox_cx", "bbox_cy", "bbox_w", "bbox_h",
] + [f"{name}_{axis}" for name in ("lt", "lb", "rb", "rt") for axis in ("x", "y", "v")]
IMAGE_FIELDS = [
    "image_id", "split", "scene_group", "image_path", "label_path", "preview_path",
    "image_exists", "label_exists", "width", "height", "manifest_plate_count", "label_line_count",
    "valid_label_count", "flipped", "glare_present", "glare_color", "light_energy_scale", "noise_sigma",
    "render_samples", "denoiser", "render_seconds", "export_seconds", "motion_blur",
    "motion_blur_probability", "motion_exposure_s", "motion_speed_m_s", "motion_max_displacement_px",
    "attempt", "camera_height_m",
]


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def name_at(names, index):
    return names[index] if isinstance(index, int) and 0 <= index < len(names) else None


def serial(value):
    if value is None:
        return "未记录"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def write_csv(path, fields, rows):
    temporary = path.with_suffix(f".csv.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serial(row.get(key)) for key in fields})
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_bucket(value, edges):
    if not finite(value):
        return "unrecorded"
    for low, high in zip(edges, edges[1:]):
        if low <= value < high:
            return f"[{low:g},{high:g})"
    return "outside_range"


def metadata_for(sample, index, plate):
    recipe = sample.get("scene_recipe") or {}
    extras = recipe.get("extras") or []
    fallback = sample if index == 0 else (extras[index - 1].get("record", {}) if index <= len(extras) else {})
    def get(key):
        return plate.get(key, fallback.get(key))
    angles = get("yaw_pitch_roll_deg") or fallback.get("measured_yaw_pitch_roll_deg")
    return {
        "region": get("region"), "distance_m": get("distance_m") or fallback.get("actual_distance_m"),
        "object_bottom_height_m": get("object_bottom_height_m"),
        "camera_height_m": sample.get("camera_height_m"),
        "visible_area_fraction": get("visible_area_fraction"),
        "yaw_deg": angles[0] if angles else None,
        "pitch_deg": angles[1] if angles else None,
        "roll_deg": angles[2] if angles else None,
    }


def object_row(sample, index, plate, parsed, width, height):
    row = {"image_id": sample["id"], "split": sample["split"], "scene_group": sample.get("scene_group"),
           "plate_index": index + 1, "label_valid": parsed is not None, "metadata_present": bool(plate)}
    row.update(metadata_for(sample, index, plate))
    row.update({key: plate.get(key) for key in ("material", "base_material")})
    buckets = plate.get("buckets") or {}
    row.update({key: buckets.get(key) for key in ("size", "pose", "occlusion", "image_quality")})
    row["image_quality"] = row["image_quality"] or "unverified"
    if parsed is None:
        return row
    row.update({key: parsed[key] for key in ("class_id", "color_id", "joint_id")})
    row["class_name"] = name_at(CLASS_NAMES, row["class_id"])
    row["color_name"] = name_at(COLOR_NAMES, row["color_id"])
    points, flags = parsed["points"], parsed["visibility"]
    for point, flag, name in zip(points, flags, ("lt", "lb", "rb", "rt")):
        row.update({f"{name}_x": point[0], f"{name}_y": point[1], f"{name}_v": flag})
    for field, value in zip(("bbox_cx", "bbox_cy", "bbox_w", "bbox_h"), parsed["bbox"]):
        row[field] = value
    bbox = labels.bbox_from_points(points)
    row.update(center_x=bbox[0], center_y=bbox[1], visibility_combo="-".join(map(str, flags)),
               visible_corners=flags.count(2), occluded_corners=flags.count(1))
    if width and height:
        area = abs(sum(points[i][0] * points[(i+1) % 4][1] -
                       points[(i+1) % 4][0] * points[i][1] for i in range(4))) / 2
        size = math.sqrt(area * width * height)
        row["projected_size_px"] = size
        row["size"] = "small" if size < 60 else "medium" if size < 160 else "large"
    if row["pose"] is None and finite(row["yaw_deg"]):
        yaw = abs(row["yaw_deg"])
        row["pose"] = "frontal" if yaw < 20 else "oblique" if yaw < 60 else "side"
    if row["occlusion"] is None and finite(row["visible_area_fraction"]):
        hidden = max(0, 1 - row["visible_area_fraction"])
        row["occlusion"] = "none" if hidden <= .02 and all(v == 2 for v in flags) else "light" if hidden <= .2 else "partial"
    return row


def collect(batch, manifest):
    images, objects, checks = [], [], []
    expected_images, expected_labels = set(), set()
    for sample in manifest["samples"]:
        name, split = sample["id"], sample["split"]
        image = batch / "images" / split / f"{name}.jpg"
        label = batch / "labels" / split / f"{name}.txt"
        preview = batch / "preview" / f"{name}.jpg"
        expected_images.add(image.relative_to(batch).as_posix())
        expected_labels.add(label.relative_to(batch).as_posix())
        width = height = None
        if image.is_file():
            try:
                width, height = check_dataset.jpeg_size(image)
            except (ValueError, OSError) as exc:
                checks.append({"category": "invalid_image", "detail": f"{split}/{name}: {exc}", "count": 1})
        plates = sample.get("plates")
        if plates is None:
            plates = [sample] if "class_id" in sample else []
        lines = label.read_text(encoding="utf-8").splitlines() if label.is_file() else []
        lines = [line for line in lines if line.strip()]
        parsed_rows = []
        for number, line in enumerate(lines, start=1):
            try:
                parsed = labels.parse_row(line)
                values = [*parsed["bbox"], *[v for point in parsed["points"] for v in point]]
                if not all(finite(v) for v in values):
                    raise ValueError("non-finite coordinates")
                if not 0 <= parsed["joint_id"] <= 35 or any(v not in (0, 1, 2) for v in parsed["visibility"]):
                    raise ValueError("invalid identity or visibility")
            except ValueError as exc:
                checks.append({"category": "invalid_label", "detail": f"{split}/{name} line {number}: {exc}", "count": 1})
                parsed = None
            parsed_rows.append(parsed)
        for index in range(max(len(plates), len(parsed_rows))):
            objects.append(object_row(sample, index, plates[index] if index < len(plates) else {},
                                      parsed_rows[index] if index < len(parsed_rows) else None, width, height))
        glare = sample.get("glare")
        images.append({
            "image_id": name, "split": split, "scene_group": sample.get("scene_group"),
            "image_path": image.relative_to(batch).as_posix(), "label_path": label.relative_to(batch).as_posix(),
            "preview_path": preview.relative_to(batch).as_posix(),
            "image_exists": image.is_file(), "label_exists": label.is_file(), "width": width, "height": height,
            "manifest_plate_count": len(plates), "label_line_count": len(lines),
            "valid_label_count": sum(p is not None for p in parsed_rows),
            "flipped": sample.get("flipped"), "glare_present": bool(glare) if "glare" in sample else None,
            "glare_color": glare.get("name") if glare else None,
            "light_energy_scale": sample.get("light_energy_scale"), "noise_sigma": sample.get("noise_sigma"),
            "render_samples": manifest.get("render_samples"), "denoiser": manifest.get("denoiser"),
            "render_seconds": sample.get("render_seconds"), "export_seconds": sample.get("export_seconds"),
            "motion_blur": sample.get("motion_blur"),
            "motion_blur_probability": sample.get("motion_blur_probability"),
            "motion_exposure_s": sample.get("motion_exposure_s"),
            "motion_speed_m_s": sample.get("motion_speed_m_s"),
            "motion_max_displacement_px": sample.get("motion_max_displacement_px"),
            "attempt": sample.get("attempt"),
            "camera_height_m": sample.get("camera_height_m"),
        })
    for folder, suffix, expected in (("images", "*.jpg", expected_images), ("labels", "*.txt", expected_labels)):
        for path in (batch / folder).glob(f"*/{suffix}"):
            relative = path.relative_to(batch).as_posix()
            if relative not in expected:
                checks.append({"category": "untracked_file", "detail": relative, "count": 1})
    return images, objects, checks


def export(batch):
    batch = Path(batch).resolve()
    manifest_path = batch / "manifest.json"
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    images, objects, checks = collect(batch, manifest)
    valid = [row for row in objects if row["label_valid"]]
    try:
        validation = check_dataset.check(batch, raise_on_error=False, quiet=True)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        validation = {"status": "failed", "issues": [f"checker error: {type(exc).__name__}: {exc}"]}
    checks.extend({"category": "quality_check", "detail": issue, "count": 1} for issue in validation["issues"])
    if not validation["issues"]:
        checks.append({"category": "quality_check", "detail": "passed", "count": 0})
    for entry in checks:
        if entry["category"] in ("invalid_image", "invalid_label", "untracked_file"):
            validation["issues"].append(entry["detail"])
    if validation["issues"]:
        validation["status"] = "failed"
    if manifest_path.read_bytes() != raw:
        raise RuntimeError("批次仍在写入，统计未导出；请在停止或完成后重试")
    class_ids = sorted(set((manifest.get("balance_plan") or {}).get("classes", [])) |
                       {row["class_id"] for row in valid})
    class_counts = Counter(row["class_id"] for row in valid)
    color_counts = Counter(row["color_id"] for row in valid)
    quotas = (manifest.get("balance_plan") or {}).get("quotas", {})
    classes = [{"class_id": c, "class_name": name_at(CLASS_NAMES, c), "count": class_counts[c],
                "fraction": class_counts[c] / len(valid) if valid else 0,
                "quota": quotas.get(str(c)),
                "difference": class_counts[c] - quotas[str(c)] if str(c) in quotas else None}
               for c in class_ids]
    colors = [{"color_id": c, "color_name": name_at(COLOR_NAMES, c), "count": color_counts[c],
               "fraction": color_counts[c] / len(valid) if valid else 0} for c in range(4)]
    splits = []
    for split in ("train", "val", "test"):
        subset = [row for row in images if row["split"] == split]
        splits.append({"split": split, "images": len(subset),
                       "objects": sum(row["valid_label_count"] for row in subset),
                       "scene_groups": len({row["scene_group"] for row in subset if row["scene_group"] is not None})})
    count_hist = Counter(row["valid_label_count"] for row in images)
    per_image = [{"plates": n, "images": count, "fraction": count / len(images) if images else 0}
                 for n, count in sorted(count_hist.items())]
    groups = defaultdict(list)
    for row in images:
        if row["scene_group"] is not None:
            groups[row["scene_group"]].append(row)
    group_rows = [{"scene_group": key, "images": len(rows),
                   "objects": sum(row["valid_label_count"] for row in rows),
                   "splits": sorted({row["split"] for row in rows}),
                   "leakage": len({row["split"] for row in rows}) > 1,
                   "image_keys": [f"{row['split']}/{row['image_id']}" for row in rows]}
                  for key, rows in groups.items()]
    cross, distributions = [], []
    dimensions = [*CATEGORIES, "split", "region", "visibility_combo"]
    for dimension in dimensions:
        counts = Counter(str(row.get(dimension) if row.get(dimension) is not None else "unrecorded") for row in valid)
        for bucket, count in sorted(counts.items()):
            distributions.append({"dimension": dimension, "bucket": bucket, "count": count,
                                  "denominator": len(valid), "fraction": count / len(valid) if valid else 0})
        for c in class_ids:
            subset = [row for row in valid if row["class_id"] == c]
            values = Counter(str(row.get(dimension) if row.get(dimension) is not None else "unrecorded") for row in subset)
            expected = {str(v) for v in CATEGORIES.get(dimension, ())}
            for bucket in sorted(set(values) | expected):
                cross.append({"class_id": c, "class_name": name_at(CLASS_NAMES, c), "dimension": dimension,
                              "bucket": bucket, "count": values[bucket], "class_total": len(subset),
                              "class_fraction": values[bucket] / len(subset) if subset else 0})
    numeric_summary = {}
    for field, edges in NUMERIC_BINS.items():
        values = [row[field] for row in valid if finite(row.get(field))]
        numeric_summary[field] = {"recorded": len(values), "missing": len(valid) - len(values),
                                  "min": min(values) if values else None, "max": max(values) if values else None,
                                  "mean": statistics.mean(values) if values else None,
                                  "median": statistics.median(values) if values else None}
        for c in class_ids:
            subset = [row for row in valid if row["class_id"] == c]
            per_class = Counter(metric_bucket(row.get(field), edges) for row in subset)
            for bucket, count in sorted(per_class.items()):
                cross.append({"class_id": c, "class_name": name_at(CLASS_NAMES, c), "dimension": field,
                              "bucket": bucket, "count": count, "class_total": len(subset),
                              "class_fraction": count / len(subset) if subset else 0})
        counts = Counter(metric_bucket(row.get(field), edges) for row in valid)
        for bucket, count in sorted(counts.items()):
            distributions.append({"dimension": field, "bucket": bucket, "count": count,
                                  "denominator": len(valid), "fraction": count / len(valid) if valid else 0})
    # Actual event counts, not merely configured probabilities.
    for row in distributions:
        row["unit"] = "object"
    for dimension in ("flipped", "glare_present", "glare_color", "motion_blur"):
        counts = Counter(str(row.get(dimension)) if row.get(dimension) is not None else "unrecorded" for row in images)
        for bucket, count in sorted(counts.items()):
            distributions.append({"dimension": dimension, "bucket": bucket, "count": count,
                                  "denominator": len(images), "fraction": count / len(images) if images else 0,
                                  "unit": "image"})
    catalog = json.loads(Path(__file__).with_name("catalog.json").read_text(encoding="utf-8"))
    available = {(e["class_id"], e["color_id"]) for e in manifest.get("asset_catalog_entries", catalog["entries"])}
    gaps = [{"class_id": row["class_id"], "class_name": row["class_name"],
             "dimension": row["dimension"], "bucket": row["bucket"],
             "reason": "missing_color_asset" if row["dimension"] == "color_id" and
             (row["class_id"], int(row["bucket"])) not in available else
             "not_verified" if row["dimension"] == "image_quality" else "no_exported_objects"}
            for row in cross if row["dimension"] in CATEGORIES and row["count"] == 0]
    rejections = manifest.get("rejected") or []
    reject_reasons = Counter(reason for item in rejections for reason in item.get("reasons", []))
    checks.extend({"category": "rejection_reason", "detail": reason, "count": n} for reason, n in sorted(reject_reasons.items()))
    checks.append({"category": "render_errors", "detail": "separate from rejected candidates",
                   "count": manifest.get("render_error_count")})
    log_path = batch / "generation.log"
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if " ERROR " in line or " STATISTICS_ERROR " in line:
                checks.append({"category": "program_error_log", "detail": line, "count": 1})
    target = len(manifest.get("split_plan") or []) or len((manifest.get("balance_plan") or {}).get("image_counts", [])) or None
    summary = {
        "generation_id": manifest.get("generation_id", batch.name), "batch_status": manifest.get("status"),
        "statistics_generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "started_at": manifest.get("started_at"), "finished_at": manifest.get("finished_at"),
        "elapsed_seconds": manifest.get("elapsed_seconds"), "active_seconds": manifest.get("active_seconds"),
        "target_images": target, "images": len(images), "valid_label_objects": len(valid),
        "manifest_objects": sum(row["manifest_plate_count"] for row in images),
        "label_lines": sum(row["label_line_count"] for row in images),
        "scene_groups_recorded": len(groups), "scene_groups_missing": sum(row["scene_group"] is None for row in images),
        "minimum_objects_per_image": min(count_hist) if count_hist else None,
        "maximum_objects_per_image": max(count_hist) if count_hist else None,
        "mean_objects_per_image": len(valid) / len(images) if images else None,
        "attempts": manifest.get("attempts"), "rejected_candidates": len(rejections),
        "render_errors": manifest.get("render_error_count"), "legacy_combined_error_count": manifest.get("error_count"),
        "rejection_fraction": len(rejections) / manifest["attempts"] if manifest.get("attempts") else None,
        "lamda": (manifest.get("balance_plan") or {}).get("lamda"),
        "quotas_met": all(row["difference"] == 0 for row in classes) if quotas else None,
        "complete": manifest.get("status") == "passed" and target is not None and len(images) == target,
        "quality_check": validation["status"], "issue_count": sum(row["count"] for row in checks if row["category"] in ("quality_check", "invalid_image", "invalid_label", "untracked_file")),
        "source_sha256": manifest.get("source_sha256"), "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "source_unchanged": manifest.get("source_unchanged"), "blender_version": manifest.get("blender_version"),
        "blenderproc_version": manifest.get("blenderproc_version"), "resolution": manifest.get("resolution"),
        "render_samples": manifest.get("render_samples"), "denoiser": manifest.get("denoiser"),
        "catalog_schema": manifest.get("catalog_schema"), "augment_schema": manifest.get("augment_schema"),
        "sampling_schema": manifest.get("sampling_schema"),
        "random_seed": manifest.get("random_seed"), "class_filter": manifest.get("class_filter"),
        "color_filter": manifest.get("color_filter"), "split_ratios": manifest.get("split_ratios"),
        "distribution_options": manifest.get("distribution_options"),
        "augmentation_settings": manifest.get("augment"),
    }
    summary_rows = [{"metric": key, "value": value} for key, value in summary.items()]
    for field, measures in numeric_summary.items():
        summary_rows.extend({"metric": f"{field}.{key}", "value": value} for key, value in measures.items())
    tables = {
        "summary": (["metric", "value"], summary_rows),
        "splits": (["split", "images", "objects", "scene_groups"], splits),
        "classes": (["class_id", "class_name", "count", "fraction", "quota", "difference"], classes),
        "colors": (["color_id", "color_name", "count", "fraction"], colors),
        "objects_per_image": (["plates", "images", "fraction"], per_image),
        "distributions": (["dimension", "bucket", "count", "denominator", "fraction", "unit"], distributions),
        "cross_tables": (["class_id", "class_name", "dimension", "bucket", "count", "class_total", "class_fraction"], cross),
        "scene_groups": (["scene_group", "images", "objects", "splits", "leakage", "image_keys"], group_rows),
        "coverage_gaps": (["class_id", "class_name", "dimension", "bucket", "reason"], gaps),
        "checks": (["category", "detail", "count"], checks),
        "images": (IMAGE_FIELDS, images), "objects": (OBJECT_FIELDS, objects),
    }
    output = batch / "statistics"
    output.mkdir(exist_ok=True)
    for table, (fields, rows) in tables.items():
        write_csv(output / f"{table}.csv", fields, rows)
    report = {"schema_version": 1, "summary": summary, "numeric_summary": numeric_summary,
              "validation": validation, "configuration": {k: v for k, v in manifest.items() if k not in ("samples", "scene_recipe")},
              "tables": {name: rows for name, (_, rows) in tables.items()},
              "notes": ["Object totals come from valid parsed TXT rows; missing metadata is null / 未记录.",
                        "File/geometry checks do not verify visual moire, blur or training quality.",
                        "Incomplete batches are not claimed balanced. Export does not change source data."]}
    temporary = output / f"report.json.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output / "report.json")
    print(f"STATISTICS_EXPORTED {output} images={len(images)} objects={len(valid)} quality={validation['status']}")
    return report


if __name__ == "__main__":
    export(sys.argv[1])
