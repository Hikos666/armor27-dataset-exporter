"""Check a generated batch without loading Blender. Does not modify the batch."""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels
import balance
import scene_groups
import distractors


def jpeg_size(path):
    data = Path(path).read_bytes()
    if data[:2] != b"\xff\xd8":
        raise ValueError(f"not a jpeg: {path}")
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            raise ValueError(f"bad jpeg marker at {i}")
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2):
            height = int.from_bytes(data[i + 5:i + 7], "big")
            width = int.from_bytes(data[i + 7:i + 9], "big")
            return width, height
        if marker in (0xD8, 0xD9):
            i += 2
            continue
        length = int.from_bytes(data[i + 2:i + 4], "big")
        i += 2 + length
    raise ValueError(f"jpeg size not found: {path}")


def check(batch, raise_on_error=True, quiet=False):
    batch = Path(batch)
    manifest = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
    issues = []
    class_counts, color_counts = Counter(), Counter()
    seen = set()
    for sample in manifest["samples"]:
        split, name = sample["split"], sample["id"]
        image = batch / "images" / split / f"{name}.jpg"
        label = batch / "labels" / split / f"{name}.txt"
        preview = batch / "preview" / f"{name}.jpg"
        key = (split, name)
        if key in seen:
            issues.append(f"duplicate {split}/{name}")
        seen.add(key)
        if not image.is_file() or not label.is_file():
            issues.append(f"missing files for {split}/{name}")
            continue
        if manifest.get("debug_images", True) and not preview.is_file():
            issues.append(f"missing preview for {name}")
            continue
        try:
            width, height = jpeg_size(image)
            if (width, height) != (1440, 1080):
                issues.append(f"{name} is {width}x{height}")
            parsed_rows = [labels.parse_row(line) for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, ValueError) as exc:
            issues.append(f"{split}/{name} invalid file: {exc}")
            continue
        plates = sample.get("plates")
        if plates is None:
            plates = [sample] if "class_id" in sample else []
        if len(parsed_rows) != len(plates):
            issues.append(f"{name} has {len(parsed_rows)} labels for {len(plates)} plates")
            continue
        for index, (parsed, plate) in enumerate(zip(parsed_rows, plates), start=1):
            if parsed["class_id"] != plate["class_id"] or parsed["color_id"] != plate["color_id"]:
                issues.append(f"{name} plate {index} label does not match manifest identity")
            if parsed["joint_id"] != plate["joint_id"]:
                issues.append(f"{name} plate {index} joint mismatch")
            geom, _, bbox = labels.geometry_issues(parsed["points"], parsed["visibility"])
            issues.extend(f"{name} plate {index} {item}" for item in geom)
            if max(abs(a - b) for a, b in zip(bbox, parsed["bbox"])) > 1e-6:
                issues.append(f"{name} plate {index} bbox does not match corners")
            for got, expected in zip(parsed["points"], plate["keypoints_normalized"]):
                if abs(got[0] - expected[0]) > 1e-6 or abs(got[1] - expected[1]) > 1e-6:
                    issues.append(f"{name} plate {index} keypoints differ from manifest")
                    break
            class_counts[plate["class_id"]] += 1
            color_counts[plate["color_id"]] += 1
    scene_splits, scene_recipes = {}, {}
    for sample in manifest["samples"]:
        if "scene_group" not in sample:
            continue
        group = sample["scene_group"]
        if group in scene_splits and scene_splits[group] != sample["split"]:
            issues.append(f"scene group {group} leaks across data splits")
        scene_splits[group] = sample["split"]
        recipe = sample.get("scene_recipe")
        if recipe:
            # Attempt counters are diagnostics, not part of scene geometry.
            geometry = (recipe["camera"], recipe["armor"], recipe["extras"], recipe.get("distractors"))
            config = manifest.get("generation_policy", {}).get("distractors")
            if config is not None:
                background = recipe.get("distractors")
                if background is None:
                    issues.append(f"{sample['id']} missing distractor recipe")
                else:
                    issues.extend(f"{sample['id']} {issue}" for issue in distractors.recipe_issues(background, config))
            if group in scene_recipes and not scene_groups.same_geometry(scene_recipes[group], geometry):
                issues.append(f"scene group {group} has inconsistent geometry")
            scene_recipes[group] = geometry
    plan = manifest.get("balance_plan")
    if plan:
        try:
            balance.verify_prefix(plan, manifest["samples"])
        except ValueError as exc:
            issues.append(str(exc))
        if manifest.get("status") == "passed":
            if len(manifest["samples"]) != len(plan["image_counts"]):
                issues.append("batch is incomplete for its class quota plan")
            for class_id, expected in plan["quotas"].items():
                if class_counts[int(class_id)] != expected:
                    issues.append(f"class {class_id} count {class_counts[int(class_id)]} != quota {expected}")
    report = {"batch": str(batch), "samples": len(manifest["samples"]),
              "class_counts": dict(class_counts), "color_counts": dict(color_counts),
              "issues": issues, "status": "passed" if not issues else "failed"}
    if not quiet:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if issues and raise_on_error:
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    check(sys.argv[1])
