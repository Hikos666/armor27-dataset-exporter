"""Basic dataset batch: pose, explicit material identity, light scale, glare, flip, noise.

Run with Blender's Python. Does not save the source blend and does not write into data/images.
Vertical flip is intentionally absent.
"""

import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = Path(os.environ.get("ARMOR27_ASSET_ROOT", ROOT)).expanduser().resolve()
DEFAULT_OUTPUT_ROOT = Path(os.environ.get(
    "ARMOR27_OUTPUT_ROOT", Path.home() / "New_space" / "armor27_dataset" / "batches"
)).expanduser().resolve()
sys.path.insert(0, str(ROOT / "tools" / "placement"))
sys.path.insert(0, str(ROOT / "tools" / "generate"))
sys.path.insert(0, str(ROOT / "tools" / "occlusion_validation"))
import constraints
import labels
import balance
import scene_groups
import motion
import distractors
import sample_field as placement
from visibility import evaluate_visibility, retention_reasons

CATALOG = json.loads((Path(__file__).with_name("catalog.json")).read_text(encoding="utf-8"))
AUGMENT = json.loads((Path(__file__).with_name("augment.json")).read_text(encoding="utf-8"))
BINDINGS = json.loads((ROOT / "tools/asset_validation/asset_bindings.json").read_text(encoding="utf-8"))
SOURCE = ASSET_ROOT / BINDINGS["source_blend"]
CAMERA_PROFILE = json.loads((ROOT / "tools/motion_validation/camera_profile.json").read_text(encoding="utf-8"))


def choose_weighted(rng, items):
    total = sum(item["weight"] for item in items)
    pick = rng.random() * total
    for item in items:
        pick -= item["weight"]
        if pick <= 0:
            return item
    return items[-1]


def tint_material(base_name, runtime_name, tint):
    existing = bpy.data.materials.get(runtime_name)
    if existing is not None:
        return existing
    base = bpy.data.materials[base_name]
    material = base.copy()
    material.name = runtime_name
    for node in material.node_tree.nodes:
        if node.type != "TEX_IMAGE" or node.image is None:
            continue
        image = node.image.copy()
        image.name = runtime_name + "_img"
        pixels = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(pixels)
        pixels = pixels.reshape(-1, 4)
        pixels[:, 0] *= tint[0]
        pixels[:, 1] *= tint[1]
        pixels[:, 2] *= tint[2]
        image.pixels.foreach_set(pixels.reshape(-1))
        image.update()
        node.image = image
        break
    else:
        raise RuntimeError(f"{base_name} has no image to tint")
    return material


def material_for(entry, cache):
    if "runtime_name" not in entry:
        return bpy.data.materials[entry["material"]]
    key = entry["runtime_name"]
    if key not in cache:
        cache[key] = tint_material(entry["material"], key, entry["tint_rgb"])
    return cache[key]


def hide_other_plates():
    armor = placement.armor
    hidden = []
    for obj in placement.scene.objects:
        if obj == armor or obj in armor.children_recursive:
            continue
        if obj.name.lower().startswith("armor"):
            for part in {obj, *obj.children_recursive}:
                part.hide_render = True
                hidden.append(part.name)
        elif obj.type == "MESH" and any(slot.material and slot.material.name.upper().startswith("LED")
                                        for slot in obj.material_slots):
            obj.hide_render = True
            hidden.append(obj.name)
    return sorted(set(hidden))


_plate_pool = []


def _set_plate_hidden(root, hidden):
    root.hide_render = hidden
    try:
        root.hide_set(hidden)
    except RuntimeError:
        pass
    for obj in root.children_recursive:
        obj.hide_render = hidden
        try:
            obj.hide_set(hidden)
        except RuntimeError:
            pass


def _clone_plate(index):
    source = placement.armor
    root = source.copy()
    root.name = f"batch_plate_{index}"
    root.rotation_mode = "QUATERNION"
    placement.scene.collection.objects.link(root)
    copied = {source: root}
    for child in list(source.children_recursive):
        clone = child.copy()
        if child.data:
            clone.data = child.data.copy()
        clone.name = f"batch_plate_{index}_{child.name}"
        placement.scene.collection.objects.link(clone)
        copied[child] = clone
    for src_obj, clone in copied.items():
        if src_obj is source:
            continue
        parent = copied.get(src_obj.parent)
        if parent is not None:
            clone.parent = parent
            clone.matrix_parent_inverse = src_obj.matrix_parent_inverse.copy()
    _set_plate_hidden(root, True)
    return root, copied[placement.panel]


def ensure_plate_pool(maximum):
    while len(_plate_pool) < max(0, maximum - 1):
        _plate_pool.append(_clone_plate(len(_plate_pool) + 1))
    return _plate_pool


def _plate_meshes(root):
    meshes = [root] if root.type == "MESH" else []
    meshes.extend(obj for obj in root.children_recursive if obj.type == "MESH")
    return meshes


def _other_armor_blocks(origin, target, own_meshes, other_roots):
    direction = target - origin
    distance = direction.length
    if distance < 1e-6:
        return False
    direction = direction.normalized()
    forbidden = set()
    for root in other_roots:
        forbidden.update(_plate_meshes(root))
    depsgraph = bpy.context.evaluated_depsgraph_get()
    start = origin.copy()
    for _ in range(12):
        remaining = (target - start).length
        if remaining < 1e-5:
            return False
        hit, location, _, _, obj, _ = placement.scene.ray_cast(
            depsgraph, start, direction, distance=remaining)
        if not hit:
            return False
        original = obj.original
        if original in own_meshes or obj.hide_get():
            start = location + direction * 1e-4
            continue
        return original in forbidden and (location - origin).length < distance - 0.002
    return False


def place_extra_plate(rng, root, panel, other_roots):
    """Place one more plate for the current camera. Returns None instead of rendering a bad pose."""
    from mathutils import Matrix, Vector
    _set_plate_hidden(root, False)
    bpy.context.view_layer.update()
    camera = placement.camera
    config = placement.config
    for _ in range(int(config.get("placement_retries", 6)) * 3):
        region_name = constraints.choose_region(rng, config["region_weights"])
        cell = constraints.choose_cell(rng, placement.cells[region_name], config.get("interior_cell_probability", 0.85))
        jitter = placement.step / 2
        x = cell["x"] + rng.uniform(-jitter, jitter)
        y = cell["y"] + rng.uniform(-jitter, jitter)
        support = placement.ground_at(x, y)
        if placement.region_of(support) != region_name:
            continue
        bottom_height = rng.uniform(*config["object_bottom_height_m"])
        yaw = rng.uniform(*config["yaw_deg"])
        pitch = rng.uniform(*config["pitch_deg"])
        roll = rng.uniform(*config["roll_deg"])
        camera_basis = tuple(tuple(camera.matrix_world.to_3x3()[row][col] for col in range(3)) for row in range(3))
        rotation = constraints.mul(
            constraints.mul(camera_basis, constraints.relative_rotation(yaw, pitch, roll)),
            constraints.transpose(placement.local_basis_tuples))
        origin_z = support["point"].z + bottom_height - constraints.bottom_offset(rotation, placement.local_tuples)
        origin = Vector((x, y, origin_z))
        if any((origin.xy - other.matrix_world.translation.xy).length < 0.22 for other in other_roots):
            continue
        distance = (origin - camera.matrix_world.translation).length
        if not (config["distance_m"][0] <= distance <= config["distance_m"][1]):
            continue
        root.location = origin
        root.rotation_quaternion = Matrix(rotation).to_quaternion()
        root.scale = (1, 1, 1)
        bpy.context.view_layer.update()
        world_points = [root.matrix_world @ local for local in placement.points_local]
        pixels = []
        for point in world_points:
            view = world_to_camera_view(placement.scene, camera, point)
            pixels.append((view.x * placement.width, (1 - view.y) * placement.height, view.z))
        if constraints.projection_reasons(pixels, placement.width, placement.height,
                                          config["image_margin_px"], config["min_projected_span_px"]):
            continue
        visible = evaluate_visibility(placement.scene, camera, panel, world_points, grid_size=config["visibility_grid"])
        screened = {"visibility": visible["visibility"],
                    "visible_area_min": visible["visible_area_fraction"],
                    "visible_corners_min": visible["visibility"].count(2)}
        if retention_reasons(screened, config["minimum_visible_area"], config["minimum_visible_corners"]):
            continue
        own = _plate_meshes(root)
        origin_cam = camera.matrix_world.translation
        blocked = any(_other_armor_blocks(origin_cam, point, own, other_roots) for point in world_points)
        if blocked:
            continue
        blocked_existing = False
        for other in other_roots:
            other_points = [other.matrix_world @ local for local in placement.points_local]
            if any(_other_armor_blocks(origin_cam, point, _plate_meshes(other), [root]) for point in other_points):
                blocked_existing = True
                break
        if blocked_existing:
            continue
        return {
            "root": root,
            "panel": panel,
            "points": [[p[0] / placement.width, p[1] / placement.height] for p in pixels],
            "visibility": visible["visibility"],
            "visible_area_fraction": visible["visible_area_fraction"],
            "distance_m": distance,
            "object_bottom_height_m": min((root.matrix_world @ local).z for local in placement.local_vectors) - support["point"].z,
            "region": region_name,
            "yaw_pitch_roll_deg": [yaw, pitch, roll],
        }
    _set_plate_hidden(root, True)
    return None


def configure_render():
    scene = placement.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = AUGMENT["render_samples"]
    scene.cycles.use_adaptive_sampling = False
    preserve = AUGMENT.get("preserve_moire", True)
    scene.cycles.use_denoising = not preserve
    if preserve:
        # Keep a narrow filter so the LED lattice can beat against the sensor.
        # Width 0 would disable filtering entirely, which the plan rejects.
        scene.cycles.filter_width = float(AUGMENT.get("moire_filter_width", 1.0))
    else:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    for device in prefs.devices:
        device.use = device.type != "CPU"
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = AUGMENT["jpeg_quality"]
    scene.render.use_compositing = bool(AUGMENT.get("use_compositor", False))
    scene.render.use_motion_blur = False
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    scene.render.resolution_x = placement.width
    scene.render.resolution_y = placement.height
    scene.render.resolution_percentage = 100


def _linearize_action(obj):
    action = obj.animation_data.action if obj.animation_data else None
    if action is None:
        return
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for curve in bag.fcurves:
                    for key in curve.keyframe_points:
                        key.interpolation = "LINEAR"


def apply_motion_blur(profile, roots, local_points_by_root):
    """Animate every plate root over a centered global-shutter exposure; labels stay at frame 2."""
    scene = placement.scene
    scene.render.fps = placement.profile["fps"]
    scene.render.fps_base = 1
    scene.render.motion_blur_position = "CENTER"
    scene.cycles.rolling_shutter_type = "NONE"
    scene.render.motion_blur_shutter = 0.0
    scene.render.use_motion_blur = bool(profile and profile.get("enabled"))
    saved = [(root, root.matrix_world.copy()) for root in roots]
    if not scene.render.use_motion_blur:
        scene.frame_set(1)
        return {"saved": saved, "enabled": False, "max_displacement_px": 0.0,
                "exposure_s": None, "speed_m_s": 0.0}
    exposure = float(profile["exposure_s"])
    speed = float(profile["speed_m_s"])
    angle = float(profile["direction_rad"])
    shutter_frames = exposure * scene.render.fps
    camera_basis = placement.camera.matrix_world.to_3x3()
    direction = (camera_basis.col[0] * math.cos(angle) + camera_basis.col[1] * math.sin(angle)).normalized()
    for root in roots:
        for obj in [root, *root.children_recursive]:
            if obj.type == "MESH":
                obj.cycles.use_motion_blur = True
                obj.cycles.motion_steps = 3
    displacement = 0.0
    accepted = False
    actual_speed = speed
    midpoint_pixels = {}
    for _ in range(7):
        for root, base in saved:
            root.animation_data_clear()
            base_location = base.translation.copy()
            offset = direction * (actual_speed * exposure * 0.5)
            root.location = base_location - offset
            root.keyframe_insert(data_path="location", frame=1)
            root.location = base_location + offset
            root.keyframe_insert(data_path="location", frame=3)
            _linearize_action(root)
        scene.render.motion_blur_shutter = shutter_frames
        scene.frame_set(2)
        bpy.context.view_layer.update()
        midpoint_pixels = {}
        for root, local_points in local_points_by_root:
            midpoint_pixels[root] = []
            for local in local_points:
                view = world_to_camera_view(scene, placement.camera, root.matrix_world @ local)
                midpoint_pixels[root].append((view.x * placement.width, (1 - view.y) * placement.height))
        displacement = 0.0
        valid = True
        for frame in (2 - shutter_frames * 0.5, 2 + shutter_frames * 0.5):
            scene.frame_set(int(math.floor(frame)), subframe=frame - math.floor(frame))
            bpy.context.view_layer.update()
            for root, local_points in local_points_by_root:
                for index, local in enumerate(local_points):
                    view = world_to_camera_view(scene, placement.camera, root.matrix_world @ local)
                    x, y = view.x * placement.width, (1 - view.y) * placement.height
                    if not (1 <= x < placement.width - 1 and 1 <= y < placement.height - 1):
                        valid = False
                    mid_x, mid_y = midpoint_pixels[root][index]
                    displacement = max(displacement, math.hypot(x - mid_x, y - mid_y))
        if valid:
            accepted = True
            break
        actual_speed *= 0.5
    scene.frame_set(2)
    if not accepted:
        for root, base in saved:
            root.animation_data_clear()
            root.matrix_world = base
        scene.render.use_motion_blur = False
        scene.render.motion_blur_shutter = 0.0
        scene.frame_set(1)
        return {"saved": saved, "enabled": False, "max_displacement_px": 0.0,
                "exposure_s": exposure, "speed_m_s": 0.0, "reason": "endpoint_out_of_frame"}
    return {"saved": saved, "enabled": True, "max_displacement_px": displacement,
            "exposure_s": exposure, "speed_m_s": actual_speed,
            "direction_rad": angle, "shutter_frames": shutter_frames}


def restore_motion_blur(state):
    scene = placement.scene
    for root, matrix in state["saved"]:
        root.animation_data_clear()
        root.matrix_world = matrix
    scene.render.use_motion_blur = False
    scene.render.motion_blur_shutter = 0.0
    scene.frame_set(1)
    bpy.context.view_layer.update()


def scale_lights(factor):
    saved = [(light, light.energy) for light in bpy.data.lights]
    for light, energy in saved:
        light.energy = energy * factor
    world_saved = None
    world = placement.scene.world
    if world and world.node_tree:
        for node in world.node_tree.nodes:
            if node.type == "BACKGROUND":
                world_saved = (node, node.inputs["Strength"].default_value)
                node.inputs["Strength"].default_value = world_saved[1] * factor
                break
    return saved, world_saved


def restore_lights(saved, world_saved):
    for light, energy in saved:
        light.energy = energy
    if world_saved is not None:
        node, strength = world_saved
        node.inputs["Strength"].default_value = strength


def load_float_image(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    width, height = image.size
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    return image, pixels.reshape(height, width, 4)


def save_float_image(image, array, path):
    image.pixels.foreach_set(np.ascontiguousarray(array, dtype=np.float32).reshape(-1))
    image.filepath_raw = str(path)
    image.file_format = "JPEG"
    image.save()


def apply_glare(array, keypoints, rng):
    if not AUGMENT.get("image_glare_enabled", True) or rng.random() > AUGMENT["glare_probability"]:
        return array, None
    height, width = array.shape[:2]
    radius = rng.uniform(*AUGMENT["glare_radius_px"])
    clearance = radius + AUGMENT["keypoint_clearance_px"]
    center = None
    for _ in range(16):
        cx = rng.uniform(radius, width - radius)
        cy = rng.uniform(radius, height - radius)
        if all((cx - x) ** 2 + (cy - y) ** 2 >= clearance ** 2 for x, y in keypoints):
            center = (cx, cy)
            break
    if center is None:
        return array, None
    color = choose_weighted(rng, AUGMENT["glare_colors"])
    alpha = rng.uniform(*AUGMENT["glare_alpha"])
    yy, xx = np.ogrid[:height, :width]
    blob = np.exp(-((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / (2 * (radius * 0.45) ** 2))
    for channel, value in enumerate(color["rgb"]):
        array[:, :, channel] = np.clip(array[:, :, channel] * (1 - alpha * blob) + value * alpha * blob, 0, 1)
    return array, {"name": color["name"], "center_px": [center[0], center[1]], "radius_px": radius, "alpha": alpha}


def paint_preview(array, points):
    preview = np.clip(array.copy(), 0, 1)
    height, width = preview.shape[:2]
    color = np.array([0.1, 1.0, 0.3, 1.0], dtype=np.float32)
    for x_norm, y_norm in points:
        x = int(round(x_norm * (width - 1)))
        # Blender stores the bottom row first; labels use top-left origin.
        y = int(round((1.0 - y_norm) * (height - 1)))
        preview[max(0, y - 3):y + 4, max(0, x - 3):x + 4] = color
    return preview


def cli_value(flag, default=None):
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if flag not in argv:
        return default
    index = argv.index(flag)
    if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
        return True
    return argv[index + 1]


def run_options():
    return {
        "batch": cli_value("--batch"),
        "resume": "--resume" in sys.argv,
        "count": int(cli_value("--count", AUGMENT["target_images"])),
        "seed": int(cli_value("--seed", AUGMENT["seed"])),
        "train": float(cli_value("--train", 0.75)),
        "val": float(cli_value("--val", 0.25)),
        "test": float(cli_value("--test", 0.0)),
        "debug": str(cli_value("--debug", "1")) != "0",
        "classes": cli_value("--classes", ""),
        "colors": cli_value("--colors", ""),
        "samples": cli_value("--samples"),
        "motion_blur_probability": float(cli_value("--motion-blur-probability", AUGMENT.get("motion_blur_probability", 0.0))),
        "min_plates": int(cli_value("--min-plates", 1)),
        "max_plates": int(cli_value("--max-plates", 1)),
        "plate_distribution": str(cli_value("--plate-distribution", "uniform")),
        "lamda": float(cli_value("--lamda", 0.0)),
        # Grouped ID swaps are a dataset policy in augment.json, never a UI/CLI toggle.
        "paired_scenes": bool(AUGMENT.get("paired_scenes", True)),
        "motion_blur_probability": float(cli_value("--motion-blur-probability", AUGMENT.get("motion_blur_probability", 0.0))),
    }


def log_line(batch, message):
    line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
    with (batch / "generation.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def blenderproc_version():
    init = ROOT / "tools/compatibility/vendor/blenderproc/blenderproc/version.py"
    candidates = list((ROOT / "tools/compatibility/vendor").glob("blenderproc/**/version.py"))
    for path in candidates:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "__version__" in line and "=" in line:
                return line.split("=", 1)[1].strip().strip("\"'")
    return "not loaded by this generator"


def save_rng(batch, pose_random, augment_rng, attempt):
    import pickle
    payload = {"pose": pose_random.getstate(), "augment": augment_rng.bit_generator.state, "attempt": attempt}
    (batch / "rng_state.pkl").write_bytes(pickle.dumps(payload))


def load_rng(batch, pose_random, augment_rng):
    import pickle
    path = batch / "rng_state.pkl"
    if not path.is_file():
        raise RuntimeError("cannot resume: rng_state.pkl is missing")
    payload = pickle.loads(path.read_bytes())
    pose_random.setstate(payload["pose"])
    augment_rng.bit_generator.state = payload["augment"]
    return int(payload["attempt"])


def write_manifest(batch, manifest):
    (batch / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def finish_statistics(batch, manifest, session_start, prior_active_seconds):
    manifest["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    manifest["active_seconds"] = prior_active_seconds + time.perf_counter() - session_start
    if manifest.get("started_at"):
        manifest["elapsed_seconds"] = (datetime.fromisoformat(manifest["finished_at"]) -
                                       datetime.fromisoformat(manifest["started_at"])).total_seconds()
    write_manifest(batch, manifest)
    try:
        import export_statistics
        report = export_statistics.export(batch)
        log_line(batch, f"STATISTICS statistics/report.json quality={report['summary']['quality_check']}")
    except Exception as exc:
        # Never discard a finished image because reporting failed.
        log_line(batch, f"STATISTICS_ERROR {type(exc).__name__}: {exc}")


def selected_ids(text, default):
    if text in (None, "", True):
        return set(default)
    return {int(part) for part in str(text).split(",") if part.strip() != ""}


def main():
    import os
    import random
    session_start = time.perf_counter()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    options = run_options()
    before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    generation_policy = {
        "version": 1, "source_sha256": before,
        "distractors": AUGMENT.get("distractors", {"enabled": False}),
        "use_compositor": AUGMENT.get("use_compositor", False),
        "image_glare_enabled": AUGMENT.get("image_glare_enabled", True),
    }
    if options["samples"] not in (None, "", True):
        AUGMENT["render_samples"] = int(options["samples"])
    class_filter = selected_ids(options["classes"], range(8))
    color_filter = selected_ids(options["colors"], (0, 1, 3))
    trainable = [item for item in CATALOG["entries"]
                 if item["color_id"] != 2 and item["class_id"] in class_filter and item["color_id"] in color_filter]
    if not trainable:
        raise RuntimeError("没有同时符合所选编号和颜色的装甲板")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    batch = Path(options["batch"]) if options["batch"] else DEFAULT_OUTPUT_ROOT / stamp
    resume = options["resume"]
    if resume and not (batch / "manifest.json").is_file():
        raise RuntimeError(f"cannot resume, no manifest in {batch}")
    distribution_options = {
        "generation_policy": generation_policy,
        "count": options["count"], "min_plates": options["min_plates"],
        "max_plates": options["max_plates"], "plate_distribution": options["plate_distribution"],
        "classes": sorted(class_filter), "colors": sorted(color_filter),
        "lamda": options["lamda"], "seed": options["seed"],
        "paired_scenes": options["paired_scenes"],
        "motion_blur_probability": options["motion_blur_probability"],
        "motion_blur_probability": options["motion_blur_probability"],
    }
    if resume:
        prior_plan = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
        if prior_plan.get("distribution_options") != distribution_options:
            raise RuntimeError("续跑必须保持原场景、干扰物/合成器配置、数量、编号、颜色、λ和同框设置；不兼容旧批次请新建任务")
        balance_plan = prior_plan["balance_plan"]
        balance.verify_prefix(balance_plan, prior_plan.get("samples", []))
    else:
        balance_plan = balance.make_plan(
            options["count"], options["min_plates"], options["max_plates"],
            options["plate_distribution"], [entry["class_id"] for entry in trainable],
            options["lamda"], options["seed"],
            group_size=len({e["class_id"] for e in trainable}) if options["paired_scenes"] else 1)
    for split in ("train", "val", "test"):
        (batch / "images" / split).mkdir(parents=True, exist_ok=True)
        (batch / "labels" / split).mkdir(parents=True, exist_ok=True)
    if options["debug"]:
        (batch / "preview").mkdir(exist_ok=True)
    stop_path = batch / "STOP"
    if stop_path.exists():
        stop_path.unlink()
    os.environ["ARMOR27_SAMPLER_OUTPUT"] = str(batch / "sampler")
    log_line(batch, "LOAD scene")
    placement.prepare()
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != before:
        raise RuntimeError("opening the blend changed the file")
    template = bpy.data.materials.get("LED-2blue")
    if template is None:
        raise RuntimeError("Missing LED replacement template LED-2blue")
    for entry in trainable:
        filename = entry["image"]
        source_image = ASSET_ROOT / "IDs" / filename
        if not source_image.is_file():
            raise RuntimeError(f"Missing plate image: {source_image}")
        material = bpy.data.materials.get(entry["material"])
        valid = material and material.node_tree and any(
            node.type == "TEX_IMAGE" and node.image and
            Path(bpy.path.abspath(node.image.filepath)).name == filename
            for node in material.node_tree.nodes)
        if not valid:
            material = template.copy()
            material.name = f"batch_{source_image.stem}"
            texture = next(node for node in material.node_tree.nodes if node.type == "TEX_IMAGE")
            texture.image = bpy.data.images.load(str(source_image), check_existing=True)
            entry["material"] = material.name
            log_line(batch, f"MATERIAL {material.name} image={filename}")
    hidden = hide_other_plates()
    configure_render()
    ensure_plate_pool(max(options["min_plates"], options["max_plates"]))
    background_pool = distractors.DistractorPool(placement, generation_policy["distractors"])
    compositor = (distractors.configure_compositor(
        placement, background_pool, [placement.panel, *[panel for _, panel in _plate_pool]])
        if generation_policy["use_compositor"] else None)
    pose_random = random.Random(options["seed"])
    augment_rng = np.random.default_rng(options["seed"] + 1)
    samples, rejected = [], []
    attempt = 0
    prior = {}
    if resume:
        prior = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
        if prior.get("random_seed") not in (None, options["seed"]):
            raise RuntimeError("resume seed does not match the original batch")
        samples = list(prior.get("samples", []))
        rejected = list(prior.get("rejected", []))
        attempt = load_rng(batch, pose_random, augment_rng)
        log_line(batch, f"RESUME samples={len(samples)} attempts={attempt}")
    else:
        log_line(batch, f"START id={batch.name} seed={options['seed']} count={options['count']}")
    target = options["count"]
    plan = scene_groups.split_groups(balance_plan["scene_groups"], options["train"], options["val"], options["test"], options["seed"])
    if resume and prior.get("split_plan"):
        plan = list(prior["split_plan"])
    cache = {}
    errors = prior.get("render_error_count", 0)
    prior_active_seconds = prior.get("active_seconds", 0)
    sampling = json.loads((ROOT / "tools/placement/sampling_config.json").read_text(encoding="utf-8"))
    manifest = {
        "generation_id": prior.get("generation_id", batch.name) if resume else batch.name,
        "created": prior.get("created", stamp) if resume else stamp,
        "random_seed": options["seed"],
        "started_at": prior.get("started_at", started_at),
        "render_error_count": errors,
        "blender_version": bpy.app.version_string,
        "blenderproc_version": blenderproc_version(),
        "catalog_schema": CATALOG.get("schema_version"),
        "augment_schema": AUGMENT.get("schema_version"),
        "sampling_schema": sampling.get("schema_version"),
        "scene_file": SOURCE.name,
        "source_sha256": before,
        "source_unchanged": True,
        "render_samples": AUGMENT["render_samples"],
        "denoiser": "NONE" if AUGMENT.get("preserve_moire", True) else "OPENIMAGEDENOISE",
        "resolution": [placement.width, placement.height],
        "debug_images": options["debug"],
        "split_ratios": {"train": options["train"], "val": options["val"], "test": options["test"]},
        "class_filter": sorted(class_filter),
        "color_filter": sorted(color_filter),
        "distribution_options": distribution_options,
        "balance_plan": balance_plan,
        "object_class_counts": dict(balance.actual_counts(samples)),
        "split_plan": plan[:target],
        "catalog_notes": CATALOG["notes"],
        "asset_catalog_entries": CATALOG["entries"],
        "unavailable": CATALOG["unavailable"],
        "excluded_materials": CATALOG["excluded_materials"],
        "augment": AUGMENT,
        "generation_policy": generation_policy,
        "compositor": compositor,
        "hidden_other_plates": hidden,
        "status": "running",
        "attempts": attempt,
        "rejected": rejected,
        "samples": samples,
        "error_count": prior.get("error_count", 0) if resume else 0,
        "not_included": [
            "vertical flip",
            "image translate/scale (camera pose already varies framing)",
            "purple plates; color_id 2 is not sampled",
            "512-sample moire calibration",
            "writing into data/images or data/labels",
        ],
    }
    save_rng(batch, pose_random, augment_rng, attempt)
    write_manifest(batch, manifest)
    log_line(batch, f"CLASS_QUOTAS lamda={options['lamda']} total={balance_plan['total_objects']} quotas={balance_plan['quotas']}")
    while len(samples) < target and attempt < max(AUGMENT["max_attempts"], target * 12):
        background_pool.hide_all()
        if stop_path.exists():
            manifest["status"] = "stopped"
            manifest["attempts"] = attempt
            manifest["error_count"] = errors + len(rejected)
            write_manifest(batch, manifest)
            manifest["sample_count"] = len(samples)
            manifest["render_error_count"] = errors
            manifest["object_class_counts"] = dict(balance.actual_counts(samples))
            log_line(batch, f"STOPPED samples={len(samples)} attempts={attempt}")
            finish_statistics(batch, manifest, session_start, prior_active_seconds)
            return
        group_id = balance_plan["scene_groups"][len(samples)]
        recipe = (samples[-1].get("scene_recipe") if samples and
                  samples[-1].get("scene_group") == group_id else None)
        ensure_plate_pool(max(options["min_plates"], options["max_plates"]))
        for root, _panel in _plate_pool:
            _set_plate_hidden(root, True)
        if recipe:
            from mathutils import Matrix
            placement.camera.matrix_world = Matrix(recipe["camera"])
            placement.armor.matrix_world = Matrix(recipe["armor"])
            bpy.context.view_layer.update()
            record = dict(recipe["record"])
            record["attempt"] = attempt
        else:
            # Hold the candidate yaw bin fixed for this scene group, not for an ID.
            pose_index = group_id % len(placement.config["yaw_bins_deg"]) if options["paired_scenes"] else attempt
            record = placement.attempt_once(pose_random, pose_index)
            record["attempt"] = attempt
        attempt += 1
        save_rng(batch, pose_random, augment_rng, attempt)
        if record["reasons"]:
            rejected.append({"attempt": record["attempt"], "reasons": record["reasons"]})
            manifest["attempts"] = attempt
            manifest["rejected"] = rejected
            manifest["error_count"] = errors + len(rejected)
            write_manifest(batch, manifest)
            log_line(batch, f"REJECT {record['attempt']} {record['reasons']}")
            continue
        geom, _, _ = labels.geometry_issues(record["keypoints_normalized"], record["visibility"])
        if geom:
            rejected.append({"attempt": record["attempt"], "reasons": geom})
            manifest["attempts"] = attempt
            manifest["rejected"] = rejected
            manifest["error_count"] = errors + len(rejected)
            write_manifest(batch, manifest)
            log_line(batch, f"REJECT {record['attempt']} {geom}")
            continue
        points_norm = record.get("keypoints_normalized") or []
        group_size_bucket = None
        if options["paired_scenes"] and not recipe:
            geom_bucket = scene_groups.buckets(
                points_norm, placement.width, placement.height,
                record["measured_yaw_pitch_roll_deg"], record["visible_area_fraction"],
                record["visibility"], -1)
            group_size_bucket = ("small", "medium", "large")[(group_id // 5) % 3]
            if geom_bucket["size"] != group_size_bucket:
                rejected.append({"attempt": record["attempt"], "reasons": ["scene_size_bucket"],
                                 "wanted": group_size_bucket, "got": geom_bucket["size"]})
                manifest["attempts"] = attempt
                write_manifest(batch, manifest)
                continue
        xs = [p[0] * placement.width for p in points_norm]
        ys = [p[1] * placement.height for p in points_norm]
        span = max(max(xs) - min(xs), max(ys) - min(ys)) if xs else 0.0
        band = AUGMENT.get("moire_span_px", [90, 420])
        in_band = band[0] <= span <= band[1]
        in_band_count = sum(1 for item in samples if item.get("in_moire_band"))
        fraction = AUGMENT.get("moire_minimum_fraction", 0.5) if AUGMENT.get("preserve_moire", True) else 0
        if not options["paired_scenes"] and not recipe and labels.needs_moire_span(in_band, in_band_count, len(samples), target, fraction):
            rejected.append({"attempt": record["attempt"], "reasons": ["outside_moire_span"], "span_px": span})
            manifest["attempts"] = attempt
            manifest["rejected"] = rejected
            manifest["error_count"] = errors + len(rejected)
            write_manifest(batch, manifest)
            log_line(batch, f"REJECT {record['attempt']} outside_moire_span span={span:.0f}px")
            continue
        # Reserve this image's slots; failed placements never consume them.
        image_entries = balance.entries_for_image(balance_plan, samples, trainable, pose_random)
        entry = image_entries[0]
        material = material_for(entry, cache)
        placement.panel.material_slots[0].material = material
        plate_total = balance_plan["image_counts"][len(samples)]
        ensure_plate_pool(max(options["min_plates"], options["max_plates"]))
        for root, _panel in _plate_pool:
            _set_plate_hidden(root, True)
        extra_plates = []
        placed_roots = [placement.armor]
        group_ok = True
        for extra_index in range(1, plate_total):
            root, panel = _plate_pool[extra_index - 1]
            if recipe:
                saved_extra = recipe["extras"][extra_index - 1]
                root.matrix_world = Matrix(saved_extra["matrix"])
                _set_plate_hidden(root, False)
                bpy.context.view_layer.update()
                placed = dict(saved_extra["record"], root=root, panel=panel)
            else:
                placed = place_extra_plate(pose_random, root, panel, placed_roots)
            if placed is None:
                group_ok = False
                break
            extra_entry = image_entries[extra_index]
            extra_material = material_for(extra_entry, cache)
            panel.material_slots[0].material = extra_material
            placed["entry"] = extra_entry
            placed["material"] = extra_material
            extra_plates.append(placed)
            placed_roots.append(root)
        if not group_ok:
            for root, _panel in _plate_pool:
                _set_plate_hidden(root, True)
            rejected.append({"attempt": record["attempt"], "reasons": ["extra_plate_blocked"]})
            manifest["attempts"] = attempt
            manifest["rejected"] = rejected
            manifest["error_count"] = errors + len(rejected)
            write_manifest(batch, manifest)
            log_line(batch, f"REJECT {record['attempt']} extra_plate_blocked wanted={plate_total}")
            continue
        motion_profile = (recipe.get("motion_profile") if recipe else None)
        if motion_profile is None:
            motion_profile = motion.sample_profile(
                pose_random, options["motion_blur_probability"],
                AUGMENT.get("motion_exposure_s", CAMERA_PROFILE["exposure_test_seconds"]),
                AUGMENT.get("motion_speed_m_s", [0.2, 0.8]))
        scene_recipe = recipe or {
            "camera": [list(row) for row in placement.camera.matrix_world],
            "armor": [list(row) for row in placement.armor.matrix_world],
            "record": record,
            "extras": [{"matrix": [list(row) for row in item["root"].matrix_world],
                        "record": {key: value for key, value in item.items()
                                   if key not in ("root", "panel", "entry", "material")}}
                       for item in extra_plates],
            "motion_profile": motion_profile,
        }
        # Identical lighting, noise and glare draws within a scene group.
        image_rng = np.random.default_rng(options["seed"] + group_id * 100003 + 17)
        factor = float(image_rng.uniform(*AUGMENT["light_energy_scale"]))
        saved, world_saved = scale_lights(factor)
        split = plan[len(samples)]
        number = labels.sample_id(split, sum(item["split"] == split for item in samples))
        image_path = batch / "images" / split / f"{number}.jpg"
        placement.scene.render.filepath = str(image_path)
        roots = [placement.armor, *[item["root"] for item in extra_plates]]
        local_points_by_root = [(root, placement.points_local) for root in roots]
        motion_state = apply_motion_blur(motion_profile, roots, local_points_by_root)
        scene_recipe["motion_profile"] = {
            key: value for key, value in motion_profile.items()
        }
        scene_recipe["motion_applied"] = {key: value for key, value in motion_state.items() if key != "saved"}
        if recipe is None:
            # Keep the actual exposure displacement identical across all swapped-ID images.
            pass
        image_start = time.perf_counter()
        try:
            if recipe:
                background = recipe["distractors"]
                background_pool.restore(background)
            else:
                background_rng = random.Random(options["seed"] + group_id * 100003 + attempt * 7919 + 53)
                background = background_pool.sample(background_rng, roots, motion_state)
            observations = distractors.measure_visibility(
                placement, [(placement.armor, placement.panel),
                            *[(item["root"], item["panel"]) for item in extra_plates]], motion_state)
            reasons = sorted({reason for observed in observations for reason in retention_reasons(
                observed, placement.config["minimum_visible_area"], placement.config["minimum_visible_corners"])})
            if reasons:
                rejected.append({"attempt": record["attempt"], "reasons": reasons,
                                 "stage": "distractor_exposure_visibility"})
                manifest.update({"attempts": attempt, "rejected": rejected,
                                 "error_count": errors + len(rejected)})
                write_manifest(batch, manifest)
                log_line(batch, f"REJECT {record['attempt']} distractor_exposure_visibility {reasons}")
                continue
            for item, observed in zip([record, *extra_plates], observations):
                item["visibility"] = observed["visibility"]
                item["visible_area_fraction"] = observed["visible_area_min"]
            scene_recipe["record"] = record
            for saved_extra, placed in zip(scene_recipe["extras"], extra_plates):
                saved_extra["record"]["visibility"] = placed["visibility"]
                saved_extra["record"]["visible_area_fraction"] = placed["visible_area_fraction"]
            background["visible_sampled"] = background_pool.visible_counts()
            background["visibility_method"] = "nine rays per instance at exposure midpoint; approximate"
            scene_recipe["distractors"] = background
            bpy.ops.render.render(write_still=True)
            render_seconds = time.perf_counter() - image_start
            image, array = load_float_image(image_path)
            plate_rows = [{
                "entry": entry, "material": material,
                "angles": record["measured_yaw_pitch_roll_deg"],
                "area": record["visible_area_fraction"],
                "region": record["region"], "distance_m": record["actual_distance_m"],
                "object_bottom_height_m": record["object_bottom_height_m"],
                "points": [tuple(p) for p in record["keypoints_normalized"]],
                "visibility": list(record["visibility"]),
            }]
            plate_rows.extend({
                "entry": item["entry"], "material": item["material"],
                "angles": item["yaw_pitch_roll_deg"], "area": item["visible_area_fraction"],
                "region": item.get("region"), "distance_m": item["distance_m"],
                "object_bottom_height_m": item["object_bottom_height_m"],
                "points": [tuple(p) for p in item["points"]],
                "visibility": list(item["visibility"]),
            } for item in extra_plates)
            all_points = [point for item in plate_rows for point in item["points"]]
            pixel_points = [(x * placement.width, y * placement.height) for x, y in all_points]
            array, glare = apply_glare(array, pixel_points, image_rng)
            flipped = bool(image_rng.random() < AUGMENT["flip_probability"])
            if flipped:
                array = array[:, ::-1, :].copy()
                for item in plate_rows:
                    item["points"], item["visibility"] = labels.horizontal_flip(item["points"], item["visibility"])
            noise_sigma = AUGMENT.get("moire_noise_sigma", 0.004) if AUGMENT.get("preserve_moire", True) else AUGMENT["noise_sigma"]
            noise = image_rng.normal(0, noise_sigma, array[:, :, :3].shape).astype(np.float32)
            array[:, :, :3] = np.clip(array[:, :, :3] + noise, 0, 1)
            save_float_image(image, array, image_path)
            flat_points = [point for item in plate_rows for point in item["points"]]
            if options["debug"]:
                preview = bpy.data.images.new(number + "_preview", placement.width, placement.height, alpha=True)
                save_float_image(preview, paint_preview(array, flat_points), batch / "preview" / f"{number}.jpg")
                bpy.data.images.remove(preview)
            bpy.data.images.remove(image)
            label_rows = [
                labels.format_row(item["entry"]["color_id"], item["entry"]["class_id"], item["points"], item["visibility"])
                for item in plate_rows
            ]
            (batch / "labels" / split / f"{number}.txt").write_text("\n".join(label_rows) + "\n", encoding="utf-8")
            first = plate_rows[0]
            samples.append({
                "id": number, "split": split, "attempt": record["attempt"],
                "material": first["material"].name, "base_material": first["entry"]["material"],
                "class_id": first["entry"]["class_id"], "color_id": first["entry"]["color_id"],
                "class_name": CATALOG["class_names"][first["entry"]["class_id"]],
                "color_name": CATALOG["color_names"][first["entry"]["color_id"]],
                "joint_id": labels.joint_id(first["entry"]["color_id"], first["entry"]["class_id"]),
                "region": record["region"],
                "object_bottom_height_m": record["object_bottom_height_m"],
                "camera_height_m": record["camera_height_m"],
                "distance_m": record["actual_distance_m"],
                "yaw_pitch_roll_deg": record["measured_yaw_pitch_roll_deg"],
                "visibility": first["visibility"], "visible_area_fraction": record["visible_area_fraction"],
                "keypoints_normalized": first["points"], "flipped": flipped, "glare": glare,
                "light_energy_scale": factor,
                "motion_blur": motion_state["enabled"],
                "motion_blur_probability": options["motion_blur_probability"],
                "motion_exposure_s": motion_state["exposure_s"],
                "motion_speed_m_s": motion_state["speed_m_s"],
                "motion_max_displacement_px": motion_state["max_displacement_px"],
                "noise_sigma": noise_sigma, "render_seconds": render_seconds,
                "export_seconds": time.perf_counter() - image_start,
                "moire_span_px": span, "in_moire_band": in_band,
                "plate_count": len(plate_rows),
                "scene_group": group_id, "scene_recipe": scene_recipe,
                "plates": [{
                    "material": item["material"].name,
                    "base_material": item["entry"]["material"],
                    "class_id": item["entry"]["class_id"],
                    "color_id": item["entry"]["color_id"],
                    "class_name": CATALOG["class_names"][item["entry"]["class_id"]],
                    "color_name": CATALOG["color_names"][item["entry"]["color_id"]],
                    "joint_id": labels.joint_id(item["entry"]["color_id"], item["entry"]["class_id"]),
                    "visibility": item["visibility"],
                    "region": item["region"], "distance_m": item["distance_m"],
                    "object_bottom_height_m": item["object_bottom_height_m"],
                    "camera_height_m": record["camera_height_m"],
                    "yaw_pitch_roll_deg": item["angles"], "visible_area_fraction": item["area"],
                    "keypoints_normalized": item["points"],
                    "buckets": {
                        **scene_groups.buckets(item["points"], placement.width, placement.height,
                                               item["angles"], item["area"], item["visibility"],
                                               item["entry"]["color_id"]),
                        "image_quality": motion.quality_bucket(
                            motion_state["max_displacement_px"], motion_state["enabled"]),
                    },
                } for item in plate_rows],
            })
            log_line(batch, f"SAMPLE {number} {split} plates={len(plate_rows)} " +
                     ",".join(item["material"].name for item in plate_rows))
        except Exception as exc:
            errors += 1
            if image_path.exists() and not (batch / "labels" / split / f"{number}.txt").exists():
                image_path.unlink()
            log_line(batch, f"ERROR attempt={record['attempt']} {type(exc).__name__}: {exc}")
        finally:
            background_pool.hide_all()
            restore_motion_blur(motion_state)
            for root, _panel in _plate_pool:
                _set_plate_hidden(root, True)
            restore_lights(saved, world_saved)
            save_rng(batch, pose_random, augment_rng, attempt)
        manifest.update({
            "status": "running", "attempts": attempt, "samples": samples, "rejected": rejected,
            "error_count": errors + len(rejected), "sample_count": len(samples),
            "render_error_count": errors,
            "object_class_counts": dict(balance.actual_counts(samples)),
            "source_unchanged": hashlib.sha256(SOURCE.read_bytes()).hexdigest() == before,
            "split_plan": plan[:target],
        })
        write_manifest(batch, manifest)
    balance.verify_prefix(balance_plan, samples)
    manifest["object_class_counts"] = dict(balance.actual_counts(samples))
    quotas_met = all(balance.actual_counts(samples)[int(c)] == n for c, n in balance_plan["quotas"].items())
    manifest["class_quotas_met"] = quotas_met
    manifest["conditional_coverage"] = scene_groups.coverage(samples, balance_plan["classes"])
    manifest["coverage_gaps"] = scene_groups.missing_buckets(manifest["conditional_coverage"])
    (batch / "coverage.json").write_text(json.dumps({
        "counts": manifest["conditional_coverage"], "missing": manifest["coverage_gaps"],
        "note": "Geometry buckets are measured; image quality is unverified. Missing buckets are not claimed covered.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["status"] = "passed" if len(samples) >= target and quotas_met else "needs_review"
    manifest["attempts"] = attempt
    manifest["sample_count"] = len(samples)
    manifest["error_count"] = errors + len(rejected)
    manifest["render_error_count"] = errors
    manifest["source_unchanged"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest() == before
    write_manifest(batch, manifest)
    log_line(batch, f"FINISH {manifest['status']} samples={len(samples)} errors={manifest['error_count']}")
    if not manifest["source_unchanged"]:
        log_line(batch, "NOTE source blend changed during the run; this generator did not save it")
    finish_statistics(batch, manifest, session_start, prior_active_seconds)
    if manifest["status"] != "passed":
        raise RuntimeError(manifest["status"])


if __name__ == "__main__":
    main()
