"""Phase 4 sampler: field position and camera pose, without writing a dataset.

Region membership comes from the object and material named in sampling_config.json.
A downward ray only supplies the contact height and confirms that identity; it is
not a color threshold. The source blend is opened read-only.
"""

import hashlib
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = Path(os.environ.get("ARMOR27_ASSET_ROOT", ROOT)).expanduser().resolve()
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "tools" / "occlusion_validation"))
import constraints
from visibility import evaluate_visibility, retention_reasons


def argument(name, default=None):
    import os
    if name == "--output" and os.environ.get("ARMOR27_SAMPLER_OUTPUT"):
        return os.environ["ARMOR27_SAMPLER_OUTPUT"]
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if name in argv:
        return argv[argv.index(name) + 1]
    return default


def assert_mathutils_agreement():
    for direction in ((0, 1, 0), (1, -0.2, 0.3), (0.2, 0.1, -0.9)):
        ours = Matrix(constraints.camera_basis(direction))
        reference = Vector(direction).to_track_quat("-Z", "Y").to_matrix()
        if max(abs(ours[i][j] - reference[i][j]) for i in range(3) for j in range(3)) > 1e-5:
            raise RuntimeError(f"Camera basis disagrees with Blender for {direction}")
    for angles in ((0, 0, 0), (40, -10, 15), (-35, 10, -25)):
        ours = Matrix(constraints.relative_rotation(*angles))
        reference = (Matrix.Rotation(math.radians(angles[2]), 3, "Z")
                     @ Matrix.Rotation(math.radians(angles[1]), 3, "X")
                     @ Matrix.Rotation(math.radians(angles[0]), 3, "Y"))
        if max(abs(ours[i][j] - reference[i][j]) for i in range(3) for j in range(3)) > 1e-5:
            raise RuntimeError(f"Relative rotation disagrees with Blender for {angles}")


def ground_at(x, y):
    origin = Vector((x, y, 30))
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for _ in range(100):
        hit, point, normal, face_index, obj, _ = scene.ray_cast(depsgraph, origin, Vector((0, 0, -1)))
        if not hit:
            return None
        original = obj.original
        if original in armor_set:
            origin = point - Vector((0, 0, 1e-4))
            continue
        if original in field_objects:
            material_name = None
            try:
                material = obj.material_slots[obj.data.polygons[face_index].material_index].material
                material_name = material.name if material else None
            except (IndexError, AttributeError):
                material_name = None
            return {"point": point.copy(), "normal": normal.normalized(),
                    "object": original.name, "material": material_name}
        origin = point - Vector((0, 0, 1e-4))
    return None


def region_of(hit):
    if hit is None:
        return None
    central = config["regions"]["central_black"]
    if hit["object"] == central["object"] and hit["material"] == central["material"]:
        return "central_black" if hit["normal"].z >= central["min_normal_z"] else None
    if hit["normal"].z >= config["regions"]["other_field"]["min_normal_z"]:
        return "other_field"
    return None


def world_xy_bounds(obj):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return (min(v.x for v in corners), max(v.x for v in corners),
            min(v.y for v in corners), max(v.y for v in corners))


def measured_angles():
    transform = camera.matrix_world.to_3x3().transposed() @ armor.matrix_world.to_3x3() @ local_basis
    pitch = math.asin(max(-1, min(1, transform[2][1])))
    roll = math.atan2(-transform[0][1], transform[1][1])
    yaw = math.atan2(-transform[2][0], transform[2][2])
    return [math.degrees(v) for v in (yaw, pitch, roll)]


def solve_pose(x, y, support_z, bottom_height, yaw, pitch, roll, distance, azimuth, cam_height, look_u, look_v):
    def pose_from(origin_z):
        origin = (x, y, origin_z)
        cam_z = support_z + cam_height
        for _ in range(3):
            horiz = constraints.horizontal_offset(distance, cam_z - origin_z)
            if horiz is None:
                return "unreachable"
            camera_xy = (x + horiz * math.cos(azimuth), y + horiz * math.sin(azimuth))
            ground = ground_at(*camera_xy)
            if ground is None:
                return "off_field"
            cam_z = ground["point"].z + cam_height
        horiz = constraints.horizontal_offset(distance, cam_z - origin_z)
        if horiz is None:
            return "unreachable"
        camera_xy = (x + horiz * math.cos(azimuth), y + horiz * math.sin(azimuth))
        ground = ground_at(*camera_xy)
        if ground is None:
            return "off_field"
        camera_location = (camera_xy[0], camera_xy[1], ground["point"].z + cam_height)
        basis = constraints.camera_basis(constraints.sub(origin, camera_location))
        right_axis = (basis[0][0], basis[1][0], basis[2][0])
        up_axis = (basis[0][1], basis[1][1], basis[2][1])
        look = constraints.add(origin, constraints.add(constraints.scale(right_axis, look_u),
                                                       constraints.scale(up_axis, look_v)))
        rotation = constraints.armor_rotation(camera_location, look, yaw, pitch, roll, local_basis_tuples)
        corrected_z = support_z + bottom_height - constraints.bottom_offset(rotation, local_tuples)
        return camera_location, look, rotation, corrected_z

    origin_z = support_z + bottom_height - min(vertex[2] for vertex in local_tuples)
    solved = None
    for _ in range(config["pose_iterations"]):
        solved = pose_from(origin_z)
        if solved == "unreachable":
            return ["camera_height_unreachable"], None
        if solved == "off_field":
            return ["camera_off_field"], None
        camera_location, look, rotation, origin_z = solved
    # Keep the rotation that belongs to this origin. A last camera rebuild absorbs the z correction.
    solved = pose_from(origin_z)
    if solved == "unreachable":
        return ["camera_height_unreachable"], None
    if solved == "off_field":
        return ["camera_off_field"], None
    camera_location, look, rotation, corrected_z = solved
    armor.location = Vector((x, y, origin_z))
    armor.rotation_quaternion = Matrix(rotation).to_quaternion()
    armor.scale = (1, 1, 1)
    camera.location = Vector(camera_location)
    camera.rotation_quaternion = (Vector(look) - camera.location).to_track_quat("-Z", "Y")
    bpy.context.view_layer.update()
    return [], {"corrected_origin_z": corrected_z}


def footprint_and_collision(expected_region, support_z):
    reasons = []
    occluder = None
    world_points = [armor.matrix_world @ vertex for vertex in local_vectors]
    min_x_ = min(p.x for p in world_points)
    max_x_ = max(p.x for p in world_points)
    min_y_ = min(p.y for p in world_points)
    max_y_ = max(p.y for p in world_points)
    bottom = min(p.z for p in world_points)
    top = max(p.z for p in world_points)
    center = sum(world_points, Vector()) / len(world_points)
    probes = [center, Vector((min_x_, min_y_, 0)), Vector((min_x_, max_y_, 0)),
              Vector((max_x_, min_y_, 0)), Vector((max_x_, max_y_, 0))]
    for probe in probes:
        hit = ground_at(probe.x, probe.y)
        if region_of(hit) != expected_region:
            reasons.append("footprint_leaves_region")
            break
        if abs(hit["point"].z - support_z) > config["support_height_tolerance_m"]:
            reasons.append("footprint_height_step")
            break
    def first_foreign(origin, direction, distance):
        start = origin.copy()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        remaining = distance
        for _ in range(16):
            if remaining <= 1e-5:
                return None, None
            hit, location, _, _, obj, _ = scene.ray_cast(depsgraph, start, direction, distance=remaining)
            if not hit:
                return None, None
            if obj.original in armor_set:
                used = (location - start).length + 1e-4
                remaining -= used
                start = location + direction * 1e-4
                continue
            return obj.original, location
        return None, None

    foreign, location = first_foreign(Vector((center.x, center.y, top + 0.2)), Vector((0, 0, -1)), top + 0.5 - support_z)
    if foreign is not None and location.z > bottom + config["clearance_above_bottom_m"]:
        reasons.append("collision")
        occluder = foreign.name
    mid_z = max(bottom + config["clearance_above_bottom_m"], (bottom + top) / 2)
    for probe in probes[1:]:
        origin = Vector((center.x, center.y, mid_z))
        ray = Vector((probe.x, probe.y, mid_z)) - origin
        if ray.length < 1e-6:
            continue
        foreign, location = first_foreign(origin, ray.normalized(), ray.length)
        if foreign is not None and (location - origin).length < ray.length - 0.005:
            reasons.append("collision")
            occluder = foreign.name
            break
    return reasons, occluder, bottom




def prepare():
    global config, bindings, profile, scene, armor, panel, field_objects, armor_set
    global points_local, local_basis, local_basis_tuples, local_vectors, local_tuples
    global width, height, camera, cells, step, source_hash
    global output, source, source_hash, min_x, max_x, min_y, max_y, scanned
    config = constraints.load_config(Path(__file__).with_name("sampling_config.json"))
    bindings = json.loads((ROOT / "tools/asset_validation/asset_bindings.json").read_text())
    profile = json.loads((ROOT / "tools/motion_validation/camera_profile.json").read_text())
    source = ASSET_ROOT / bindings["source_blend"]
    output = Path(argument("--output", str(ROOT / "tools/placement/runs" / datetime.now().strftime("%Y%m%d-%H%M%S"))))
    output.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    bpy.ops.wm.open_mainfile(filepath=str(source), load_ui=False, use_scripts=False)
    scene = bpy.context.scene
    if bindings["field_root"] != config["regions"]["field_root"]:
        raise RuntimeError("Field root in bindings and sampling config disagree")

    armor = bpy.data.objects[bindings["armor_root"]]
    panel = bpy.data.objects[bindings["panel_object"]]
    field = bpy.data.objects[config["regions"]["field_root"]]
    floor = bpy.data.objects[config["regions"]["floor_bounds_object"]]
    central_object = bpy.data.objects[config["regions"]["central_black"]["object"]]
    field_objects = {field, *field.children_recursive}
    armor_set = {armor, *armor.children_recursive}
    if central_object not in field_objects:
        raise RuntimeError("Central region object is not under the field root")

    points_local = [Vector(p) for p in bindings["keypoints"]["armor_local_coordinates"]]
    bpy.context.view_layer.update()
    for index, local in zip(bindings["keypoints"]["panel_vertex_indices"], points_local):
        actual = armor.matrix_world.inverted() @ panel.matrix_world @ panel.data.vertices[index].co
        if (actual - local).length > 1e-6:
            raise RuntimeError("Bound LED corners no longer match the panel mesh")
    right = (points_local[3] - points_local[0]).normalized()
    up = (points_local[0] - points_local[1]).normalized()
    normal = right.cross(up).normalized()
    up = normal.cross(right).normalized()
    local_basis = Matrix((right, up, normal)).transposed()
    local_basis_tuples = tuple(tuple(row) for row in local_basis)
    local_vectors = []
    for obj in armor.children_recursive:
        if obj.type != "MESH":
            continue
        for vertex in obj.data.vertices:
            local_vectors.append(armor.matrix_world.inverted() @ obj.matrix_world @ vertex.co)
    local_tuples = [tuple(v) for v in local_vectors]
    if not local_tuples:
        raise RuntimeError("Armor has no mesh vertices")



    assert_mathutils_agreement()
    width, height = profile["width"], profile["height"]
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    camera = scene.camera
    camera.rotation_mode = "QUATERNION"
    armor.rotation_mode = "QUATERNION"
    camera.data.type = "PERSP"
    camera.data.sensor_width = profile["sensor_width_mm"]
    camera.data.sensor_height = profile["sensor_height_mm"]
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.lens = profile["focal_length_mm"]
    camera.data.shift_x = camera.data.shift_y = 0
    camera.data.dof.use_dof = False
    camera.data.clip_start = min(camera.data.clip_start, 0.01)



    min_x, max_x, min_y, max_y = world_xy_bounds(floor)
    step = config["grid_step_m"]
    cells = {"central_black": [], "other_field": []}
    scanned = Counter()
    x = min_x + step / 2
    while x < max_x:
        y = min_y + step / 2
        while y < max_y:
            hit = ground_at(x, y)
            kind = region_of(hit)
            scanned[kind or "rejected_surface"] += 1
            if kind:
                cells[kind].append({"x": x, "y": y, "z": hit["point"].z,
                                    "object": hit["object"], "material": hit["material"]})
            y += step
        x += step
        print("PLACEMENT_GRID", round(x, 2), {k: len(v) for k, v in cells.items()}, flush=True)
    if min(len(v) for v in cells.values()) < 10:
        raise RuntimeError(f"Sampling mask is too small: { {k: len(v) for k, v in cells.items()} }")
    for region_cells in cells.values():
        constraints.mark_interior(region_cells, step)




def attempt_once(rng, attempt):
    region_name = constraints.choose_region(rng, config["region_weights"])
    pool = cells[region_name]
    retries = int(config.get("placement_retries", 6))
    jitter = step / 2
    support_z = None
    x = y = None
    for _ in range(retries):
        cell = constraints.choose_cell(rng, pool, config.get("interior_cell_probability", 0.85))
        x = cell["x"] + rng.uniform(-jitter, jitter)
        y = cell["y"] + rng.uniform(-jitter, jitter)
        support = ground_at(x, y)
        if region_of(support) == region_name:
            support_z = support["point"].z
            break
    reasons = []
    if support_z is None:
        reasons.append("left_region_cell")
    yaw, yaw_bin = constraints.sample_yaw(rng, config["yaw_bins_deg"], attempt)
    pitch = rng.uniform(*config["pitch_deg"])
    roll = rng.uniform(*config["roll_deg"])
    cam_height = rng.uniform(*config["camera_height_m"])
    bottom_height = rng.uniform(*config["object_bottom_height_m"])
    origin_clearance = bottom_height - min(vertex[2] for vertex in local_tuples)
    distance = None
    low, high = config["distance_m"]
    for _ in range(retries):
        gap = abs(cam_height - origin_clearance) + 0.05
        distance = constraints.sample_distance_at_least(
            rng, low, high, config["distance_distribution"], gap)
        if distance is not None:
            break
        cam_height = rng.uniform(*config["camera_height_m"])
        bottom_height = rng.uniform(*config["object_bottom_height_m"])
        origin_clearance = bottom_height - min(vertex[2] for vertex in local_tuples)
    if distance is None:
        reasons.append("camera_height_unreachable")
    look_span = config["look_offset_fraction_of_distance"] * (distance or low)
    look_u = rng.uniform(-look_span, look_span)
    look_v = rng.uniform(-look_span, look_span)
    record = {"attempt": attempt, "region": region_name, "yaw_bin_deg": yaw_bin,
              "requested_yaw_pitch_roll_deg": [yaw, pitch, roll], "distance_m": distance}
    if support_z is not None and distance is not None:
        pose_reasons = ["camera_height_unreachable"]
        for _ in range(retries):
            azimuth = rng.uniform(0, math.tau)
            pose_reasons, _ = solve_pose(
                x, y, support_z, bottom_height, yaw, pitch, roll, distance,
                azimuth, cam_height, look_u, look_v)
            if ("camera_height_unreachable" not in pose_reasons
                    and "camera_off_field" not in pose_reasons):
                break
        reasons.extend(pose_reasons)
    if not reasons:
        bpy.context.view_layer.update()
        origin = armor.matrix_world.translation
        camera_origin = camera.matrix_world.translation
        actual_distance = (camera_origin - origin).length
        object_height = min((armor.matrix_world @ vertex).z for vertex in local_vectors) - support_z
        camera_ground = ground_at(camera_origin.x, camera_origin.y)
        camera_height = None if camera_ground is None else camera_origin.z - camera_ground["point"].z
        angles = measured_angles()
        if camera_height is None:
            reasons.append("camera_off_field")
        else:
            reasons.extend(constraints.metric_reasons(object_height, camera_height, actual_distance, config))
        if max(abs(a - b) for a, b in zip(angles, (yaw, pitch, roll))) > config["pose_angle_tolerance_deg"]:
            reasons.append("angle_drift")
        if abs(actual_distance - distance) > config["pose_distance_tolerance_m"]:
            reasons.append("distance_drift")
        if abs(object_height - bottom_height) > 0.005:
            reasons.append("height_drift")
        pixels = []
        for local in points_local:
            view = world_to_camera_view(scene, camera, armor.matrix_world @ local)
            pixels.append((view.x * width, (1 - view.y) * height, view.z))
        reasons.extend(constraints.projection_reasons(pixels, width, height, config["image_margin_px"],
                                                      config["min_projected_span_px"]))
        world_normal = armor.matrix_world.to_3x3() @ local_basis.col[2]
        center = sum((armor.matrix_world @ local for local in points_local), Vector()) / 4
        if world_normal.dot(camera_origin - center) <= 0:
            reasons.append("back_face")
        footprint_reasons, occluder, bottom = footprint_and_collision(region_name, support_z)
        reasons.extend(footprint_reasons)
        if not reasons:
            visible = evaluate_visibility(scene, camera, panel,
                                          [armor.matrix_world @ local for local in points_local],
                                          grid_size=config["visibility_grid"])
            screened = {"visibility": visible["visibility"],
                        "visible_area_min": visible["visible_area_fraction"],
                        "visible_corners_min": visible["visibility"].count(2)}
            reasons.extend(retention_reasons(screened, config["minimum_visible_area"],
                                             config["minimum_visible_corners"]))
        else:
            visible = None
            occluder = occluder
        record.update({"object_bottom_height_m": object_height,
                       "camera_height_m": camera_height, "actual_distance_m": actual_distance,
                       "measured_yaw_pitch_roll_deg": angles,
                       "support_z_m": support_z, "occluder": occluder,
                       "visibility": None if visible is None else visible["visibility"],
                       "visible_area_fraction": None if visible is None else visible["visible_area_fraction"],
                       "keypoints_normalized": [[p[0] / width, p[1] / height] for p in pixels]})
    record["reasons"] = sorted(set(reasons))
    if record["reasons"]:
        print("PLACEMENT_REJECTED", attempt, record["reasons"], flush=True)
    else:
        print("PLACEMENT_ACCEPTED", attempt, region_name, [round(v, 2) for v in angles], flush=True)
    return record



def validate():
    global output, source, source_hash, min_x, max_x, min_y, max_y, scanned
    global config, profile, cells, step
    rng = constraints.rng_from_config(config)
    accepted, rejected = [], []
    for attempt in range(config["attempts"]):
        record = attempt_once(rng, attempt)
        if record["reasons"]:
            rejected.append(record)
        else:
            accepted.append(record)
    report = {
        "status": "passed" if len(accepted) >= config["minimum_accepted"] else "needs_review",
        "source_sha256": source_hash,
        "source_unchanged": hashlib.sha256(source.read_bytes()).hexdigest() == source_hash,
        "config": config,
        "camera_profile": profile,
        "mask": {"floor_xy_m": [min_x, max_x, min_y, max_y], "step_m": step,
                 "cells": {name: len(items) for name, items in cells.items()},
                 "scanned": dict(scanned),
                 "definition": "object and material bindings in sampling_config.json; normal_z only rejects steep surfaces"},
        "sampling": "log-uniform distance, stratified yaw bins, rejection sampling; accepted set is not a uniform distribution",
        "checks": "discrete footprint, radial rays and visibility grid; not a continuous collision proof",
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "rejection_reasons": dict(Counter(reason for item in rejected for reason in item["reasons"])),
        "accepted_regions": dict(Counter(item["region"] for item in accepted)),
        "accepted_yaw_bins": dict(Counter(str(item["yaw_bin_deg"]) for item in accepted)),
        "accepted": accepted,
        "rejected": rejected,
    }
    (output / "placement_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PLACEMENT_RESULT", report["status"], report["accepted_count"], report["rejection_reasons"], output, flush=True)
    if not report["source_unchanged"] or report["status"] != "passed":
        raise RuntimeError(report["status"])


if __name__ == "__main__":
    prepare()
    validate()
