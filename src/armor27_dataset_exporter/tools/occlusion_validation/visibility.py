import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


def first_camera_hit(scene, depsgraph, origin, target, tolerance):
    direction = target - origin
    distance = direction.length
    direction.normalize()
    start = origin.copy()
    for _ in range(64):
        remaining = distance + tolerance - (start - origin).length
        if remaining <= 0:
            return None, None
        hit, location, _, _, obj, _ = scene.ray_cast(depsgraph, start, direction, distance=remaining)
        if not hit:
            return None, None
        if obj.hide_render or not obj.visible_camera:
            start = location + direction * tolerance
            continue
        return obj.original, (location - origin).length
    raise RuntimeError("Too many camera-invisible surfaces in visibility ray")


def evaluate_visibility(scene, camera, panel, points_world, grid_size=24):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    origin = camera.matrix_world.translation
    projected = [world_to_camera_view(scene, camera, point) for point in points_world]
    width = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
    height = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)
    center = sum(points_world, Vector()) / 4
    normal = (points_world[3] - points_world[0]).cross(points_world[0] - points_world[1]).normalized()
    front_facing = normal.dot(origin - center) > 0
    pixel_footprint_m = (origin - center).length * camera.data.sensor_width / (camera.data.lens * width)
    tolerance = max(1e-6, pixel_footprint_m * 0.05)
    states, hits = [], []
    for point, projected_point in zip(points_world, projected):
        if not front_facing or projected_point.z <= camera.data.clip_start or not (
                0 <= projected_point.x <= 1 and 0 <= projected_point.y <= 1):
            states.append(0)
            hits.append(["invalid_projection"])
            continue
        neighbors, neighbor_hits = [], []
        for fraction in (0.0002, 0.0005, 0.001):
            target = point.lerp(center, fraction)
            hit_obj, hit_depth = first_camera_hit(scene, depsgraph, origin, target, tolerance)
            target_depth = (target - origin).length
            if hit_obj == panel and abs(hit_depth - target_depth) <= tolerance:
                neighbors.append(2)
            elif hit_obj is not None and hit_depth < target_depth - tolerance:
                neighbors.append(1)
            else:
                neighbors.append(0)
            neighbor_hits.append(hit_obj.name if hit_obj else "ray_miss")
        states.append(0 if 0 in neighbors else (1 if 1 in neighbors else 2))
        hits.append(neighbor_hits)

    polygon = [Vector((p.x, 1 - p.y)) for p in projected]
    low_x, high_x = min(p.x for p in polygon), max(p.x for p in polygon)
    low_y, high_y = min(p.y for p in polygon), max(p.y for p in polygon)
    inverse_projection = camera.calc_matrix_camera(depsgraph, x=width, y=height,
        scale_x=scene.render.pixel_aspect_x, scale_y=scene.render.pixel_aspect_y).inverted()
    visible, total = 0, 0
    if front_facing and all(p.z > camera.data.clip_start for p in projected):
        for row in range(grid_size):
            y = low_y + (row + 0.5) * (high_y - low_y) / grid_size
            for col in range(grid_size):
                x = low_x + (col + 0.5) * (high_x - low_x) / grid_size
                crosses = [(polygon[(i + 1) % 4].x - polygon[i].x) * (y - polygon[i].y) -
                           (polygon[(i + 1) % 4].y - polygon[i].y) * (x - polygon[i].x) for i in range(4)]
                if not (all(c >= -1e-10 for c in crosses) or all(c <= 1e-10 for c in crosses)):
                    continue
                total += 1
                if not (0 <= x <= 1 and 0 <= y <= 1):
                    continue
                view = inverse_projection @ Vector((2 * x - 1, 1 - 2 * y, 0, 1))
                view /= view.w
                direction = camera.matrix_world @ view.to_3d() - origin
                denominator = direction.dot(normal)
                if abs(denominator) < 1e-12:
                    continue
                target = origin + direction * ((center - origin).dot(normal) / denominator)
                hit_obj, hit_depth = first_camera_hit(scene, depsgraph, origin, target, tolerance)
                if hit_obj == panel and abs(hit_depth - (target - origin).length) <= tolerance:
                    visible += 1
    return {"visibility": states, "corner_hits": hits,
            "visible_area_fraction": visible / total if total else 0,
            "area_samples": total, "area_grid": grid_size,
            "normalized_corners": [[p.x, 1 - p.y] for p in projected],
            "front_facing": front_facing, "depth_tolerance_m": tolerance}


def aggregate_exposure(observations):
    states = []
    for index in range(4):
        values = [o["visibility"][index] for o in observations]
        states.append(0 if 0 in values else (2 if all(v == 2 for v in values) else 1))
    return {"visibility": states,
            "visible_time_fractions": [sum(o["visibility"][i] == 2 for o in observations) / len(observations)
                                       for i in range(4)],
            "visible_area_min": min(o["visible_area_fraction"] for o in observations),
            "visible_corners_min": min(o["visibility"].count(2) for o in observations),
            "evaluated_times": len(observations)}


def retention_reasons(result, minimum_area=0.3, minimum_visible_corners=2):
    reasons = []
    if 0 in result["visibility"]:
        reasons.append("invalid_corner")
    if result["visible_area_min"] < minimum_area:
        reasons.append("insufficient_visible_area")
    if (result["visible_corners_min"] < minimum_visible_corners or
            result["visibility"].count(2) < minimum_visible_corners):
        reasons.append("insufficient_visible_corners")
    return reasons
