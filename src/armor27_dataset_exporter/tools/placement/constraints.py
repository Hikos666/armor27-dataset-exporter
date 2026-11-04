"""Placement constraints shared by tests and the Blender sampler.

Angles match tools/motion_validation/test_motion.py: column-vector
R = Rz(roll) @ Rx(pitch) @ Ry(yaw), then armor = camera_basis @ R @ local_basis.T.
Yaw is about the camera up axis, pitch about camera right, roll about the view axis.
Zero yaw/pitch/roll puts the LED normal toward the camera.
"""

import math
import random


def mul(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def apply(m, v):
    return tuple(sum(m[i][k] * v[k] for k in range(3)) for i in range(3))


def transpose(m):
    return tuple(tuple(m[r][c] for r in range(3)) for c in range(3))


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def length(v):
    return math.sqrt(dot(v, v))


def normalize(v):
    scale = length(v)
    if scale < 1e-12:
        raise ValueError("zero vector")
    return (v[0] / scale, v[1] / scale, v[2] / scale)


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale(v, s):
    return (v[0] * s, v[1] * s, v[2] * s)


def rot_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))


def rot_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))


def rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def relative_rotation(yaw_deg, pitch_deg, roll_deg):
    yaw, pitch, roll = (math.radians(yaw_deg), math.radians(pitch_deg), math.radians(roll_deg))
    return mul(mul(rot_z(roll), rot_x(pitch)), rot_y(yaw))


def camera_basis(direction_to_target):
    """Columns are camera +X, +Y, +Z. Camera looks along -Z toward the target."""
    forward = normalize(direction_to_target)
    z_axis = scale(forward, -1.0)
    up = (0.0, 0.0, 1.0)
    if abs(dot(up, z_axis)) > 0.999:
        up = (0.0, 1.0, 0.0)
    y_axis = normalize(sub(up, scale(z_axis, dot(up, z_axis))))
    x_axis = cross(y_axis, z_axis)
    return (
        (x_axis[0], y_axis[0], z_axis[0]),
        (x_axis[1], y_axis[1], z_axis[1]),
        (x_axis[2], y_axis[2], z_axis[2]),
    )


def armor_rotation(camera_location, look_target, yaw_deg, pitch_deg, roll_deg, local_basis):
    basis = camera_basis(sub(look_target, camera_location))
    return mul(mul(basis, relative_rotation(yaw_deg, pitch_deg, roll_deg)), transpose(local_basis))


def measured_angles(camera_rotation, armor_rotation_matrix, local_basis):
    transform = mul(mul(transpose(camera_rotation), armor_rotation_matrix), local_basis)
    pitch = math.asin(max(-1.0, min(1.0, transform[2][1])))
    roll = math.atan2(-transform[0][1], transform[1][1])
    yaw = math.atan2(-transform[2][0], transform[2][2])
    return [math.degrees(yaw), math.degrees(pitch), math.degrees(roll)]


def bottom_offset(rotation, local_vertices):
    return min(apply(rotation, vertex)[2] for vertex in local_vertices)


def choose_region(rng, weights):
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("region weights must be positive")
    pick = rng.random() * total
    for name, weight in weights.items():
        pick -= weight
        if pick <= 0:
            return name
    return next(reversed(weights))


def sample_distance(rng, low, high, distribution):
    if distribution == "log_uniform":
        if low <= 0 or high <= 0:
            raise ValueError("log_uniform distance must be positive")
        return math.exp(rng.uniform(math.log(low), math.log(high)))
    if distribution == "uniform":
        return rng.uniform(low, high)
    raise ValueError(f"unknown distance distribution: {distribution}")


def sample_distance_at_least(rng, low, high, distribution, minimum):
    """Sample a legal distance that can still satisfy a known vertical gap."""
    low = max(low, minimum)
    if low > high:
        return None
    return sample_distance(rng, low, high, distribution)


def mark_interior(cells, step):
    """A cell is interior when the whole 3x3 block belongs to the same list."""
    keyed = {(round(cell["x"] / step), round(cell["y"] / step)) for cell in cells}
    for cell in cells:
        ix = round(cell["x"] / step)
        iy = round(cell["y"] / step)
        cell["interior"] = all(
            (ix + dx, iy + dy) in keyed
            for dx in (-1, 0, 1) for dy in (-1, 0, 1)
        )
    return cells


def choose_cell(rng, cells, interior_probability):
    interior = [cell for cell in cells if cell.get("interior")]
    if interior and rng.random() < interior_probability:
        return rng.choice(interior)
    return rng.choice(cells)


def sample_yaw(rng, bins, index):
    low, high = bins[index % len(bins)]
    return rng.uniform(low, high), bins[index % len(bins)]


def metric_reasons(object_height, camera_height, distance, limits):
    reasons = []
    if not (limits["object_bottom_height_m"][0] <= object_height <= limits["object_bottom_height_m"][1]):
        reasons.append("object_height")
    if not (limits["camera_height_m"][0] <= camera_height <= limits["camera_height_m"][1]):
        reasons.append("camera_height")
    if not (limits["distance_m"][0] <= distance <= limits["distance_m"][1]):
        reasons.append("distance")
    return reasons


def angle_reasons(yaw, pitch, roll, limits):
    reasons = []
    if not (limits["yaw_deg"][0] <= yaw <= limits["yaw_deg"][1]):
        reasons.append("yaw")
    if not (limits["pitch_deg"][0] <= pitch <= limits["pitch_deg"][1]):
        reasons.append("pitch")
    if not (limits["roll_deg"][0] <= roll <= limits["roll_deg"][1]):
        reasons.append("roll")
    return reasons


def projection_reasons(pixels, width, height, margin_px, min_span_px):
    """pixels are image coordinates with a positive depth in column 2."""
    reasons = []
    xs = [p[0] for p in pixels]
    ys = [p[1] for p in pixels]
    if any(p[2] <= 0 for p in pixels):
        reasons.append("behind_camera")
    if any(not (margin_px <= p[0] < width - margin_px and margin_px <= p[1] < height - margin_px) for p in pixels):
        reasons.append("out_of_frame")
    if max(xs) - min(xs) < min_span_px or max(ys) - min(ys) < min_span_px:
        reasons.append("projection_too_small")
    return reasons


def horizontal_offset(distance, dz):
    remaining = distance * distance - dz * dz
    if remaining < 0:
        return None
    return math.sqrt(remaining)


def load_config(path):
    import json
    from pathlib import Path
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rng_from_config(config):
    return random.Random(config["seed"])
