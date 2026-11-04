import math
from collections import Counter


KINDS = ("light_bar", "bullet_mini", "bullet_large")


def sample_count(rng, limits):
    return rng.randint(int(limits[0]), int(limits[1]))


def sample_height(rng, config):
    ground = rng.random() < config["ground_probability"]
    low, high = config["ground_bottom_height_m" if ground else "air_bottom_height_m"]
    return low + (high - low) * rng.random(), "ground" if ground else "air"


def sample_emission(rng, baseline, height, config):
    low, high = config["emission_multiplier"]
    bright = baseline * math.exp(rng.uniform(math.log(low), math.log(high)))
    dimmed = height < config["dim_below_m"] and rng.random() < config["dim_probability"]
    factor = rng.betavariate(1.0, config["dim_beta"]) if dimmed else 1.0
    return {"bright_strength": bright, "dimmed": dimmed,
            "dim_factor": factor, "emission_strength": bright * factor}


def validate_config(config):
    if not config.get("enabled"):
        return
    for kind in KINDS:
        low, high = config[kind]["count"]
        if int(low) != low or int(high) != high or not 0 <= low <= high:
            raise ValueError(f"Invalid {kind} count range")
    bullets = config["bullets"]
    for key in ("ground_probability", "dim_probability"):
        if not 0 <= bullets[key] <= 1:
            raise ValueError(f"Invalid {key}")
    for limits in (config["light_bar"]["bottom_height_m"],
                   bullets["ground_bottom_height_m"], bullets["air_bottom_height_m"]):
        if not 0 <= limits[0] <= limits[1]:
            raise ValueError("Invalid bottom height range")
    if not 0 < config["depth_m"][0] <= config["depth_m"][1]:
        raise ValueError("Invalid distractor depth range")
    if not 0 < bullets["emission_multiplier"][0] <= bullets["emission_multiplier"][1]:
        raise ValueError("Emission multiplier must be positive")
    if bullets["dim_beta"] <= 2 or config["placement_retries"] < 1:
        raise ValueError("Expected nonlinear decreasing beta density and positive retries")


def recipe_issues(recipe, config):
    issues = []
    instances = recipe["instances"]
    names = [entry["object"] for entry in instances]
    if len(names) != len(set(names)):
        issues.append("duplicate distractor objects")
    counts = Counter(entry["kind"] for entry in instances)
    if set(counts) - set(KINDS):
        issues.append("unknown distractor kind")
    if not config.get("enabled"):
        return issues + (["distractors present while disabled"] if instances else [])
    for kind in KINDS:
        low, high = config[kind]["count"]
        requested = recipe["requested"].get(kind, -1)
        placed = recipe["placed"].get(kind, -1)
        if not low <= requested <= high or not 0 <= placed <= requested:
            issues.append(f"invalid {kind} counts")
        if counts[kind] != placed:
            issues.append(f"{kind} instance count mismatch")
        visible = recipe.get("visible_sampled", {}).get(kind, 0)
        if not 0 <= visible <= placed:
            issues.append(f"invalid {kind} visible count")
    bullets = config["bullets"]
    for entry in instances:
        kind = entry["kind"]
        if kind not in KINDS:
            continue
        height = entry["bottom_height_m"]
        if kind == "light_bar":
            low, high = config[kind]["bottom_height_m"]
            if entry["material"] not in config[kind]["materials"]:
                issues.append("unknown light-bar material")
        else:
            if entry["height_band"] not in ("ground", "air"):
                issues.append("unknown bullet height band")
                continue
            low, high = bullets[entry["height_band"] + "_bottom_height_m"]
            bright, strength, factor = (entry[k] for k in ("bright_strength", "emission_strength", "dim_factor"))
            if not 0 <= factor <= 1 or bright <= 0 or not math.isclose(strength, bright * factor):
                issues.append("invalid bullet emission")
            if entry["dimmed"] and height >= bullets["dim_below_m"]:
                issues.append("dimmed bullet above threshold")
            if not entry["dimmed"] and factor != 1:
                issues.append("undimmed bullet has reduced strength")
        if not low - 1e-6 <= height <= high + 1e-6:
            issues.append(f"{kind} height outside range")
    return issues


def exposure_frames(scene, motion_state):
    if not motion_state["enabled"]:
        return [float(scene.frame_current) + scene.frame_subframe]
    shutter = motion_state["shutter_frames"]
    return [2 + shutter * fraction for fraction in (-0.5, -0.25, 0, 0.25, 0.5)]


def set_frame(scene, frame):
    whole = math.floor(frame)
    scene.frame_set(whole, subframe=frame - whole)


def measure_visibility(placement, roots_and_panels, motion_state):
    from visibility import aggregate_exposure, evaluate_visibility
    scene = placement.scene
    saved_frame = scene.frame_current + scene.frame_subframe
    observations = [[] for _ in roots_and_panels]
    try:
        for frame in exposure_frames(scene, motion_state):
            set_frame(scene, frame)
            for samples, (root, panel) in zip(observations, roots_and_panels):
                samples.append(evaluate_visibility(
                    scene, placement.camera, panel,
                    [root.matrix_world @ local for local in placement.points_local],
                    grid_size=placement.config["visibility_grid"]))
    finally:
        set_frame(scene, saved_frame)
    return [aggregate_exposure(samples) for samples in observations]


class DistractorPool:
    def __init__(self, placement, config):
        import bpy
        validate_config(config)
        self.bpy = bpy
        self.placement = placement
        self.config = config
        self.items = {kind: [] for kind in KINDS}
        self.templates = {}
        self.baselines = {}
        if not config.get("enabled"):
            for kind in KINDS:
                name = config.get(kind, {}).get("object")
                source = bpy.data.objects.get(name) if name else None
                if source is not None:
                    self._hide(source, True)
            return
        for name, preset in config["light_bar"].get("material_presets", {}).items():
            if name not in bpy.data.materials:
                material = bpy.data.materials[preset["template"]].copy()
                material.name = name
                node = next(n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
                for socket, value in preset["inputs"].items():
                    node.inputs[socket].default_value = value
        for kind in KINDS:
            source = bpy.data.objects.get(config[kind]["object"])
            if source is None or source.name not in placement.scene.objects:
                raise RuntimeError(f"Missing distractor template: {config[kind]['object']}")
            if not source.material_slots or source.material_slots[0].material is None:
                raise RuntimeError(f"Missing distractor material: {source.name}")
            self.templates[kind] = source
            if kind != "light_bar":
                mat = source.material_slots[0].material
                node = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
                if node.inputs["Emission Strength"].is_linked:
                    raise RuntimeError(f"Linked emission strength: {mat.name}")
                self.baselines[kind] = node.inputs["Emission Strength"].default_value
            for index in range(config[kind]["count"][1]):
                obj = source.copy()
                obj.name = f"batch_{kind}_{index:03d}"
                placement.scene.collection.objects.link(obj)
                obj.parent = None
                obj.matrix_world = source.matrix_world.copy()
                obj.rotation_mode = "XYZ"
                obj.material_slots[0].link = "OBJECT"
                if kind != "light_bar":
                    material = source.material_slots[0].material.copy()
                    material.name = obj.name + "_material"
                    obj.material_slots[0].material = material
                self.items[kind].append(obj)
            self._hide(source, True)
        for name in config["light_bar"]["materials"]:
            if name not in bpy.data.materials:
                raise RuntimeError(f"Missing light-bar material: {name}")
        self._build_ground()
        self.hide_all()

    def _build_ground(self):
        # Field-only rays avoid height drift from stepping through moving distractors.
        from mathutils.bvhtree import BVHTree
        vertices, faces, self.ground_faces = [], [], []
        graph = self.bpy.context.evaluated_depsgraph_get()
        for obj in sorted(self.placement.field_objects, key=lambda item: item.name):
            if obj.type != "MESH" or obj.hide_render:
                continue
            evaluated = obj.evaluated_get(graph)
            mesh = evaluated.to_mesh()
            try:
                offset = len(vertices)
                vertices.extend(evaluated.matrix_world @ v.co for v in mesh.vertices)
                for polygon in mesh.polygons:
                    faces.append(tuple(offset + index for index in polygon.vertices))
                    material = (obj.material_slots[polygon.material_index].material
                                if polygon.material_index < len(obj.material_slots) else None)
                    self.ground_faces.append((obj.name, material.name if material else None))
            finally:
                evaluated.to_mesh_clear()
        self.ground_bvh = BVHTree.FromPolygons(vertices, faces)

    def _ground_at(self, x, y):
        from mathutils import Vector
        point, normal, index, _ = self.ground_bvh.ray_cast(Vector((x, y, 30)), Vector((0, 0, -1)))
        if point is None:
            return None
        name, material = self.ground_faces[index]
        return {"point": point, "normal": normal, "object": name, "material": material}

    def _hide(self, obj, hidden):
        obj.hide_render = hidden
        obj.hide_set(hidden)

    def hide_all(self):
        for objects in self.items.values():
            for obj in objects:
                self._hide(obj, True)

    def _bounds(self, objects):
        from bpy_extras.object_utils import world_to_camera_view
        from mathutils import Vector
        points = []
        graph = self.bpy.context.evaluated_depsgraph_get()
        for obj in objects:
            if obj.type not in ("MESH", "SURFACE", "CURVE"):
                continue
            evaluated = obj.evaluated_get(graph)
            for corner in evaluated.bound_box:
                world = obj.matrix_world @ Vector(corner)
                point = world_to_camera_view(self.placement.scene, self.placement.camera, world)
                points.append(point)
        if not points or min(p.z for p in points) <= self.placement.camera.data.clip_start:
            return None
        return (min(p.x for p in points), max(p.x for p in points),
                min(p.y for p in points), max(p.y for p in points),
                min(p.z for p in points), max(p.z for p in points))

    def plate_bounds(self, roots, motion_state):
        scene = self.placement.scene
        saved_frame = scene.frame_current + scene.frame_subframe
        all_bounds = [[] for _ in roots]
        try:
            for frame in exposure_frames(scene, motion_state):
                set_frame(scene, frame)
                for bounds, root in zip(all_bounds, roots):
                    value = self._bounds([root, *root.children_recursive])
                    if value is not None:
                        bounds.append(value)
        finally:
            set_frame(scene, saved_frame)
        return [(min(b[0] for b in bounds), max(b[1] for b in bounds),
                 min(b[2] for b in bounds), max(b[3] for b in bounds),
                 min(b[4] for b in bounds), max(b[5] for b in bounds))
                for bounds in all_bounds if bounds]

    def lamp_blocks(self, bounds, targets):
        pad_x = 2 / self.placement.width
        pad_y = 2 / self.placement.height
        for target in targets:
            overlaps = (bounds[0] < target[1] + pad_x and bounds[1] > target[0] - pad_x and
                        bounds[2] < target[3] + pad_y and bounds[3] > target[2] - pad_y)
            if overlaps and bounds[4] <= target[5] + 0.002:
                return True
        return False

    def _position(self, obj, kind, rng):
        from mathutils import Vector
        placement = self.placement
        camera = placement.camera
        scene = placement.scene
        graph = self.bpy.context.evaluated_depsgraph_get()
        projection = camera.calc_matrix_camera(graph, x=placement.width, y=placement.height)
        u, v = rng.uniform(0.02, 0.98), rng.uniform(0.02, 0.98)
        low, high = self.config["depth_m"]
        depth = math.exp(rng.uniform(math.log(low), math.log(high)))
        local = Vector(((2 * u - 1) * depth / projection[0][0],
                        (2 * v - 1) * depth / projection[1][1], -depth))
        candidate = camera.matrix_world @ local
        support = self._ground_at(candidate.x, candidate.y)
        if placement.region_of(support) is None:
            return None, "off_field"
        if kind == "light_bar":
            height = rng.uniform(*self.config[kind]["bottom_height_m"])
            band = "light_bar"
        else:
            height, band = sample_height(rng, self.config["bullets"])
        source = self.templates[kind]
        obj.rotation_euler = source.matrix_world.to_euler("XYZ")
        obj.rotation_euler.z = rng.uniform(0, math.tau)
        obj.scale = source.matrix_world.to_scale()
        obj.location = (candidate.x, candidate.y, support["point"].z)
        self.bpy.context.view_layer.update()
        evaluated = obj.evaluated_get(self.bpy.context.evaluated_depsgraph_get())
        offsets = [obj.matrix_world @ Vector(corner) for corner in evaluated.bound_box]
        obj.location.z += support["point"].z + height - min(p.z for p in offsets)
        self.bpy.context.view_layer.update()
        return {"bottom_height_m": height, "support_z_m": support["point"].z,
                "height_band": band, "yaw_rad": obj.rotation_euler.z}, None

    def sample(self, rng, roots, motion_state):
        self.hide_all()
        recipe = {"instances": [], "requested": {}, "placed": {}, "rejected": {}}
        if not self.config.get("enabled"):
            return recipe
        targets = self.plate_bounds(roots, motion_state)
        for kind in KINDS:
            count = sample_count(rng, self.config[kind]["count"])
            recipe["requested"][kind] = count
            failures = Counter()
            placed = 0
            for obj in self.items[kind][:count]:
                self._hide(obj, False)
                accepted = None
                for _ in range(self.config["placement_retries"]):
                    info, failure = self._position(obj, kind, rng)
                    if failure is None:
                        bounds = self._bounds([obj])
                        if bounds is None or not (0 <= bounds[0] < bounds[1] <= 1 and
                                                  0 <= bounds[2] < bounds[3] <= 1):
                            failure = "out_of_frame"
                        elif kind == "light_bar" and self.lamp_blocks(bounds, targets):
                            failure = "light_bar_blocks_plate"
                    if failure:
                        failures[failure] += 1
                        continue
                    accepted = info
                    break
                if accepted is None:
                    self._hide(obj, True)
                    continue
                if kind == "light_bar":
                    name = rng.choice(self.config[kind]["materials"])
                    obj.material_slots[0].material = self.bpy.data.materials[name]
                else:
                    material = obj.material_slots[0].material
                    emission = sample_emission(rng, self.baselines[kind],
                                               accepted["bottom_height_m"], self.config["bullets"])
                    self._set_emission(material, emission["emission_strength"])
                    accepted.update(emission)
                accepted.update({"kind": kind, "object": obj.name,
                                 "material": obj.material_slots[0].material.name,
                                 "matrix": [list(row) for row in obj.matrix_world]})
                recipe["instances"].append(accepted)
                placed += 1
            recipe["placed"][kind] = placed
            recipe["rejected"][kind] = dict(failures)
        self.bpy.context.view_layer.update()
        return recipe

    def _set_emission(self, material, strength):
        node = next(n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        node.inputs["Emission Strength"].default_value = strength

    def restore(self, recipe):
        from mathutils import Matrix
        self.hide_all()
        owned = {o.name: o for objects in self.items.values() for o in objects}
        for entry in recipe["instances"]:
            obj = owned[entry["object"]]
            obj.matrix_world = Matrix(entry["matrix"])
            obj.material_slots[0].material = self.bpy.data.materials[entry["material"]]
            if "emission_strength" in entry:
                self._set_emission(obj.material_slots[0].material, entry["emission_strength"])
            self._hide(obj, False)
        self.bpy.context.view_layer.update()

    def visible_counts(self):
        from mathutils import Vector
        from bpy_extras.object_utils import world_to_camera_view
        from visibility import first_camera_hit
        scene = self.placement.scene
        camera = self.placement.camera
        graph = self.bpy.context.evaluated_depsgraph_get()
        result = {}
        for kind, objects in self.items.items():
            visible = 0
            for obj in objects:
                if obj.hide_render:
                    continue
                evaluated = obj.evaluated_get(graph)
                corners = [Vector(c) for c in evaluated.bound_box]
                center = sum(corners, Vector()) / len(corners)
                samples = [center, *[center.lerp(corner, 0.5) for corner in corners]]
                for local in samples:
                    target = obj.matrix_world @ local
                    point = world_to_camera_view(scene, camera, target)
                    if point.z <= 0 or not (0 <= point.x <= 1 and 0 <= point.y <= 1):
                        continue
                    hit_obj, _ = first_camera_hit(
                        scene, graph, camera.matrix_world.translation, target, 1e-5)
                    if hit_obj == obj:
                        visible += 1
                        break
            result[kind] = visible
        return result


def configure_compositor(placement, pool, panels):
    import bpy
    scene = placement.scene
    tree = scene.compositing_node_group
    if tree is None:
        raise RuntimeError("Source scene has no compositor node group")
    tree = tree.copy()
    tree.name = "batch_distractor_compositor"
    scene.compositing_node_group = tree
    required = ("Cryptomatte", "Cryptomatte.001", "眩光", "眩光.001", "混合", "混合.001", "组输出")
    if any(tree.nodes.get(name) is None for name in required):
        raise RuntimeError("Expected light-bar and bullet compositor branches in source scene")
    names = [obj.name for obj in pool.items["light_bar"]] + [panel.name for panel in panels]
    tree.nodes["Cryptomatte"].matte_id = ",".join(names)
    materials = [obj.material_slots[0].material.name
                 for kind in ("bullet_mini", "bullet_large") for obj in pool.items[kind]]
    tree.nodes["Cryptomatte.001"].matte_id = ",".join(materials)
    for layer in scene.view_layers:
        layer.use_pass_cryptomatte_object = True
        layer.use_pass_cryptomatte_material = True
    for glare, mix in (("眩光", "混合"), ("眩光.001", "混合.001")):
        node = tree.nodes[mix]
        socket = next(s for s in node.inputs if s.name == "B" and s.type == "RGBA")
        tree.links.new(tree.nodes[glare].outputs["Glare"], socket)
    output = tree.nodes["混合.001"].outputs["Result"]
    tree.links.new(output, tree.nodes["组输出"].inputs["图像"])
    for node in tree.nodes:
        if node.bl_idname == "CompositorNodeViewer":
            tree.links.new(output, node.inputs["Image"])
    return {"node_group": tree.name, "light_bar_and_panel_objects": names,
            "bullet_materials": materials, "glare_only": True}
