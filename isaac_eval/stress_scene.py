from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils, Vt
from realistic_scene import _bind, _box, _mesh, _validated_assets

from isaac_eval.stress_layout import build_layout, layout_sha256

_ROOT = "/World/VisualStress"
_PEOPLE = {
    "person_business": "F_Business_02",
    "person_worker": "male_adult_construction_03",
    "person_police": "female_adult_police_02",
}


def validate_stress_assets(asset_root):
    root = Path(asset_root).resolve()
    _, pavement_manifest_hash, pavement = _validated_assets(root)
    people_root = root / "stress_people"
    raw = (people_root / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if (
        manifest.get("version") != 1
        or manifest.get("default_prim") != "/Person"
        or manifest.get("meters_per_unit") != 1
        or manifest.get("up_axis") != "Z"
        or set(manifest.get("assets", {})) != set(_PEOPLE.values())
    ):
        raise ValueError("Invalid stress people manifest metadata")
    hashes = {str(Path(k)): v for k, v in pavement.items()}
    hashes["manifest.json"] = pavement_manifest_hash
    hashes["stress_people/manifest.json"] = hashlib.sha256(raw).hexdigest()
    for name, asset in manifest["assets"].items():
        bounds = np.asarray(asset["bounds_m"], dtype=float)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
            raise ValueError(f"Invalid bounds for {name}")
        size = bounds[1] - bounds[0]
        if not (
            1.6 <= size[2] <= 1.9
            and max(abs(bounds[:, :2]).flatten()) <= 0.7
            and abs(bounds[0, 2]) < 0.08
            and asset["meshes"] > 0
            and asset["joint_positions_m"]
            and asset["textures"]
        ):
            raise ValueError(f"Invalid scale, pose or contents for {name}: {bounds.tolist()}")
        joints = asset["joint_positions_m"]
        for side in ("L", "R"):
            shoulder = np.asarray(joints[f"{side}_Upperarm"], dtype=float)
            hand = np.asarray(joints[f"{side}_Hand"], dtype=float)
            if (
                shoulder.shape != (3,)
                or hand.shape != (3,)
                or not np.isfinite(shoulder).all()
                or not np.isfinite(hand).all()
                or shoulder[2] - hand[2] < 0.2
            ):
                raise ValueError(f"Person pose has raised or invalid arms: {name}")
        files = {asset["usd"]: asset["sha256"], **{str(Path(name) / k): v for k, v in asset["textures"].items()}}
        for rel, expected in files.items():
            path = (people_root / rel).resolve()
            if not path.is_relative_to(people_root) or not path.is_file():
                raise ValueError(f"Missing or nonlocal person asset: {rel}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected:
                raise ValueError(f"Person asset hash mismatch: {rel}")
            hashes[str(path.relative_to(root))] = digest
        usd_path = (people_root / asset["usd"]).resolve()
        person_stage = Usd.Stage.Open(str(usd_path))
        if (
            not person_stage
            or not person_stage.GetDefaultPrim()
            or str(person_stage.GetDefaultPrim().GetPath()) != "/Person"
            or UsdGeom.GetStageUpAxis(person_stage) != UsdGeom.Tokens.z
            or UsdGeom.GetStageMetersPerUnit(person_stage) != 1
        ):
            raise ValueError(f"Invalid person USD stage: {name}")
        meshes = [p for p in person_stage.Traverse() if p.IsA(UsdGeom.Mesh)]
        if len(meshes) != asset["meshes"]:
            raise ValueError(f"Person mesh count mismatch: {name}")
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        actual = cache.ComputeWorldBound(person_stage.GetDefaultPrim()).ComputeAlignedRange()
        actual_bounds = np.array([list(actual.GetMin()), list(actual.GetMax())])
        if (
            not np.isfinite(actual_bounds).all()
            or np.any(actual_bounds[0] < bounds[0] - 0.005)
            or np.any(actual_bounds[1] > bounds[1] + 0.005)
            or not 1.6 <= actual_bounds[1, 2] - actual_bounds[0, 2] <= 1.9
        ):
            raise ValueError(f"Person visible bounds exceed declared bounds or human scale: {name}")
        layers, dependencies, unresolved = UsdUtils.ComputeAllDependencies(str(usd_path))
        actual_files = {Path(str(dep)).resolve() for dep in dependencies}
        expected_files = {(people_root / name / rel).resolve() for rel in asset["textures"]}
        if unresolved or actual_files != expected_files or len(layers) != 1:
            raise ValueError(f"Person asset dependencies are incomplete or nonlocal: {name}")
    return manifest, hashlib.sha256(raw).hexdigest(), dict(sorted(hashes.items()))


def _cylinder(stage, path, x, y, z, radius, height, material):
    obj = UsdGeom.Cylinder.Define(stage, path)
    obj.CreateAxisAttr("Z")
    obj.CreateRadiusAttr(radius)
    obj.CreateHeightAttr(height)
    obj.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
    _bind(obj.GetPrim(), material)


def _tire(stage, path, x, y, base, material, groove):
    profile = ((0.23, 0.0), (0.40, 0.0), (0.43, 0.04), (0.43, 0.19), (0.40, 0.23), (0.23, 0.23))
    segments = 32
    points = [
        (x + radius * math.cos(math.tau * i / segments), y + radius * math.sin(math.tau * i / segments), base + z)
        for radius, z in profile
        for i in range(segments)
    ]
    faces = [
        [
            j * segments + i,
            j * segments + (i + 1) % segments,
            (j + 1) * segments + (i + 1) % segments,
            (j + 1) * segments + i,
        ]
        for j in range(len(profile) - 1)
        for i in range(segments)
    ]
    _mesh(stage, path + "/Body", points, faces, material)
    for i in range(16):
        angle = math.tau * i / 16
        _box(
            stage,
            f"{path}/Groove{i}",
            (x + 0.325 * math.cos(angle), y + 0.325 * math.sin(angle), base + 0.2305),
            (0.13, 0.008, 0.001),
            groove,
            math.degrees(angle),
        )


def _local_xy(x, y, yaw, dx, dy):
    return (x + dx * math.cos(yaw) - dy * math.sin(yaw), y + dx * math.sin(yaw) + dy * math.cos(yaw))


def _weathered_ground(stage, weather, source):
    path = f"{_ROOT}/Materials/WeatheredAsphalt"
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    st = UsdShade.Shader.Define(stage, path + "/ST")
    st.CreateIdAttr("UsdPrimvarReader_float2")
    st.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
    st.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    reader = UsdShade.Shader.Define(stage, path + "/Gain")
    reader.CreateIdAttr("UsdPrimvarReader_float4")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("stressGain")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float4)
    for role, target, output, typ in (
        ("diffuse", "diffuseColor", "rgb", Sdf.ValueTypeNames.Color3f),
        ("roughness", "roughness", "r", Sdf.ValueTypeNames.Float),
        ("normal_gl", "normal", "rgb", Sdf.ValueTypeNames.Normal3f),
    ):
        original = UsdShade.Shader.Get(stage, str(source.GetPath()) + "/" + role)
        texture = UsdShade.Shader.Define(stage, path + "/" + role)
        texture.CreateIdAttr("UsdUVTexture")
        for field in original.GetInputs():
            if field.GetBaseName() != "st" and field.Get() is not None:
                texture.CreateInput(field.GetBaseName(), field.GetTypeName()).Set(field.Get())
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        if role == "diffuse":
            texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).ConnectToSource(reader.ConnectableAPI(), "result")
        texture.CreateOutput(output, Sdf.ValueTypeNames.Float if output == "r" else Sdf.ValueTypeNames.Float3)
        shader.CreateInput(target, typ).ConnectToSource(texture.ConnectableAPI(), output)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    low, high = np.asarray(weather["bounds_m"], dtype=float)
    cells = np.ceil((high - low) / weather["spacing_m"]).astype(int)
    xs = np.linspace(low[0], high[0], cells[0] + 1)
    ys = np.linspace(low[1], high[1], cells[1] + 1)
    xx, yy = np.meshgrid(xs, ys)
    gain = np.zeros_like(xx)
    for field in weather["fields"]:
        cx, cy = field["center"]
        rx, ry = field["radius_m"]
        gain += field["gain"] * np.exp(-0.5 * (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2))
    edge = np.minimum.reduce((xx - low[0], high[0] - xx, yy - low[1], high[1] - yy))
    fade = np.clip(edge / weather["edge_fade_m"], 0.0, 1.0)
    fade = fade * fade * (3.0 - 2.0 * fade)
    gain = weather["base_gain"] * np.clip(1.0 + gain * fade, *weather["gain_limits"])
    points = [(float(x), float(y), 0.0015) for x, y in zip(xx.flat, yy.flat, strict=True)]
    stride = len(xs)
    faces = [
        [j * stride + i, j * stride + i + 1, (j + 1) * stride + i + 1, (j + 1) * stride + i]
        for j in range(len(ys) - 1)
        for i in range(len(xs) - 1)
    ]
    uv = [(points[k][0] / 2, points[k][1] / 2) for face in faces for k in face]
    mesh = _mesh(stage, f"{_ROOT}/Ground", points, faces, mat, uv)
    primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "stressGain", Sdf.ValueTypeNames.Float4Array, UsdGeom.Tokens.vertex
    )
    primvar.Set(Vt.Vec4fArray([Gf.Vec4f(float(v), float(v), float(v), 1.0) for v in gain.flat]))


def _ribbon(stage, path, item, mat, *, offset=0.0, branch=0.0, start=0.0, end=1.0):
    x, y = item["position"]
    length, width = item["size"]
    angle = item["yaw_rad"]
    along = (math.cos(angle), math.sin(angle))
    normal = (-along[1], along[0])
    points = []
    steps = 12
    for j in range(steps + 1):
        t = start + (end - start) * j / steps
        bend = (
            0.18 * math.sin(t * math.pi)
            if item["kind"] == "rubber"
            else 0.10 * math.sin(t * math.tau + item["variation"][0] * 4)
            if item["kind"] == "tar"
            else 0.0
        )
        side = offset + bend + branch * (t - start)
        cx = x + length * (t - 0.5) * along[0] + side * normal[0]
        cy = y + length * (t - 0.5) * along[1] + side * normal[1]
        half = width * 0.5 * item["variation"][j % 12]
        points.extend(
            [
                (cx - half * normal[0], cy - half * normal[1], 0.006),
                (cx + half * normal[0], cy + half * normal[1], 0.006),
            ]
        )
    faces = [[2 * j, 2 * j + 1, 2 * j + 3, 2 * j + 2] for j in range(steps)]
    _mesh(stage, path, points, faces, mat)


def _concrete_wall(stage, path, position, size, yaw, mat):
    x, y = position
    width, depth, height = size
    c, s = math.cos(yaw), math.sin(yaw)
    local = [
        (-width / 2, -depth / 2, 0),
        (width / 2, -depth / 2, 0),
        (width / 2, depth / 2, 0),
        (-width / 2, depth / 2, 0),
        (-width / 2, -depth / 2, height),
        (width / 2, -depth / 2, height),
        (width / 2, depth / 2, height),
        (-width / 2, depth / 2, height),
    ]
    points = [(x + a * c - b * s, y + a * s + b * c, z) for a, b, z in local]
    faces = [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 6, 7]]
    uv = []
    for face in faces:
        a, b, d = (points[face[i]] for i in (0, 1, 3))
        u = math.dist(a, b) / 2
        v = math.dist(a, d) / 2
        uv.extend([(0, 0), (u, 0), (u, v), (0, v)])
    _mesh(stage, path, points, faces, mat, uv)


def apply_visual_stress(stage, env, asset_root):
    """Dress an already matched scene and return its complete visual receipt."""
    before_rng = json.dumps(env.rng.bit_generator.state, sort_keys=True)
    people, manifest_hash, hashes = validate_stress_assets(asset_root)
    layout = build_layout(env)
    if stage.GetPrimAtPath(_ROOT):
        raise ValueError("Visual stress overlay already exists")
    UsdGeom.Xform.Define(stage, _ROOT)
    asphalt = UsdShade.Material.Get(stage, "/World/Appearance/Materials/asphalt_track")
    if not asphalt:
        raise ValueError("Matched scene and scanned pavement materials are required")

    def material(name, color, roughness=0.85, metallic=0.0):
        path = f"{_ROOT}/Materials/{name}"
        mat = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, path + "/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return mat

    mats = {
        "tar": material("Tar", (0.045, 0.043, 0.040)),
        "rubber": material("Rubber", (0.085, 0.080, 0.073)),
        "paint": material("OldPaint", (0.47, 0.43, 0.32)),
        "steel": material("Steel", (0.27, 0.29, 0.30), 0.52, 0.5),
        "blue": material("Blue", (0.025, 0.12, 0.56)),
        "yellow": material("Yellow", (0.82, 0.55, 0.025)),
        "dark": material("Dark", (0.055, 0.055, 0.050)),
        "groove": material("Groove", (0.035, 0.035, 0.033)),
        "concrete": material("Concrete", (0.43, 0.43, 0.39)),
        "wood": material("Wood", (0.32, 0.23, 0.14)),
    }
    repairs = stage.GetPrimAtPath("/World/Appearance/Repairs")
    if repairs:
        for child in repairs.GetChildren():
            if child.GetName().startswith("Patch"):
                UsdGeom.Imageable(child).MakeInvisible()
    _weathered_ground(stage, layout["weather"], asphalt)
    for i, item in enumerate(layout["surface"]):
        kind = item["kind"]
        path = f"{_ROOT}/Surface/{kind}{i}"
        if kind == "tar":
            _ribbon(stage, path, item, mats[kind])
            _ribbon(stage, path + "Branch", item, mats[kind], branch=0.5, start=0.35, end=0.8)
        elif kind == "rubber":
            _ribbon(stage, path + "A", item, mats[kind], offset=-0.32)
            _ribbon(stage, path + "B", item, mats[kind], offset=0.32)
        else:
            for j in range(3):
                _ribbon(stage, path + f"Dash{j}", item, mats[kind], start=j / 3, end=(j + 0.67) / 3)
    for i, obj in enumerate(layout["objects"]):
        kind = obj["kind"]
        x, y, _ = obj["position"]
        yaw = math.degrees(obj["yaw_rad"])
        path = f"{_ROOT}/Objects/O{i:04d}"
        if kind in _PEOPLE:
            xf = UsdGeom.Xform.Define(stage, path)
            xf.AddTranslateOp().Set(Gf.Vec3d(x, y, 0))
            xf.AddRotateZOp().Set(yaw)
            asset = people["assets"][_PEOPLE[kind]]
            xf.GetPrim().GetReferences().AddReference(
                str((Path(asset_root).resolve() / "stress_people" / asset["usd"]).resolve())
            )
        elif kind.startswith("barrel"):
            color = mats["blue" if kind.endswith("blue") else "yellow"]
            _cylinder(stage, path + "/Body", x, y, 0.46, 0.34, 0.92, color)
            for j, z in enumerate((0.04, 0.30, 0.62, 0.90)):
                _cylinder(stage, f"{path}/Rim{j}", x, y, z, 0.35, 0.025, color)
            _cylinder(stage, path + "/Bung", x + 0.14, y, 0.925, 0.045, 0.01, mats["steel"])
        elif kind == "tires":
            for j in range(3):
                _tire(stage, f"{path}/Tire{j}", x, y, 0.015 + j * 0.23, mats["dark"], mats["groove"])
        elif kind == "worktable":
            _box(stage, path + "/Top", (x, y, 0.83), (1.9, 0.9, 0.10), mats["wood"], yaw)
            for j, (dx, dy) in enumerate(((-0.8, -0.32), (-0.8, 0.32), (0.8, -0.32), (0.8, 0.32))):
                lx, ly = _local_xy(x, y, obj["yaw_rad"], dx, dy)
                _box(stage, f"{path}/Leg{j}", (lx, ly, 0.39), (0.1, 0.1, 0.78), mats["steel"], yaw)
        elif kind == "equipment":
            _box(stage, path + "/Body", (x, y, 0.60), (1.9, 0.9, 0.9), mats["blue"], yaw)
            _box(stage, path + "/Top", (x, y, 1.08), (1.9, 0.9, 0.08), mats["dark"], yaw)
            for j, (dx, dy) in enumerate(((-0.7, -0.3), (-0.7, 0.3), (0.7, -0.3), (0.7, 0.3))):
                wx, wy = _local_xy(x, y, obj["yaw_rad"], dx, dy)
                _cylinder(stage, f"{path}/Caster{j}", wx, wy, 0.075, 0.08, 0.15, mats["dark"])
            for j, dx in enumerate((-0.25, 0.25)):
                hx, hy = _local_xy(x, y, obj["yaw_rad"], dx, 0.0)
                _box(stage, f"{path}/HandlePost{j}", (hx, hy, 1.18), (0.035, 0.035, 0.16), mats["steel"], yaw)
            _box(stage, path + "/Handle", (x, y, 1.26), (0.54, 0.04, 0.04), mats["steel"], yaw)
        elif kind == "wall":
            concrete = UsdShade.Material.Get(stage, "/World/Appearance/Materials/concrete_floor_01")
            _concrete_wall(stage, path, (x, y), (*obj["footprint_size"], 1.55), obj["yaw_rad"], concrete)
        else:
            _box(stage, path, (x, y, 0.36), (*obj["footprint_size"], 0.72), mats["yellow"], yaw)
    if before_rng != json.dumps(env.rng.bit_generator.state, sort_keys=True):
        raise AssertionError("Stress scene consumed the environment RNG")
    counts = {
        kind: sum(o["kind"] == kind for o in layout["objects"])
        for kind in sorted({o["kind"] for o in layout["objects"]})
    }
    required = set(_PEOPLE) | {"wall", "barrier", "worktable", "tires", "barrel_blue", "barrel_yellow", "equipment"}
    if not required.issubset(counts):
        raise ValueError(f"Stress layout is missing required object categories: {sorted(required - counts.keys())}")
    return {
        "counts": counts,
        "variant": "visual_stress_v1",
        "layout_sha256": layout_sha256(layout),
        "weather_sha256": layout_sha256(layout["weather"]),
        "layout": layout,
        "objects": layout["objects"],
        "asset_manifest_sha256": manifest_hash,
        "asset_sha256": hashes,
    }
