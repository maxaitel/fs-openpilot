"""Render-only Formula Student test venue for the frozen-model evaluator.

Call after the baseline scene is built and before world.reset(). The original
rigid bodies, collision shapes and physics materials remain authoritative.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt
from realistic_car import dress_car

from fsop.track.rules import RULES
from fsop.track.schema import ConeColor

_ROOT = "/World/Appearance"
_ASSETS = ("asphalt_track", "asphalt_02", "concrete_floor_01", "gravel_floor_04", "derelict_airfield_01")


def _validated_assets(root):
    root = Path(root).resolve()
    raw = (root / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    used = {}
    for name in _ASSETS:
        item = manifest["assets"][name]
        if item["license"] != "CC0-1.0":
            raise ValueError(f"Unexpected license for {name}")
        for info in item["files"].values():
            path = (root / info["path"]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"Missing or invalid asset: {info['path']}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != info["sha256"] or path.stat().st_size != info["size"]:
                raise ValueError(f"Asset verification failed: {info['path']}")
            used[info["path"]] = digest
    return manifest, hashlib.sha256(raw).hexdigest(), used


def _material(stage, name, color, roughness=0.8, metallic=0.0):
    path = f"{_ROOT}/Materials/{name}"
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _pbr(stage, name, root, *, albedo_gain=1.0):
    path = f"{_ROOT}/Materials/{name}"
    mat = UsdShade.Material.Define(stage, path)
    surf = UsdShade.Shader.Define(stage, path + "/Surface")
    surf.CreateIdAttr("UsdPreviewSurface")
    surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    reader = UsdShade.Shader.Define(stage, path + "/ST")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    for role, source, output, typ in (
        ("diffuse", "diffuseColor", "rgb", Sdf.ValueTypeNames.Color3f),
        ("roughness", "roughness", "r", Sdf.ValueTypeNames.Float),
        ("normal_gl", "normal", "rgb", Sdf.ValueTypeNames.Normal3f),
    ):
        tex = UsdShade.Shader.Define(stage, path + "/" + role)
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(root / name / f"{role}.jpg")))
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB" if role == "diffuse" else "raw")
        if role == "diffuse" and albedo_gain != 1.0:
            tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(albedo_gain, albedo_gain, albedo_gain, 1))
        if role == "normal_gl":
            tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
            tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
        tex.CreateOutput(output, Sdf.ValueTypeNames.Float if output == "r" else Sdf.ValueTypeNames.Float3)
        surf.CreateInput(source, typ).ConnectToSource(tex.ConnectableAPI(), output)
    mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
    return mat


def _bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def _mesh(stage, path, points, faces, mat, uv=None, normals=None):
    obj = UsdGeom.Mesh.Define(stage, path)
    obj.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]))
    obj.CreateFaceVertexCountsAttr([len(face) for face in faces])
    obj.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
    obj.CreateSubdivisionSchemeAttr("none")
    obj.CreateDoubleSidedAttr(True)
    if uv is not None:
        primvar = UsdGeom.PrimvarsAPI(obj).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        )
        primvar.Set(Vt.Vec2fArray([Gf.Vec2f(*p) for p in uv]))
    if normals is not None:
        obj.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in normals]))
        obj.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    _bind(obj.GetPrim(), mat)
    return obj


def _rect(stage, path, x0, y0, x1, y1, z, mat, tile=1.0):
    points = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
    return _mesh(stage, path, points, [[0, 1, 2, 3]], mat, [(x / tile, y / tile) for x, y, _ in points])


def _box(stage, path, pos, size, mat, rotation=0.0):
    obj = UsdGeom.Cube.Define(stage, path)
    obj.CreateSizeAttr(1.0)
    obj.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    if rotation:
        obj.AddRotateZOp().Set(float(rotation))
    obj.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
    _bind(obj.GetPrim(), mat)
    return obj


def _pole(stage, path, x, y, height, radius, mat):
    obj = UsdGeom.Cylinder.Define(stage, path)
    obj.CreateAxisAttr("Z")
    obj.CreateHeightAttr(height)
    obj.CreateRadiusAttr(radius)
    obj.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), height / 2))
    _bind(obj.GetPrim(), mat)


def _cone(stage, path, spec, material, white, base, grime, index):
    n = 32
    # Existing cone mesh spans z=.02 to .02+height in its rigid-body frame.
    levels = [
        (0.02, spec.radius_bottom),
        (0.02 + spec.height * 0.48, spec.radius_bottom * 0.55),
        (0.02 + spec.height * 0.62, spec.radius_bottom * 0.43),
        (0.02 + spec.height * 0.97, spec.radius_top * 1.08),
        (0.02 + spec.height, spec.radius_top),
    ]
    points, normals = [], []
    slope = (spec.radius_bottom - spec.radius_top) / spec.height
    for z, radius in levels:
        for j in range(n):
            a = 2 * math.pi * j / n
            points.append((radius * math.cos(a), radius * math.sin(a), z))
            normal = np.array([math.cos(a), math.sin(a), slope])
            normals.append(tuple(normal / np.linalg.norm(normal)))
    faces, uv = [], []
    for level in range(len(levels) - 1):
        for j in range(n):
            faces.append([level * n + j, level * n + (j + 1) % n, (level + 1) * n + (j + 1) % n, (level + 1) * n + j])
            u0, u1 = j / n, (j + 1) / n
            v0 = (levels[level][0] - 0.02) / spec.height
            v1 = (levels[level + 1][0] - 0.02) / spec.height
            uv.extend([(u0, v0), (u1, v0), (u1, v1), (u0, v1)])
    _mesh(stage, path + "/Shell", points, faces, material, uv, normals)
    band = UsdGeom.Subset.Define(stage, path + "/Shell/WhiteBand")
    band.CreateElementTypeAttr("face")
    band.CreateIndicesAttr(list(range(n, 2 * n)))
    band.CreateFamilyNameAttr("materialBind")
    _bind(band.GetPrim(), white)
    # Truncated corners and a sloping rim retain the original square footprint.
    half = spec.base / 2
    footprint = [(-1, -0.78), (-0.78, -1), (0.78, -1), (1, -0.78), (1, 0.78), (0.78, 1), (-0.78, 1), (-1, 0.78)]
    foot_points = [
        (x * half * scale, y * half * scale, z)
        for scale, z in ((1, -0.02), (1, 0.011), (0.91, 0.02))
        for x, y in footprint
    ]
    foot_faces = [[j, (j + 1) % 8, 8 + (j + 1) % 8, 8 + j] for j in range(8)]
    foot_faces += [[8 + j, 8 + (j + 1) % 8, 16 + (j + 1) % 8, 16 + j] for j in range(8)]
    foot_faces += [list(range(16, 24))]
    _mesh(stage, path + "/Foot", foot_points, foot_faces, base)
    # The dark inset and narrow lip suggest the hollow molded tip.
    z = 0.02 + spec.height
    lip_points = [
        (radius * math.cos(2 * math.pi * j / n), radius * math.sin(2 * math.pi * j / n), height)
        for radius, height in ((spec.radius_top, z), (spec.radius_top * 0.63, z), (spec.radius_top * 0.63, z - 0.009))
        for j in range(n)
    ]
    lip_faces = [[j, (j + 1) % n, n + (j + 1) % n, n + j] for j in range(n)]
    lip_faces += [[n + j, n + (j + 1) % n, 2 * n + (j + 1) % n, 2 * n + j] for j in range(n)]
    _mesh(stage, path + "/Lip", lip_points, lip_faces, material)
    _mesh(stage, path + "/Opening", lip_points[2 * n :], [list(range(n))], base)
    # Restrained molded-plastic scuffs, varied by cone index and below band scale.
    scuff_points, scuff_faces = [], []
    for j, fraction in enumerate((0.16, 0.30, 0.73)):
        angle = (index * 1.71 + j * 2.27) % (2 * math.pi)
        z0 = 0.02 + spec.height * fraction
        dz = spec.height * 0.018
        r0 = float(np.interp(z0, [level[0] for level in levels], [level[1] for level in levels])) + 0.00015
        r1 = float(np.interp(z0 + dz, [level[0] for level in levels], [level[1] for level in levels])) + 0.00015
        for radius, theta, height in (
            (r0, angle, z0),
            (r0, angle + 0.13, z0),
            (r1, angle + 0.11, z0 + dz),
            (r1, angle + 0.02, z0 + dz),
        ):
            scuff_points.append((radius * math.cos(theta), radius * math.sin(theta), height))
        scuff_faces.append(list(range(j * 4, j * 4 + 4)))
    _mesh(stage, path + "/Scuffs", scuff_points, scuff_faces, grime)


def _venue(stage, lo, hi, mats):
    """Modular venue stays beyond the axis-aligned cone envelope plus 15 m."""
    left, bottom = lo - 15
    right, top = hi + 15
    # Ground stays a single collisionless mesh; shoulders are separate, slightly raised.
    for name, bounds in {
        "West": (-300, -300, min(left, 299), 300),
        "East": (max(right, -299), -300, 300, 300),
        "South": (max(-300, left), -300, min(300, right), min(bottom, 299)),
        "North": (max(-300, left), max(top, -299), min(300, right), 300),
    }.items():
        x0, y0, x1, y1 = bounds
        if x1 > x0 and y1 > y0:
            _rect(stage, f"{_ROOT}/Shoulder{name}", x0, y0, x1, y1, 0.002, mats["gravel"], 2.46)
    # Fence and repeated equipment are placed on a line outside the safe envelope.
    y = top + 3
    if y > 290:
        return {"fence_sections": 0, "workshops": 0, "tents": 0}
    center = (left + right) / 2
    start, end = max(-275, left - 12), min(275, right + 12)
    count = max(1, min(80, int((end - start) / 4)))
    dx = (end - start) / count
    for i in range(count + 1):
        x = start + i * dx
        _pole(stage, f"{_ROOT}/Fence/Post{i}", x, y, 2.1, 0.035, mats["steel"])
    for k, z in enumerate((0.18, 1.05, 2.0)):
        _box(stage, f"{_ROOT}/Fence/Rail{k}", ((start + end) / 2, y, z), (end - start, 0.045, 0.045), mats["steel"])
    # Fine vertical rods read as mesh at a distance without expensive alpha textures.
    for i in range(count * 4 + 1):
        x = start + i * dx / 4
        _pole(stage, f"{_ROOT}/Fence/Wire{i}", x, y, 1.9, 0.007, mats["steel"])
    for i in range(8):
        x = center - 12 + i * 3.1
        _box(
            stage,
            f"{_ROOT}/Barriers/Body{i}",
            (x, y - 1, 0.38),
            (2.8, 0.42, 0.76),
            mats["barrier_red"] if i % 2 else mats["white"],
        )
        _box(stage, f"{_ROOT}/Barriers/Foot{i}", (x, y - 1, 0.07), (2.9, 0.65, 0.14), mats["dark"])
    for i in range(3):
        x = max(-270, min(270, center + (i - 1) * 22))
        _pole(stage, f"{_ROOT}/Lights/Pole{i}", x, y + 5, 7.0, 0.07, mats["steel"])
        _box(stage, f"{_ROOT}/Lights/Head{i}", (x, y + 5, 7), (1.2, 0.35, 0.15), mats["white"])
    # Workshop with articulated facade, corrugated seams, doors, windows and canopy.
    wx, wy = max(-250, min(250, center)), y + 15
    _box(stage, f"{_ROOT}/Workshop/Body", (wx, wy, 2.25), (20, 8, 4.5), mats["wall"])
    _box(stage, f"{_ROOT}/Workshop/Roof", (wx, wy, 4.58), (20.8, 8.7, 0.18), mats["roof"])
    for i in range(3):
        x = wx - 6 + i * 6
        _box(stage, f"{_ROOT}/Workshop/Door{i}", (x, wy - 4.015, 1.65), (4.6, 0.04, 3.2), mats["door"])
        for j in range(6):
            _box(
                stage,
                f"{_ROOT}/Workshop/DoorSlat{i}_{j}",
                (x, wy - 4.045, 0.4 + j * 0.47),
                (4.5, 0.025, 0.025),
                mats["steel"],
            )
        _box(stage, f"{_ROOT}/Workshop/Window{i}", (x, wy - 4.06, 2.9), (2.8, 0.035, 0.35), mats["glass"])
    for i in range(21):
        _box(stage, f"{_ROOT}/Workshop/Clad{i}", (wx - 10 + i, wy - 4.03, 2.3), (0.025, 0.03, 4.35), mats["steel"])
    _box(stage, f"{_ROOT}/Workshop/Canopy", (wx, wy - 5.0, 4), (20.4, 2.1, 0.12), mats["roof"])
    # Paddock tents with pitched canvas roofs and open sides.
    for i in range(2):
        tx, ty = wx + 17 + i * 8, wy - 2
        for a, (dx0, dy0) in enumerate(((-2.5, -2), (-2.5, 2), (2.5, -2), (2.5, 2))):
            _pole(stage, f"{_ROOT}/Tent{i}/Leg{a}", tx + dx0, ty + dy0, 2.4, 0.035, mats["steel"])
        points = [
            (tx - 2.8, ty - 2.3, 2.4),
            (tx + 2.8, ty - 2.3, 2.4),
            (tx + 2.8, ty + 2.3, 2.4),
            (tx - 2.8, ty + 2.3, 2.4),
            (tx, ty, 3.25),
        ]
        _mesh(stage, f"{_ROOT}/Tent{i}/Roof", points, [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], mats["canvas"])
        _box(stage, f"{_ROOT}/Tent{i}/Table", (tx, ty, 0.8), (2, 0.75, 0.08), mats["white"])
        for j in range(3):
            _box(stage, f"{_ROOT}/Tent{i}/Case{j}", (tx - 1 + j * 0.8, ty, 0.95), (0.6, 0.5, 0.25), mats["dark"])
    return {"fence_sections": count, "workshops": 1, "tents": 2}


def apply_realism(stage, cones_xy, cone_colors, *, asset_root: Path, seed: int) -> dict:
    """Apply a deterministic render-only overlay and return its provenance."""
    root = Path(asset_root).resolve()
    manifest, manifest_hash, hashes = _validated_assets(root)
    xy = np.asarray(cones_xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) != len(cone_colors) or not len(xy) or not np.isfinite(xy).all():
        raise ValueError("Expected matching finite cone positions and colors")
    if stage.GetPrimAtPath(_ROOT):
        raise ValueError("Realistic appearance already exists on this stage")
    if not stage.GetPrimAtPath("/World/Ground"):
        raise ValueError("Baseline ground is missing")
    for i in range(len(xy)):
        for part in ("Base", "Cone"):
            if not stage.GetPrimAtPath(f"/World/Cones/c{i:04d}/{part}"):
                raise ValueError(f"Baseline cone {i} is missing {part}")
    UsdGeom.Xform.Define(stage, _ROOT)
    mats = {
        "asphalt": _pbr(stage, "asphalt_track", root, albedo_gain=4.0),
        "patch": _pbr(stage, "asphalt_02", root),
        "concrete": _pbr(stage, "concrete_floor_01", root),
        "gravel": _pbr(stage, "gravel_floor_04", root),
        "steel": _material(stage, "Steel", (0.28, 0.30, 0.31), 0.5, 0.65),
        "wall": _material(stage, "Wall", (0.43, 0.45, 0.43), 0.86),
        "roof": _material(stage, "Roof", (0.19, 0.21, 0.22), 0.75),
        "door": _material(stage, "Door", (0.32, 0.35, 0.35), 0.7),
        "glass": _material(stage, "Glass", (0.075, 0.12, 0.14), 0.2, 0.3),
        "canvas": _material(stage, "Canvas", (0.72, 0.73, 0.69), 0.92),
        "white": _material(stage, "White", (0.75, 0.75, 0.70), 0.72),
        "dark": _material(stage, "Dark", (0.055, 0.06, 0.06), 0.88),
        "tar": _material(stage, "Tar", (0.035, 0.034, 0.032), 0.95),
        "paint": _material(stage, "OldPaint", (0.50, 0.48, 0.40), 0.97),
        "barrier_red": _material(stage, "BarrierRed", (0.55, 0.08, 0.055), 0.78),
    }
    UsdGeom.Imageable(stage.GetPrimAtPath("/World/Ground")).MakeInvisible()
    _rect(stage, f"{_ROOT}/Asphalt", -300, -300, 300, 300, 0.001, mats["asphalt"], 2.0)
    # A few irregular repairs and branching tar seams break up repeated texture.
    rng = np.random.default_rng(seed)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    xmin, ymin = np.maximum(lo - 12, -290)
    xmax, ymax = np.minimum(hi + 12, 290)
    for i in range(14):
        x, y = rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)
        width, height = rng.uniform(1.0, 3.0), rng.uniform(0.7, 2.0)
        points = [
            (x + width * math.cos(a) * rng.uniform(0.78, 1.0), y + height * math.sin(a) * rng.uniform(0.78, 1.0), 0.003)
            for a in np.linspace(0, 2 * math.pi, 9, endpoint=False)
        ]
        _mesh(
            stage,
            f"{_ROOT}/Repairs/Patch{i}",
            points,
            [list(range(len(points)))],
            mats["patch"],
            [(px / 3, py / 3) for px, py, _ in points],
        )
    for i in range(20):
        x, y = rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)
        angle = rng.uniform(-math.pi, math.pi)
        for j in range(4):
            length = rng.uniform(0.45, 1.6)
            angle += rng.uniform(-0.35, 0.35)
            nx, ny = x + length * math.cos(angle), y + length * math.sin(angle)
            width = rng.uniform(0.012, 0.026)
            _box(
                stage,
                f"{_ROOT}/Repairs/Crack{i}_{j}",
                ((x + nx) / 2, (y + ny) / 2, 0.004),
                (length, width, 0.001),
                mats["tar"],
                math.degrees(angle),
            )
            x, y = nx, ny
    # Faded parking bays are confined to an orderly staging row beyond the cones.
    bay_y = min(hi[1] + 11.0, 285.0)
    if bay_y >= hi[1] + 10:
        for i in range(7):
            x = (lo[0] + hi[0]) / 2 + (i - 3) * 2.6
            _box(stage, f"{_ROOT}/Parking/Bay{i}", (x, bay_y, 0.005), (0.055, 4.3, 0.001), mats["paint"])
        _box(
            stage,
            f"{_ROOT}/Parking/End",
            ((lo[0] + hi[0]) / 2, bay_y + 2.15, 0.005),
            (15.6, 0.055, 0.001),
            mats["paint"],
        )
        for i in range(2):
            x = (lo[0] + hi[0]) / 2 + (i - 0.5) * 12
            _box(stage, f"{_ROOT}/Drains/Frame{i}", (x, bay_y - 4, 0.006), (1.0, 0.36, 0.01), mats["dark"])
            for j in range(8):
                _box(
                    stage,
                    f"{_ROOT}/Drains/Slat{i}_{j}",
                    (x - 0.42 + j * 0.12, bay_y - 4, 0.012),
                    (0.035, 0.34, 0.005),
                    mats["steel"],
                )
    cone_mats = {
        ConeColor.BLUE: _material(stage, "ConeBlue", (0.018, 0.085, 0.65), 0.62),
        ConeColor.YELLOW: _material(stage, "ConeYellow", (0.82, 0.55, 0.018), 0.64),
        ConeColor.ORANGE: _material(stage, "ConeOrange", (0.90, 0.14, 0.018), 0.62),
        ConeColor.BIG_ORANGE: _material(stage, "ConeBigOrange", (0.90, 0.14, 0.018), 0.62),
        ConeColor.UNKNOWN: _material(stage, "ConeUnknown", (0.32, 0.32, 0.32), 0.8),
    }
    stripe = _material(stage, "ConeStripe", (0.8, 0.8, 0.75), 0.5)
    grime = _material(stage, "ConeScuff", (0.21, 0.20, 0.17), 0.95)
    for i, color in enumerate(cone_colors):
        parent = f"/World/Cones/c{i:04d}"
        UsdGeom.Imageable(stage.GetPrimAtPath(parent + "/Base")).MakeInvisible()
        UsdGeom.Imageable(stage.GetPrimAtPath(parent + "/Cone")).MakeInvisible()
        spec = RULES.large_cone if color == ConeColor.BIG_ORANGE else RULES.small_cone
        _cone(stage, parent + "/Appearance", spec, cone_mats[color], stripe, cone_mats[color], grime, i)
    venue_counts = _venue(stage, lo, hi, mats)
    car_record = dress_car(stage)
    # Keep the existing lights' prims but use only the photographed HDR environment.
    sky = UsdLux.DomeLight.Get(stage, "/World/Sky")
    sun = UsdLux.DistantLight.Get(stage, "/World/Sun")
    if not sky or not sun:
        raise ValueError("Baseline lighting is missing")
    sky.CreateTextureFileAttr().Set(Sdf.AssetPath(str(root / "derelict_airfield_01/hdri.hdr")))
    sky.CreateTextureFormatAttr().Set("latlong")
    sky.CreateColorAttr().Set(Gf.Vec3f(1, 1, 1))
    sky.CreateIntensityAttr().Set(500.0)
    sun.CreateIntensityAttr().Set(0.0)
    return {
        "appearance": "realistic_v1",
        "seed": int(seed),
        "manifest_sha256": manifest_hash,
        "asset_sha256": hashes,
        "assets": {name: manifest["assets"][name] for name in _ASSETS},
        "settings": {
            "asphalt": "asphalt_track",
            "asphalt_tile_m": 2.0,
            "asphalt_albedo_gain": 4.0,
            "hdri": "derelict_airfield_01",
            "dome_intensity": 500.0,
            "distant_sun_intensity": 0.0,
            "repairs": 14,
            "branching_seams": 20,
            "parking_bays": 6 if bay_y >= hi[1] + 10 else 0,
        },
        "preview_selection": {
            "v0": "rejected: asphalt too dark",
            "v1": "rejected: repeated texture cracks formed a grid",
            "v2": "retained: asphalt_track with 4x linear albedo gain",
        },
        "counts": {"cones": len(xy), **venue_counts},
        "car": car_record,
    }
