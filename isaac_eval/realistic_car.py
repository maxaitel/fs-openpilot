"""Original, render-only Formula Student bodywork for the generic PhysX car.

All added geometry is a child of the existing chassis or wheel attachment.
The existing colliders are only hidden; no physical property is authored here.
Dimensions describe an illustrative FS car, not a particular team's vehicle.
"""

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade

_CHASSIS = "/World/Car/Chassis"
_LOOK = _CHASSIS + "/Visual"
_WHEELS = (("fl", 0.8, 0.6), ("fr", 0.8, -0.6), ("rl", -0.75, 0.6), ("rr", -0.75, -0.6))


def _material(stage, name, color, roughness, metallic=0.0, clearcoat=0.0):
    mat = UsdShade.Material.Define(stage, _LOOK + "/Materials/" + name)
    shader = UsdShade.Shader.Define(stage, str(mat.GetPath()) + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    for key, value in (("roughness", roughness), ("metallic", metallic), ("clearcoat", clearcoat)):
        shader.CreateInput(key, Sdf.ValueTypeNames.Float).Set(float(value))
    shader.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(0.24)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def _mesh(stage, path, vertices, faces, mat, smooth=False):
    points = np.asarray(vertices, dtype=float)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in points])
    mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
    mesh.CreateFaceVertexIndicesAttr([int(i) for face in faces for i in face])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr([Gf.Vec3f(*points.min(axis=0)), Gf.Vec3f(*points.max(axis=0))])
    if smooth:
        normals = np.zeros_like(points)
        for face in faces:
            normal = np.cross(points[face[1]] - points[face[0]], points[face[2]] - points[face[0]])
            for index in face:
                normals[index] += normal
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        mesh.CreateNormalsAttr([Gf.Vec3f(*v) for v in normals])
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    _bind(mesh.GetPrim(), mat)
    return mesh


def _box(stage, path, center, size, mat, tilt=0.0):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*center))
    if tilt:
        cube.AddRotateYOp().Set(float(tilt))
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    _bind(cube.GetPrim(), mat)


def _rod(stage, path, start, end, radius, mat):
    a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    delta = b - a
    tube = UsdGeom.Cylinder.Define(stage, path)
    tube.CreateAxisAttr(UsdGeom.Tokens.z)
    tube.CreateRadiusAttr(float(radius))
    tube.CreateHeightAttr(float(np.linalg.norm(delta)))
    tube.AddTranslateOp().Set(Gf.Vec3d(*((a + b) / 2)))
    q = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*delta)).GetQuat()
    tube.AddOrientOp().Set(Gf.Quatf(q.GetReal(), Gf.Vec3f(q.GetImaginary())))
    _bind(tube.GetPrim(), mat)


def _loft(stage, path, sections, mat, lateral=0.0):
    vertices = []
    for x, width, bottom, top in sections:
        bevel = min((top - bottom) * 0.22, width * 0.28)
        ring = [
            (-width * 0.80, bottom),
            (-width, bottom + bevel),
            (-width, top - bevel),
            (-width * 0.78, top),
            (width * 0.78, top),
            (width, top - bevel),
            (width, bottom + bevel),
            (width * 0.80, bottom),
        ]
        vertices.extend((x, lateral + y, z) for y, z in ring)
    faces = [list(range(8))]
    for section in range(len(sections) - 1):
        a, b = section * 8, (section + 1) * 8
        faces.extend([[a + j, b + j, b + (j + 1) % 8, a + (j + 1) % 8] for j in range(8)])
    faces.append(list(reversed(range((len(sections) - 1) * 8, len(sections) * 8))))
    _mesh(stage, path, vertices, faces, mat, smooth=True)


def _sweep(stage, path, points, radius, mat, sides=8):
    centers = np.asarray(points, dtype=float)
    vertices = []
    for index, point in enumerate(centers):
        tangent = centers[min(index + 1, len(centers) - 1)] - centers[max(0, index - 1)]
        tangent /= np.linalg.norm(tangent)
        reference = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tangent, reference)) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        u = np.cross(tangent, reference)
        u /= np.linalg.norm(u)
        v = np.cross(tangent, u)
        for angle in np.linspace(0, 2 * np.pi, sides, endpoint=False):
            vertices.append(point + radius * (np.cos(angle) * u + np.sin(angle) * v))
    faces = []
    for ring in range(len(centers) - 1):
        a, b = ring * sides, (ring + 1) * sides
        faces.extend([[a + j, a + (j + 1) % sides, b + (j + 1) % sides, b + j] for j in range(sides)])
    faces.extend([list(reversed(range(sides))), list(range((len(centers) - 1) * sides, len(centers) * sides))])
    _mesh(stage, path, vertices, faces, mat, smooth=True)


def _lathe(stage, path, profile, mat, segments=64):
    vertices = [
        (r * np.cos(t), y, r * np.sin(t))
        for y, r in profile
        for t in np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ]
    faces = []
    for ring in range(len(profile)):
        a, b = ring * segments, ((ring + 1) % len(profile)) * segments
        faces.extend([[a + j, b + j, b + (j + 1) % segments, a + (j + 1) % segments] for j in range(segments)])
    _mesh(stage, path, vertices, faces, mat, smooth=True)


def _wheel(stage, root, rubber, alloy, steel, dark):
    _lathe(
        stage,
        root + "/Slick",
        [
            (-0.095, 0.135),
            (-0.100, 0.165),
            (-0.093, 0.202),
            (-0.075, 0.224),
            (-0.047, 0.230),
            (0.047, 0.230),
            (0.075, 0.224),
            (0.093, 0.202),
            (0.100, 0.165),
            (0.095, 0.135),
            (0.074, 0.130),
            (-0.074, 0.130),
        ],
        rubber,
    )
    _lathe(
        stage,
        root + "/RimBarrel",
        [
            (-0.091, 0.126),
            (-0.091, 0.135),
            (0.091, 0.135),
            (0.091, 0.126),
            (0.076, 0.121),
            (-0.076, 0.121),
        ],
        alloy,
    )
    _rod(stage, root + "/Hub", (0, -0.092, 0), (0, 0.092, 0), 0.040, dark)
    for side in (-1, 1):
        name = "Inside" if side < 0 else "Outside"
        y = side * 0.078
        _lathe(
            stage,
            root + "/Brake" + name,
            [(y - 0.003, 0.105), (y + 0.003, 0.105), (y + 0.003, 0.059), (y - 0.003, 0.059)],
            steel,
        )
        _rod(stage, root + "/AxleCap" + name, (0, side * 0.089, 0), (0, side * 0.099, 0), 0.028, alloy)
        for index, theta in enumerate(np.linspace(0, 2 * np.pi, 6, endpoint=False)):
            vertices = []
            for depth in (-0.006, 0.006):
                for radial, angle, width in ((0.034, theta, 0.012), (0.122, theta + 0.12, 0.008)):
                    for direction in (-1, 1):
                        vertices.append(
                            (
                                radial * np.cos(angle) - direction * width * np.sin(angle),
                                side * (0.085 + depth),
                                radial * np.sin(angle) + direction * width * np.cos(angle),
                            )
                        )
            faces = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 4, 6, 2], [1, 3, 7, 5], [0, 1, 5, 4], [2, 6, 7, 3]]
            _mesh(stage, root + f"/Spoke{name}{index}", vertices, faces, alloy)
            x, z = 0.048 * np.cos(theta), 0.048 * np.sin(theta)
            _rod(stage, root + f"/Bolt{name}{index}", (x, side * 0.091, z), (x, side * 0.098, z), 0.005, steel)


def dress_car(stage) -> dict:
    """Add original visual bodywork; hide only existing visible collider shapes.

    Call after create_car() and before world.reset(). No physics schemas,
    collision geometry, or existing transforms are changed. Wheel meshes are
    under the existing wheel attachments and inherit their simulated rotation.
    """
    required = [_CHASSIS + "/Body", _CHASSIS + "/Cockpit"]
    required += [_CHASSIS + "/Wheel_" + label + "/Tire" for label, _, _ in _WHEELS]
    missing = [path for path in required if not stage.GetPrimAtPath(path).IsValid()]
    if missing:
        raise ValueError("FS appearance requires the original generic car: " + ", ".join(missing))
    receipt = {
        "asset": "original_procedural_fs_car_v1",
        "description": "Generic FS visual bodywork; original PhysX vehicle retained",
        "hidden_prims": required,
        "visual_roots": [_LOOK] + [_CHASSIS + "/Wheel_" + name + "/Visual" for name, _, _ in _WHEELS],
        "external_assets": [],
    }
    if stage.GetPrimAtPath(_LOOK).IsValid():
        if stage.GetPrimAtPath(_LOOK).GetCustomDataByKey("fsAppearance") != receipt["asset"]:
            raise ValueError("Car visual namespace already belongs to another asset")
        return receipt
    root = UsdGeom.Xform.Define(stage, _LOOK)
    carbon = _material(stage, "Carbon", (0.025, 0.031, 0.035), 0.31, clearcoat=0.32)
    red = _material(stage, "RedPaint", (0.55, 0.015, 0.008), 0.25, clearcoat=0.45)
    dark = _material(stage, "BlackAnodized", (0.022, 0.026, 0.029), 0.32, metallic=0.7)
    rubber = _material(stage, "SlickRubber", (0.021, 0.023, 0.025), 0.75)
    fabric = _material(stage, "SeatFabric", (0.018, 0.020, 0.024), 0.95)
    alloy = _material(stage, "MachinedAlloy", (0.57, 0.60, 0.63), 0.25, metallic=1.0)
    steel = _material(stage, "Steel", (0.23, 0.25, 0.27), 0.42, metallic=1.0)
    white = _material(stage, "IvoryPaint", (0.76, 0.77, 0.74), 0.36)

    _loft(
        stage,
        _LOOK + "/Floor",
        [
            (-1.22, 0.29, -0.24, -0.18),
            (-0.7, 0.38, -0.24, -0.17),
            (0.45, 0.32, -0.24, -0.17),
            (1.34, 0.11, -0.15, -0.11),
        ],
        carbon,
    )
    for side, sign in (("Left", 1), ("Right", -1)):
        _loft(
            stage,
            _LOOK + "/Sill" + side,
            [(-0.65, 0.064, -0.17, 0.28), (-0.2, 0.063, -0.18, 0.21), (0.32, 0.055, -0.15, 0.20)],
            carbon,
            sign * 0.28,
        )
        _loft(
            stage,
            _LOOK + "/Sidepod" + side,
            [
                (-0.94, 0.065, -0.17, 0.05),
                (-0.62, 0.115, -0.20, 0.11),
                (0.09, 0.105, -0.18, 0.12),
                (0.28, 0.070, -0.16, 0.07),
            ],
            red,
            sign * 0.39,
        )
        _box(stage, _LOOK + "/RadiatorOpening" + side, (0.285, sign * 0.39, -0.035), (0.005, 0.105, 0.115), dark)
        for slat in range(5):
            _rod(
                stage,
                _LOOK + f"/RadiatorVane{side}{slat}",
                (0.290, sign * 0.39 - 0.046, -0.079 + slat * 0.021),
                (0.290, sign * 0.39 + 0.046, -0.079 + slat * 0.021),
                0.0025,
                steel,
            )
        _rod(stage, _LOOK + "/CockpitRim" + side, (-0.58, sign * 0.27, 0.27), (0.28, sign * 0.27, 0.21), 0.011, dark)

    _loft(
        stage,
        _LOOK + "/Nose",
        [
            (0.23, 0.27, -0.16, 0.20),
            (0.54, 0.23, -0.13, 0.17),
            (0.91, 0.17, -0.11, 0.095),
            (1.34, 0.10, -0.13, -0.015),
            (1.57, 0.065, -0.13, -0.065),
        ],
        red,
    )
    _loft(
        stage,
        _LOOK + "/NoseCenterStripe",
        [
            (0.28, 0.052, 0.201, 0.205),
            (0.54, 0.045, 0.171, 0.175),
            (0.91, 0.035, 0.096, 0.100),
            (1.34, 0.025, -0.014, -0.010),
            (1.53, 0.016, -0.059, -0.055),
        ],
        carbon,
    )
    _loft(
        stage,
        _LOOK + "/RearCowl",
        [(-1.25, 0.13, -0.18, 0.11), (-0.97, 0.27, -0.17, 0.22), (-0.61, 0.29, -0.17, 0.25)],
        carbon,
    )
    _box(stage, _LOOK + "/SeatBase", (-0.21, 0, -0.080), (0.48, 0.36, 0.085), fabric)
    _box(stage, _LOOK + "/SeatBack", (-0.48, 0, 0.14), (0.09, 0.35, 0.48), fabric, tilt=-12)
    _box(stage, _LOOK + "/Headrest", (-0.53, 0, 0.38), (0.075, 0.22, 0.16), fabric)
    for sign in (-1, 1):
        _box(
            stage,
            _LOOK + ("/HarnessLeft" if sign > 0 else "/HarnessRight"),
            (-0.426, sign * 0.105, 0.155),
            (0.013, 0.040, 0.34),
            red,
            tilt=-12,
        )
    _rod(stage, _LOOK + "/SteeringColumn", (0.18, 0, -0.02), (0.08, 0, 0.24), 0.014, steel)
    steering = [(0.075, 0.14 * np.cos(t), 0.25 + 0.09 * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 33)]
    _sweep(stage, _LOOK + "/SteeringWheel", steering, 0.012, rubber)
    _box(stage, _LOOK + "/SteeringHub", (0.075, 0, 0.25), (0.035, 0.11, 0.055), carbon)
    _box(stage, _LOOK + "/DashDisplay", (0.285, 0, 0.222), (0.020, 0.12, 0.065), dark, tilt=-18)

    hoop = [
        (-0.54, -0.25, 0.10),
        (-0.54, -0.25, 0.49),
        (-0.54, -0.23, 0.57),
        (-0.54, -0.17, 0.625),
        (-0.54, 0, 0.65),
        (-0.54, 0.17, 0.625),
        (-0.54, 0.23, 0.57),
        (-0.54, 0.25, 0.49),
        (-0.54, 0.25, 0.10),
    ]
    _sweep(stage, _LOOK + "/RollHoop", hoop, 0.021, steel, sides=12)
    for side, sign in (("Left", 1), ("Right", -1)):
        _rod(stage, _LOOK + "/RollBrace" + side, (-0.54, sign * 0.20, 0.55), (-1.13, sign * 0.22, 0.04), 0.015, steel)
        _loft(
            stage,
            _LOOK + "/Accent" + side,
            [(-0.80, 0.012, 0.088, 0.094), (-0.62, 0.016, 0.111, 0.117), (0.05, 0.012, 0.121, 0.127)],
            white,
            sign * 0.39,
        )

    for label, x, y in _WHEELS:
        sign = 1 if y > 0 else -1
        for height, z in (("Upper", -0.055), ("Lower", -0.205)):
            tip = (x, sign * 0.475, z)
            for branch, offset in (("Fore", 0.18), ("Aft", -0.18)):
                _rod(
                    stage,
                    _LOOK + f"/Wishbone_{label}_{height}_{branch}",
                    (x + offset, sign * 0.27, z + 0.025),
                    tip,
                    0.010,
                    carbon,
                )
        start = np.array((x - 0.13, sign * 0.25, 0.07))
        end = np.array((x, sign * 0.46, -0.17))
        _rod(stage, _LOOK + "/Damper_" + label, start, end, 0.016, alloy)
        axis = end - start
        axis /= np.linalg.norm(axis)
        u = np.cross(axis, np.array([1.0, 0.0, 0.0]))
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        coil = [
            start + t * (end - start) + 0.027 * (np.cos(t * 12 * np.pi) * u + np.sin(t * 12 * np.pi) * v)
            for t in np.linspace(0.15, 0.85, 78)
        ]
        _sweep(stage, _LOOK + "/Spring_" + label, coil, 0.0045, red, sides=6)
        wheel_root = _CHASSIS + "/Wheel_" + label + "/Visual"
        UsdGeom.Xform.Define(stage, wheel_root)
        _wheel(stage, wheel_root, rubber, alloy, steel, dark)

    for path in required:
        UsdGeom.Imageable(stage.GetPrimAtPath(path)).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    root.GetPrim().SetCustomDataByKey("fsAppearance", receipt["asset"])
    return receipt
