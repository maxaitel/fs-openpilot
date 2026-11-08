import numpy as np
from isaacsim.core.utils.numpy.rotations import rot_matrices_to_quats
from isaacsim.sensors.camera import Camera
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade
from vehicle import AckermannActuators as AckermannActuators
from vehicle import create_car as create_car

from fsop.geometry.camera import CAMERA_MODELS
from fsop.track.rules import RULES
from fsop.track.schema import ConeColor

WHEEL_RADIUS = 0.23
WHEELBASE = 1.55
TRACK_WIDTH = 1.20


def material(stage, path, color, roughness=0.8):
    m = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    m.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return m


def bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def cube(stage, path, position, size, mat, collision=True):
    obj = UsdGeom.Cube.Define(stage, path)
    obj.CreateSizeAttr(1.0)
    obj.AddTranslateOp().Set(Gf.Vec3d(*position))
    obj.AddScaleOp().Set(Gf.Vec3f(*size))
    if collision:
        UsdPhysics.CollisionAPI.Apply(obj.GetPrim())
    bind(obj.GetPrim(), mat)
    return obj.GetPrim()


def body(stage, path, position, mass):
    obj = UsdGeom.Xform.Define(stage, path)
    obj.AddTranslateOp().Set(Gf.Vec3d(*position))
    p = obj.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(p)
    mass_api = UsdPhysics.MassAPI.Apply(p)
    mass_api.CreateMassAttr(mass)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateEnableCCDAttr(False)
    PhysxSchema.PhysxRigidBodyAPI(p).CreateMaxDepenetrationVelocityAttr(1.0)
    return p


def create_ground(stage, seed=0):
    asphalt = material(stage, "/World/Materials/Asphalt", (0.095, 0.105, 0.115))
    physics = UsdPhysics.MaterialAPI.Apply(asphalt.GetPrim())
    physics.CreateStaticFrictionAttr(1.4)
    physics.CreateDynamicFrictionAttr(1.25)
    physics.CreateRestitutionAttr(0.0)
    ground = cube(stage, "/World/Ground", (0.0, 0.0, -0.1), (600.0, 600.0, 0.2), asphalt)
    UsdShade.MaterialBindingAPI(ground).Bind(asphalt, materialPurpose="physics")
    rng = np.random.default_rng(seed)
    sky = UsdLux.DomeLight.Define(stage, "/World/Sky")
    sky.CreateIntensityAttr(float(rng.uniform(400, 650)))
    sky.CreateColorAttr(Gf.Vec3f(0.78, 0.86, 1.0))
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(float(rng.uniform(1800, 2400)))
    sun.CreateAngleAttr(1.0)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-55, 20, float(rng.uniform(-60, 60))))


def create_cones(stage, xy, colors):
    palette = {
        ConeColor.BLUE: (0.015, 0.06, 0.8),
        ConeColor.YELLOW: (1.0, 0.7, 0.01),
        ConeColor.ORANGE: (1.0, 0.15, 0.015),
        ConeColor.BIG_ORANGE: (1.0, 0.15, 0.015),
        ConeColor.UNKNOWN: (0.5, 0.5, 0.5),
    }
    mats = {key: material(stage, "/World/Materials/Cone_" + key.value, col) for key, col in palette.items()}
    white = material(stage, "/World/Materials/Stripe", (0.9, 0.9, 0.9))
    for i, (pos, color) in enumerate(zip(xy, colors, strict=True)):
        spec = RULES.large_cone if color == ConeColor.BIG_ORANGE else RULES.small_cone
        path = f"/World/Cones/c{i:04d}"
        p = body(stage, path, (float(pos[0]), float(pos[1]), 0.02), 1.5 if color == ConeColor.BIG_ORANGE else 0.8)
        PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0.0)
        cube(stage, path + "/Base", (0.0, 0.0, 0.0), (spec.base, spec.base, 0.04), mats[color])
        mesh = UsdGeom.Mesh.Define(stage, path + "/Cone")
        n = 24
        levels = [
            (0.02, spec.radius_bottom),
            (0.02 + spec.height * 0.48, spec.radius_bottom * 0.55),
            (0.02 + spec.height * 0.62, spec.radius_bottom * 0.43),
            (0.02 + spec.height, spec.radius_top),
        ]
        points = [
            (r * np.cos(j * 2 * np.pi / n), r * np.sin(j * 2 * np.pi / n), z) for z, r in levels for j in range(n)
        ]
        faces = []
        for level in range(3):
            faces.extend(
                [
                    [level * n + j, level * n + (j + 1) % n, (level + 1) * n + (j + 1) % n, (level + 1) * n + j]
                    for j in range(n)
                ]
            )
        mesh.CreatePointsAttr([Gf.Vec3f(*q) for q in points])
        mesh.CreateFaceVertexCountsAttr([4] * len(faces))
        mesh.CreateFaceVertexIndicesAttr([j for f in faces for j in f])
        mesh.CreateSubdivisionSchemeAttr("none")
        bind(mesh.GetPrim(), mats[color])
        subset = UsdGeom.Subset.Define(stage, path + "/Cone/Stripe")
        subset.CreateElementTypeAttr("face")
        subset.CreateIndicesAttr(list(range(n, 2 * n)))
        subset.CreateFamilyNameAttr("materialBind")
        bind(subset.GetPrim(), white)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")


def make_cameras(with_chase=True):
    cameras = {}
    for name, model in CAMERA_MODELS.items():
        cam = Camera("/World/Camera_" + name, resolution=(512, 256), frequency=-1)
        cam.set_focal_length(0.012)
        cam.set_horizontal_aperture(0.012 * 512 / model.K[0, 0])
        cam.set_vertical_aperture(0.012 * 256 / model.K[1, 1])
        cam.set_opencv_pinhole_properties(
            cx=float(model.K[0, 2]),
            cy=float(model.K[1, 2]),
            fx=float(model.K[0, 0]),
            fy=float(model.K[1, 1]),
            pinhole=[0.0] * 12,
        )
        cam.set_clipping_range(0.05, 500.0)
        cam.initialize()
        cameras[name] = cam
    if not with_chase:
        return cameras, None
    chase = Camera("/World/Chase", resolution=(768, 512), frequency=-1)
    chase.set_focal_length(0.018)
    chase.set_horizontal_aperture(0.036)
    chase.set_vertical_aperture(0.024)
    chase.initialize()
    return cameras, chase


def pose_cameras(cameras, chase, position, rotation):
    basis = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    eye = position + rotation @ np.array([0.0, 0.0, 1.1 - 0.36])
    q = rot_matrices_to_quats(rotation @ basis)
    for cam in cameras.values():
        cam.set_world_pose(eye, q, camera_axes="usd")
    eye2 = position + rotation @ np.array([-5.0, -4.0, 3.0])
    f = position + np.array([0.0, 0.0, 0.3]) - eye2
    f /= np.linalg.norm(f)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    chase.set_world_pose(eye2, rot_matrices_to_quats(np.column_stack([right, up, -f])), camera_axes="usd")
