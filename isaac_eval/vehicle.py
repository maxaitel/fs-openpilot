"""FS-sized generic car using NVIDIA's PhysX Vehicle tire/suspension solver.

USD setup follows Isaac Sim 5.1's omni.physx.vehicle BasicSetup/WheelController
samples. Coordinates are adapted to x-forward, y-left, z-up.
"""

import numpy as np
from isaacsim.core.prims import SingleRigidPrim
from omni.physx.scripts.physicsUtils import add_collision_to_collision_group
from omni.physx.scripts.utils import set_custom_metadata
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

WHEEL_RADIUS = 0.23
WHEELBASE = 1.55
TRACK_WIDTH = 1.20


def create_car(stage, world=None):
    from isaacsim.core.utils.extensions import enable_extension
    from scene import bind, body, cube, material

    enable_extension("omni.physx.vehicle")
    scene = next(p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene))
    context = PhysxSchema.PhysxVehicleContextAPI.Apply(scene)
    context.CreateUpdateModeAttr(PhysxSchema.Tokens.velocityChange)
    context.CreateVerticalAxisAttr(PhysxSchema.Tokens.posZ)
    context.CreateLongitudinalAxisAttr(PhysxSchema.Tokens.posX)
    groups = {name: f"/World/Groups/{name}" for name in ["Chassis", "Wheel", "Query", "Ground"]}
    filters = {
        "Chassis": ["Query"],
        "Wheel": ["Query", "Ground"],
        "Query": ["Chassis", "Wheel"],
        "Ground": ["Ground", "Wheel"],
    }
    for name, path in groups.items():
        group = UsdPhysics.CollisionGroup.Define(stage, path)
        group.CreateFilteredGroupsRel().SetTargets([groups[key] for key in filters[name]])
    add_collision_to_collision_group(stage, "/World/Ground", groups["Ground"])
    table = PhysxSchema.PhysxVehicleTireFrictionTable.Define(stage, "/World/TireFriction")
    table.CreateGroundMaterialsRel().SetTargets(["/World/Materials/Asphalt"])
    table.CreateFrictionValuesAttr([1.4])
    table.CreateDefaultFrictionValueAttr(1.4)
    red = material(stage, "/World/Materials/Car", (0.55, 0.025, 0.015), 0.4)
    black = material(stage, "/World/Materials/Tire", (0.025, 0.025, 0.025))
    UsdGeom.Xform.Define(stage, "/World/Car")
    path = "/World/Car/Chassis"
    chassis = body(stage, path, (0.0, 0.0, 0.36), 230.0)
    UsdPhysics.MassAPI(chassis).CreateDiagonalInertiaAttr(Gf.Vec3f(40.0, 140.0, 110.0))
    PhysxSchema.PhysxRigidBodyAPI(chassis).CreateDisableGravityAttr(True)
    vehicle = PhysxSchema.PhysxVehicleAPI.Apply(chassis)
    vehicle.CreateVehicleEnabledAttr(True)
    vehicle.CreateSubStepThresholdLongitudinalSpeedAttr(5.0)
    vehicle.CreateLowForwardSpeedSubStepCountAttr(3)
    vehicle.CreateHighForwardSpeedSubStepCountAttr(1)
    vehicle.CreateMinPassiveLongitudinalSlipDenominatorAttr(4.0)
    vehicle.CreateMinActiveLongitudinalSlipDenominatorAttr(0.1)
    vehicle.CreateMinLateralSlipDenominatorAttr(1.0)
    set_custom_metadata(chassis, PhysxSchema.Tokens.referenceFrameIsCenterOfMass, False)
    cube(stage, path + "/Body", (0.15, 0.0, 0.0), (2.9, 0.96, 0.25), red)
    add_collision_to_collision_group(stage, path + "/Body", groups["Chassis"])
    cube(stage, path + "/Cockpit", (-0.1, 0.0, 0.20), (0.8, 0.55, 0.3), black, False)
    controllers = []
    for i, (label, x, y) in enumerate([("fl", 0.8, 0.6), ("fr", 0.8, -0.6), ("rl", -0.75, 0.6), ("rr", -0.75, -0.6)]):
        wheel_path = path + "/Wheel_" + label
        wheel = UsdGeom.Xform.Define(stage, wheel_path)
        wheel.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.23 - 0.36))
        attachment = PhysxSchema.PhysxVehicleWheelAttachmentAPI.Apply(wheel.GetPrim())
        attachment.CreateCollisionGroupRel().SetTargets([groups["Query"]])
        attachment.CreateSuspensionTravelDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        sprung_mass = 230.0 * (0.75 if i < 2 else 0.8) / 1.55 / 2
        attachment.CreateSuspensionFramePositionAttr(Gf.Vec3f(x, y, 0.23 - 0.36 + 0.08 - sprung_mass * 9.81 / 20000.0))
        attachment.CreateIndexAttr(i)
        w = PhysxSchema.PhysxVehicleWheelAPI.Apply(wheel.GetPrim())
        w.CreateRadiusAttr(0.23)
        w.CreateWidthAttr(0.20)
        w.CreateMassAttr(5.0)
        w.CreateMoiAttr(0.132)
        w.CreateDampingRateAttr(0.025)
        tire = PhysxSchema.PhysxVehicleTireAPI.Apply(wheel.GetPrim())
        tire.CreateLateralStiffnessGraphAttr(Gf.Vec2f(2.0, 16000.0 if i < 2 else 18000.0))
        tire.CreateLongitudinalStiffnessAttr(10000.0)
        tire.CreateCamberStiffnessAttr(0.0)
        tire.CreateFrictionVsSlipGraphAttr([Gf.Vec2f(0.0, 1.0), Gf.Vec2f(0.1, 1.0), Gf.Vec2f(1.0, 1.0)])
        tire.CreateFrictionTableRel().SetTargets(["/World/TireFriction"])
        suspension = PhysxSchema.PhysxVehicleSuspensionAPI.Apply(wheel.GetPrim())
        suspension.CreateSpringStrengthAttr(20000.0)
        suspension.CreateSpringDamperRateAttr(2000.0)
        suspension.CreateTravelDistanceAttr(0.08)
        mesh = UsdGeom.Cylinder.Define(stage, wheel_path + "/Tire")
        mesh.CreateRadiusAttr(0.23)
        mesh.CreateHeightAttr(0.20)
        mesh.CreateAxisAttr("Y")
        mesh.CreateExtentAttr(UsdGeom.Cylinder.ComputeExtentFromPlugins(mesh, 0))
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        add_collision_to_collision_group(stage, str(mesh.GetPath()), groups["Wheel"])
        bind(mesh.GetPrim(), black)
        ctrl = PhysxSchema.PhysxVehicleWheelControllerAPI.Apply(wheel.GetPrim())
        ctrl.CreateDriveTorqueAttr(0.0)
        ctrl.CreateBrakeTorqueAttr(0.0)
        ctrl.CreateSteerAngleAttr(0.0)
        controllers.append(ctrl)
    robot = SingleRigidPrim(path, name="car")
    if world is not None:
        robot = world.scene.add(robot)
    robot.wheel_controllers = controllers
    return robot


class AckermannActuators:
    def __init__(self, robot):
        self.robot = robot
        self.angle = 0.0
        self.accel = 0.0
        for wheel in robot.wheel_controllers:
            wheel.GetDriveTorqueAttr().Set(0.0)
            wheel.GetBrakeTorqueAttr().Set(0.0)
            wheel.GetSteerAngleAttr().Set(0.0)

    def apply(self, steer, accel, dt=0.05):
        target = float(np.clip(steer, -np.deg2rad(24), np.deg2rad(24)))
        self.angle += np.clip((target - self.angle) * min(dt / 0.06, 1), -np.deg2rad(150) * dt, np.deg2rad(150) * dt)
        self.accel += (float(np.clip(accel, -13, 7)) - self.accel) * min(dt / 0.12, 1.0)
        curvature = np.tan(self.angle) / WHEELBASE
        angles = [
            np.arctan2(WHEELBASE * curvature, 1 - curvature * TRACK_WIDTH / 2),
            np.arctan2(WHEELBASE * curvature, 1 + curvature * TRACK_WIDTH / 2),
            0.0,
            0.0,
        ]
        torque = 230.0 * self.accel * WHEEL_RADIUS / 4
        for wheel, angle in zip(self.robot.wheel_controllers, angles, strict=True):
            wheel.GetDriveTorqueAttr().Set(max(torque, 0.0))
            wheel.GetBrakeTorqueAttr().Set(max(-torque, 0.0))
            wheel.GetSteerAngleAttr().Set(float(angle))
