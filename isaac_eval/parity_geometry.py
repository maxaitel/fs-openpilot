import numpy as np
from scipy.spatial.transform import Rotation

from fsop.geometry.camera import se3

USD_FROM_CAMERA_BODY = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
BODY_REFERENCE_HEIGHT = 0.36


def camera_transforms(position, rotation, nominal_mount, planar_ego):
    body_mount = np.array(nominal_mount, copy=True)
    body_mount[2, 3] -= BODY_REFERENCE_HEIGHT
    world_camera = se3(rotation, position) @ body_mount
    return world_camera, np.linalg.inv(planar_ego) @ world_camera


def pose_model_cameras(cameras, position, rotation, nominal_mount, planar_ego):
    world_camera, projection = camera_transforms(position, rotation, nominal_mount, planar_ego)
    q = Rotation.from_matrix(world_camera[:3, :3] @ USD_FROM_CAMERA_BODY).as_quat()
    for cam in cameras.values():
        cam.set_world_pose(world_camera[:3, 3], np.r_[q[3], q[:3]], camera_axes="usd")
    return world_camera, projection
