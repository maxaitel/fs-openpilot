import hashlib
import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdLux
from realistic_scene import apply_realism
from scipy.spatial.transform import Rotation


def dress_matched_scene(stage, env, asset_root):
    receipt = apply_realism(stage, env.cones_xy, env.cone_colors, asset_root=asset_root, seed=env.seed)
    scene = env.scene
    sky = UsdLux.DomeLight.Get(stage, "/World/Sky")
    sun = UsdLux.DistantLight.Get(stage, "/World/Sun")
    panorama = Path(scene.sky_pano) if scene.sky_pano else None
    if panorama is not None:
        if not panorama.is_file():
            raise FileNotFoundError(panorama)
        sky.CreateTextureFileAttr().Set(Sdf.AssetPath(str(panorama.resolve())))
        sky.CreateColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    else:
        sky.CreateTextureFileAttr().Set(Sdf.AssetPath(""))
        sky.CreateColorAttr().Set(Gf.Vec3f(*scene.sky_color))
    sky.CreateIntensityAttr().Set(500.0 * scene.ambient / 0.45)
    xf = UsdGeom.Xformable(sky)
    xf.ClearXformOpOrder()
    xf.AddRotateZOp().Set(math.degrees(scene.sky_yaw))
    direction = -np.array(
        [
            math.cos(scene.sun_azimuth) * math.cos(scene.sun_elevation),
            math.sin(scene.sun_azimuth) * math.cos(scene.sun_elevation),
            math.sin(scene.sun_elevation),
        ]
    )
    rotation, _ = Rotation.align_vectors(direction[None], np.array([[0.0, 0.0, -1.0]]))
    q = rotation.as_quat()
    xf = UsdGeom.Xformable(sun)
    xf.ClearXformOpOrder()
    xf.AddOrientOp().Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3])))
    sun.CreateIntensityAttr().Set(2000.0 * scene.sun_intensity)
    sun.CreateColorAttr().Set(Gf.Vec3f(*scene.sun_color))
    receipt["matching"] = {
        "sampled_scene": scene.to_dict(),
        "panorama_sha256": hashlib.sha256(panorama.read_bytes()).hexdigest() if panorama else None,
        "sun_direction": direction.tolist(),
        "sun_intensity": 2000.0 * scene.sun_intensity,
        "ambient_intensity": 500.0 * scene.ambient / 0.45,
        "exact_shared": [
            "panorama asset",
            "sky yaw",
            "sun direction and color",
            "camera mount",
            "sampled cone layout",
            "image postprocess function and parameters",
        ],
        "backend_differences": [
            "Panda light units mapped to fixed USD intensity scales; radiometry is not identical",
            "Scanned asphalt, detailed cones and paddock props replace Panda procedural geometry/materials",
            "Panda fog, grass-ring, paint gaps, cone fade/lettering/brightness "
            "and ground texture parameters are not reproduced",
            "PhysX moves contacted cones; Panda uses its geometric knockdown animation",
        ],
    }
    return receipt
