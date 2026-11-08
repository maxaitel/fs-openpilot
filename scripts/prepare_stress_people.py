#!/usr/bin/env python3
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdSkel, UsdUtils, Vt

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "isaac_eval/stress_people.lock.json"
CACHE = ROOT / ".work/stress/assets-probe/source"
OUTPUT = ROOT / "isaac_eval/assets/stress_people"
POSES = {
    "F_Business_02": (80, -79, 10, -8),
    "male_adult_construction_03": (79, -80, 8, -10),
    "female_adult_police_02": (80, -78, 10, -8),
}

TINTS = {
    "F_Business_02": {
        "opaque__fabric__bluejeans": (0.25, 0.43, 0.8),
        "opaque__fabric__shirt": (0.20, 0.32, 0.55),
        "opaque__fabric__vest": (0.40, 0.46, 0.56),
    },
    "male_adult_construction_03": {
        "opaque__fabric__jeans": (0.26, 0.44, 0.76),
        "opaque__fabric__shirt": (0.30, 0.40, 0.53),
        "opaque__fabric__safetyvest": (2.0, 0.85, 0.05),
        "opaque__plastic__hardhat": (2.0, 1.55, 0.06),
    },
    "female_adult_police_02": {
        "opaque__fabric__policepants": (0.16, 0.25, 0.48),
        "opaque__fabric__policeshirt": (0.68, 0.80, 1.0),
        "opaque__fabric__safetyvest": (1.85, 1.65, 0.08),
    },
}
VISIBLE_PARTS = {
    "male_adult_construction_03": {"opaque__plastic__hardhat", "retro__reflective__safetyvest"},
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(lock, name, relative, spec):
    path = CACHE / name / relative
    if not path.is_file() or path.stat().st_size != spec["size"] or digest(path) != spec["sha256"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".download")
        urllib.request.urlretrieve(lock["source_root"] + name + "/" + relative, temporary)
        if temporary.stat().st_size != spec["size"] or digest(temporary) != spec["sha256"]:
            temporary.unlink()
            raise ValueError(f"Unexpected download: {relative}")
        temporary.replace(path)
    return path


def pose_skeleton(skeleton, angles):
    joints = skeleton.GetJointsAttr().Get()
    topology = UsdSkel.Topology(joints)
    rest = list(skeleton.GetRestTransformsAttr().Get())
    rest_world = UsdSkel.ConcatJointTransforms(topology, rest)
    for suffix, angle, axis in zip(
        ("L_Upperarm", "R_Upperarm", "L_Forearm", "R_Forearm"), angles, ((0, 1, 0), (0, 1, 0), (0, 0, 1), (0, 0, 1)),
        strict=True,
    ):
        indexes = [i for i, joint in enumerate(joints) if str(joint).split("/")[-1] == suffix]
        if len(indexes) != 1:
            raise ValueError(f"Expected one {suffix} joint, got {len(indexes)}")
        i = indexes[0]
        local_axis = rest_world[i].GetInverse().TransformDir(Gf.Vec3d(*axis)).GetNormalized()
        rotation = Gf.Matrix4d(1).SetRotate(Gf.Rotation(local_axis, angle))
        rest[i] = rotation * rest[i]
    posed = UsdSkel.ConcatJointTransforms(topology, rest)
    binds = skeleton.GetBindTransformsAttr().Get()
    matrices = np.array([np.array(b.GetInverse() * p) for b, p in zip(binds, posed, strict=True)])
    positions = {
        str(j).split("/")[-1]: list(posed[i].ExtractTranslation())
        for i, j in enumerate(joints)
        if str(j).split("/")[-1] in {"L_Upperarm", "R_Upperarm", "L_Forearm", "R_Forearm", "L_Hand", "R_Hand"}
    }
    return matrices, positions


def preview_material(stage, base, name, tint):
    material = UsdShade.Material.Define(stage, "/Person/Looks/" + name)
    shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild("Preview"))
    shader.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    uv = UsdShade.Shader.Define(stage, material.GetPath().AppendChild("UV"))
    uv.CreateIdAttr("UsdPrimvarReader_float2")
    uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st0")
    uv.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    for suffix, input_name, channel in [
        ("BaseColor", "diffuseColor", "rgb"),
        ("Normal", "normal", "rgb"),
        ("ORM", "roughness", "g"),
    ]:
        tex = UsdShade.Shader.Define(stage, material.GetPath().AppendChild(suffix))
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("./textures/" + base + "_" + suffix + ".png")
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB" if suffix == "BaseColor" else "raw")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), "result")
        if suffix == "BaseColor" and tint is not None:
            tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*tint, 1))
        if suffix == "Normal":
            tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
            tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
        kind = Sdf.ValueTypeNames.Float3 if channel == "rgb" else Sdf.ValueTypeNames.Float
        tex.CreateOutput(channel, kind)
        shader.CreateInput(
            input_name,
            Sdf.ValueTypeNames.Normal3f
            if suffix == "Normal"
            else (Sdf.ValueTypeNames.Color3f if suffix == "BaseColor" else Sdf.ValueTypeNames.Float),
        ).ConnectToSource(tex.ConnectableAPI(), channel)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0)
    return material


def bake(name, files):
    source = Usd.Stage.Open(str(CACHE / name / (name + ".usd")))
    skeletons = [UsdSkel.Skeleton(p) for p in source.Traverse() if p.IsA(UsdSkel.Skeleton)]
    if len(skeletons) != 1:
        raise ValueError(f"{name}: expected one skeleton")
    matrices, joints = pose_skeleton(skeletons[0], POSES[name])
    destination = OUTPUT / name
    destination.mkdir(parents=True, exist_ok=True)
    outpath = destination / "person.usd"
    if outpath.exists():
        outpath.unlink()
    stage = Usd.Stage.Open(Sdf.Layer.CreateNew(str(outpath), {"format": "usda"}))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Person")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(stage, "/Person/Looks")
    UsdGeom.Scope.Define(stage, "/Person/Geometry")
    base = next(Path(f).name.removesuffix("_BaseColor.png") for f in files if f.endswith("_BaseColor.png"))
    for relative in files:
        if relative.endswith(".png"):
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(CACHE / name / relative, target)
    all_points = []
    mesh_count = 0
    for prim in source.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        if not Gf.IsClose(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), Gf.Matrix4d(1), 1e-6
        ):
            raise ValueError(f"{prim.GetPath()}: unexpected mesh transform")
        binding = UsdSkel.BindingAPI(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        normals = np.asarray(mesh.GetNormalsAttr().Get(), dtype=np.float64)
        if mesh.GetNormalsInterpolation() != "vertex" or len(normals) != len(points):
            raise ValueError(f"{prim.GetPath()}: expected vertex normals")
        if binding.GetJointsAttr().Get() is not None:
            raise ValueError(f"{prim.GetPath()}: unexpected per-mesh joint order")
        indices_pv = binding.GetJointIndicesPrimvar()
        indices = np.asarray(indices_pv.ComputeFlattened(), dtype=np.int64).reshape(len(points), -1)
        weights = np.asarray(binding.GetJointWeightsPrimvar().ComputeFlattened(), dtype=np.float64).reshape(
            indices.shape
        )
        if not np.allclose(weights.sum(axis=1), 1, atol=1e-4):
            raise ValueError(f"{prim.GetPath()}: joint weights do not sum to one")
        geom_bind = binding.GetGeomBindTransformAttr().Get() or Gf.Matrix4d(1)
        to_skel = np.array(geom_bind)
        homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1) @ to_skel
        skin = matrices[indices]
        moved = np.einsum("ni,nkij,nk->nj", homogeneous, skin, weights)[:, :3]
        normal_matrix = np.linalg.inv(to_skel[:3, :3]).T
        moved_normals = np.einsum("ni,nkij,nk->nj", normals @ normal_matrix, skin[:, :, :3, :3], weights)
        moved_normals /= np.maximum(np.linalg.norm(moved_normals, axis=1, keepdims=True), 1e-12)
        target_path = Sdf.Path("/Person/Geometry/" + prim.GetName())
        Sdf.CopySpec(source.GetRootLayer(), prim.GetPath(), stage.GetRootLayer(), target_path)
        target = stage.GetPrimAtPath(target_path)
        baked = UsdGeom.Mesh(target)
        baked.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(moved.astype(np.float32)))
        baked.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(moved_normals.astype(np.float32)))
        baked.GetExtentAttr().Set([Gf.Vec3f(*moved.min(axis=0)), Gf.Vec3f(*moved.max(axis=0))])
        for prop in list(target.GetProperties()):
            if (
                "skel:" in prop.GetName()
                or prop.GetName().startswith("xformOp")
                or prop.GetName() == "material:binding"
            ):
                target.RemoveProperty(prop.GetName())
        if prim.GetName() in VISIBLE_PARTS.get(name, set()):
            UsdGeom.Imageable(target).MakeVisible()
        material = preview_material(stage, base, prim.GetName(), TINTS[name].get(prim.GetName()))
        UsdShade.MaterialBindingAPI.Apply(target).Bind(material)
        all_points.append(moved)
        mesh_count += 1
    if not mesh_count:
        raise ValueError(f"{name}: no meshes")
    points = np.concatenate(all_points)
    stage.GetRootLayer().Save()
    _, dependencies, unresolved = UsdUtils.ComputeAllDependencies(str(outpath))
    if unresolved or len(dependencies) != 3:
        raise ValueError(f"{name}: unexpected dependencies: {dependencies}, {unresolved}")
    bounds = [points.min(axis=0).tolist(), points.max(axis=0).tolist()]
    return {
        "usd": f"{name}/person.usd",
        "sha256": digest(outpath),
        "meshes": mesh_count,
        "bounds_m": bounds,
        "joint_positions_m": joints,
        "pose_angles_degrees": list(POSES[name]),
        "base_color_scales": TINTS[name],
        "enabled_parts": sorted(VISIBLE_PARTS.get(name, set())),
        "textures": {rel: digest(destination / rel) for rel in files if rel.endswith(".png")},
    }


def main():
    lock = json.loads(LOCK.read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    assets = {}
    for name, data in lock["assets"].items():
        for relative, spec in data["files"].items():
            download(lock, name, relative, spec)
        assets[name] = bake(name, data["files"])
        assets[name]["source_files"] = data["files"]
        print(name, assets[name]["bounds_m"], flush=True)
    manifest = {
        "version": 1,
        "source_root": lock["source_root"],
        "license_url": lock["license_url"],
        "default_prim": "/Person",
        "meters_per_unit": 1,
        "up_axis": "Z",
        "assets": assets,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
