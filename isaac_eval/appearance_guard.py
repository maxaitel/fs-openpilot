"""Compare the physical USD contract before and after dressing a scene."""

import hashlib
import json


def physical_contract(stage):
    from pxr import UsdGeom, UsdPhysics

    values = {}
    geom_attrs = {"points", "faceVertexCounts", "faceVertexIndices", "size", "radius", "height", "axis"}
    for prim in stage.Traverse():
        attrs = {}
        schemas = [s for s in prim.GetAppliedSchemas() if s.startswith(("Physics", "Physx"))]
        if schemas:
            attrs["__physics_schemas"] = sorted(schemas)
        is_collider = prim.HasAPI(UsdPhysics.CollisionAPI)
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if name.startswith(("physics:", "physx")) or (is_collider and name in geom_attrs):
                attrs[name] = str(attr.Get())
        rels = {
            r.GetName(): [str(v) for v in r.GetTargets()]
            for r in prim.GetRelationships()
            if r.GetName().startswith(("physics:", "physx", "collection:colliders"))
            or r.GetName() == "material:binding:physics"
        }
        if attrs or rels:
            transforms = {}
            parent = prim
            while parent and parent.GetPath().pathString != "/":
                xf = UsdGeom.Xformable(parent)
                if xf:
                    transforms[str(parent.GetPath())] = {
                        "matrix": str(xf.GetLocalTransformation()),
                        "reset_stack": xf.GetResetXformStack(),
                    }
                parent = parent.GetParent()
            values[str(prim.GetPath())] = {"attrs": attrs, "relationships": rels, "transforms": transforms}
    return values


def assert_appearance_only(before, stage):
    after = physical_contract(stage)
    differences = [p for p in sorted(before.keys() | after.keys()) if before.get(p) != after.get(p)]
    assert not differences, f"Appearance changed physics: {differences}"
    return hashlib.sha256(json.dumps(after, sort_keys=True).encode()).hexdigest()


def evidence_sources():
    """Bind preview/integration evidence to the exact executable scene adapter."""
    from pathlib import Path

    names = [
        "scene.py",
        "rendering.py",
        "vehicle.py",
        "realistic_scene.py",
        "realistic_car.py",
        "appearance_guard.py",
        "export_scene.py",
        "preview_realism.py",
        "verify_integration.py",
        "evaluate.py",
    ]
    root = Path(__file__).resolve().parent
    paths = [root / name for name in names] + [
        root / "assets/manifest.json",
        root.parent / "src/fsop/geometry/camera.py",
    ]
    return {str(path.relative_to(root.parent)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
