import os
import shutil
from pathlib import Path

from pxr import Sdf, Usd, UsdUtils


def export_scene(stage, target, asset_root):
    target, asset_root = Path(target).resolve(), Path(asset_root).resolve()
    stage.GetRootLayer().Export(str(target))
    layer = Sdf.Layer.FindOrOpen(str(target))
    dependencies = []

    def copy_asset(value):
        if not value:
            return value
        source = Path(value)
        if not source.is_absolute():
            source = Path.cwd() / source
        source = source.resolve()
        if not source.is_relative_to(asset_root):
            raise ValueError(f"Unexpected external scene asset: {value}")
        relative = source.relative_to(asset_root)
        dest = target.parent / "assets" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            try:
                os.link(source, dest)
            except OSError:
                shutil.copy2(source, dest)
        dependencies.append(str(relative))
        return "assets/" + str(relative)

    UsdUtils.ModifyAssetPaths(layer, copy_asset)
    layer.Save()
    reopened = Usd.Stage.Open(str(target))
    assert reopened and reopened.GetPrimAtPath("/World/Car/Chassis")
    _, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(target)))
    assert not unresolved, f"Unresolved assets: {unresolved}"
    assert all(Path(a).resolve().is_relative_to(target.parent) for a in assets), "Export refers outside its bundle"
    return {"resolved_assets": len(assets), "dependencies": sorted(set(dependencies))}
