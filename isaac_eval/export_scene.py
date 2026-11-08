import shutil
from pathlib import Path

from pxr import Ar, Sdf, Usd, UsdUtils


def export_scene(stage, target, asset_root):
    target, asset_root = Path(target).resolve(), Path(asset_root).resolve()
    layer = stage.Flatten()
    dependencies = set()

    def copy_asset(value):
        if not value:
            return value
        resolved = str(Ar.GetResolver().Resolve(value))
        if not resolved or '://' in value:
            raise ValueError(f'Unresolved or remote scene asset: {value}')
        source = Path(resolved).resolve()
        if not source.is_file() or not source.is_relative_to(asset_root):
            raise ValueError(f'Unexpected external scene asset: {value}')
        relative = source.relative_to(asset_root)
        dest = target.parent / 'assets' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.read_bytes() != source.read_bytes():
            raise ValueError(f'Conflicting scene asset: {relative}')
        if not dest.exists():
            shutil.copy2(source, dest)
        dependencies.add(str(relative))
        return 'assets/' + relative.as_posix()

    UsdUtils.ModifyAssetPaths(layer, copy_asset)
    layer.Export(str(target))
    reopened = Usd.Stage.Open(str(target))
    if not reopened or not reopened.GetPrimAtPath('/World/Car/Chassis'):
        raise ValueError('Exported scene failed to reopen')
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(target)))
    if unresolved:
        raise ValueError(f'Unresolved assets: {unresolved}')
    for item in [*(layer.realPath for layer in layers), *assets]:
        if not item or not Path(str(item)).resolve().is_relative_to(target.parent):
            raise ValueError(f'Export refers outside its bundle: {item}')
    return {'resolved_assets': len(assets), 'dependencies': sorted(dependencies)}
