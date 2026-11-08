"""Freeze all 41 matched cases before starting any Isaac evaluation."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
from pathlib import Path

from isaac_eval.parity_cases import (
    all_cases,
    build_environment,
    json_sha256,
    realized_environment,
    suite_manifest,
    verify_checkpoint,
    verify_environment_sources,
    verify_panorama_assets,
)


def write_frozen_json(path: Path, value: dict) -> None:
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"Refusing to replace a different frozen parity artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def prepare(output: Path, checkpoint: Path, panorama_dir: Path) -> dict:
    manifest = suite_manifest()
    verify_checkpoint(checkpoint)
    manifest["current_environment_source_hashes"] = verify_environment_sources()
    manifest["panorama_asset_hashes"] = verify_panorama_assets(panorama_dir)
    manifest["runtime"] = {"python": platform.python_version(), "platform": platform.platform(),
                           "packages": {name: importlib.import_module(name).__version__
                                        for name in ("numpy", "scipy", "cv2", "shapely")}}
    realizations = []
    for case in all_cases():
        env = build_environment(case)
        realization = realized_environment(env)
        path = Path("realizations") / f"{case.case_id}.json"
        write_frozen_json(output / path, realization)
        realizations.append({"case_id": case.case_id, "path": str(path),
                             "realization_sha256": realization["realization_sha256"],
                             "track_sha256": realization["track_sha256"]})
    manifest["realizations"] = realizations
    manifest["manifest_sha256"] = json_sha256(manifest)
    verify_checkpoint(checkpoint)
    write_frozen_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--panorama-dir", default=Path("artifacts/panoramas"), type=Path)
    args = parser.parse_args()
    manifest = prepare(args.output, args.checkpoint, args.panorama_dir)
    print(json.dumps({"manifest": str(args.output / "manifest.json"), "cases": manifest["case_count"],
                      "manifest_sha256": manifest["manifest_sha256"]}))


if __name__ == "__main__":
    main()
