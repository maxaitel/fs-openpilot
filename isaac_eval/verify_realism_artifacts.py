"""Audit completed episode bundles with OpenUSD and ffprobe."""

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

from pxr import Sdf, Usd, UsdGeom, UsdUtils


def audit_episode(directory):
    summary = json.loads((directory / "summary.json").read_text())
    frames = json.loads((directory / "frames.json").read_text())
    appearance = json.loads((directory / "appearance.json").read_text())
    contacts = summary["contact_events"]
    assert len({e["cone"] for e in contacts}) == len(contacts) == summary["physical_cone_hits"]
    assert math.isclose(len(frames) * 0.05, summary["seconds_simulated"], abs_tol=1e-8)
    stage_path = directory / "scene.usda"
    stage = Usd.Stage.Open(str(stage_path))
    assert stage and stage.GetPrimAtPath("/World/Car/Chassis")
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert UsdGeom.GetStageUpAxis(stage) == "Z"
    _, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(stage_path)))
    assert not unresolved, unresolved
    hashes = {}
    for asset in assets:
        path = Path(asset).resolve()
        assert path.is_relative_to(directory / "assets"), path
        hashes[str(path.relative_to(directory / "assets"))] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert hashes == appearance.get("asset_sha256", {}), f"Asset mismatch: {directory.name}"
    video = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,nb_frames,r_frame_rate",
                "-of",
                "json",
                str(directory / "video.mp4"),
            ]
        )
    )["streams"][0]
    assert video["codec_name"] == "h264" and (video["width"], video["height"]) == (1280, 640)
    assert video["r_frame_rate"] == "10/1"
    assert int(video["nb_frames"]) == (len(frames) + 1) // 2 + 30
    return {
        "case": directory.name,
        "pass": True,
        "asset_count": len(hashes),
        "asset_sha256": hashes,
        "recorded_steps": len(frames),
        "unique_contacts": len(contacts),
        "video": video,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("isaac_eval/runs/realism_v1"))
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    audits = [
        audit_episode(root / f"{variant}_{case['discipline']}_{case['seed']}")
        for case in protocol["cases"]
        for variant in protocol["variants"]
    ]
    result = {"pass": True, "cases": audits}
    (root / "artifact_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Verified {len(audits)} scene bundles, contact counts and full H264 recordings")


if __name__ == "__main__":
    main()
