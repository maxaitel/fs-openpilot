"""Verify a complete native dataset against its frozen membership and producer sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from fsop.data.native_replay import load_episodes
from fsop.track.fixed_layouts import skidpad
from fsop.track.generate import TrackGenCfg, generate_track
from isaac_lab_fsop.cases import Case, environment


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_value(value):
    return json.loads(json.dumps(value))


def verify(manifest_path, membership_path):
    manifest_path, membership_path = Path(manifest_path), Path(membership_path)
    manifest = json.loads(manifest_path.read_text())
    membership = json.loads(membership_path.read_text())
    membership_sha = digest(membership_path)
    cases = {c["id"]: c for c in membership["cases"] if c["split"] in ("train", "val")}
    if (manifest.get("complete") is not True or manifest["expected_episodes"] != len(cases)
            or manifest["membership_sha256"] != membership_sha or len(manifest["episodes"]) != len(cases)):
        raise ValueError("Collection is incomplete or belongs to another membership")
    paths, seen, receipts = [], set(), []
    for row in manifest["episodes"]:
        path = (manifest_path.parent / row["path"]).resolve()
        if not path.is_relative_to(manifest_path.parent.resolve()):
            raise ValueError("Episode escapes dataset root")
        if digest(path / "meta.json") != row["meta_sha256"]:
            raise ValueError("Episode metadata digest differs from manifest")
        m = json.loads((path / "meta.json").read_text())
        identity = m["id"]
        if identity in seen or m["case"] != cases.get(identity) or m["membership_sha256"] != membership_sha:
            raise ValueError("Duplicate or mismatched frozen case")
        seen.add(identity)
        case = Case(**m["case"])
        if (m["split"], m["discipline"], m["seed"]) != (case.split, case.discipline, case.seed):
            raise ValueError("Episode split metadata contradicts frozen case")
        track = skidpad() if case.discipline == "skidpad" else generate_track(case.track_seed, TrackGenCfg())
        if case.discipline != "skidpad":
            track.laps = 1
        saved_track = json.loads((path / "track.json").read_text())
        if json_value(track.to_dict()) != saved_track or m["track_id"] != track.id:
            raise ValueError("Track does not reproduce from frozen seed")
        env = environment(case)
        if (json_value(env.track.to_dict()) != saved_track or json_value(env.scene.to_dict()) != m["scene"]
                or json_value(asdict(env.vp)) != m["vehicle"]):
            raise ValueError("Runtime realization differs from the recorded episode")
        np.testing.assert_allclose(env.T_ego_cam(), m["calibration"]["T_ego_cam"], atol=1e-12, rtol=0)
        if digest(path / "source.zip") != m["source_archive_sha256"]:
            raise ValueError("Producer archive digest differs")
        with zipfile.ZipFile(path / "source.zip") as archive:
            if sorted(archive.namelist()) != sorted(m["source_sha256"]):
                raise ValueError("Producer archive inventory differs")
            for name, expected in m["source_sha256"].items():
                if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                    raise ValueError("Archived producer source digest differs")
        paths.append(path)
        receipts.append({"id": identity, "track_sha256": digest(path / "track.json"),
                         "meta_sha256": row["meta_sha256"], "source_sha256": m["source_archive_sha256"]})
    if seen != set(cases):
        raise ValueError("Membership cases are missing")
    # This additionally verifies every array digest, timestamps, native teacher
    # qualification, actual-future masks, and train/development track isolation.
    episodes = load_episodes(paths)
    for ep, row in zip(episodes, manifest["episodes"], strict=True):
        if (len(ep.anchors) != row["accepted"] or ep.rejections != row["rejections"]
                or ep.meta["summary"] != row["summary"]):
            raise ValueError("Manifest qualification differs from independently read episode")
    return {"passed": True, "manifest_sha256": digest(manifest_path), "membership_sha256": membership_sha,
            "episodes": receipts,
            "anchors": {s: sum(len(ep.anchors) for ep in episodes if ep.meta["split"] == s)
                        for s in ("train", "val")},
            "limits": ["This verifies recorded data and reproducibility of metadata, not a new physics replay.",
                       "Source archives are checked internally; pilot and bulk producer sources may differ."]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--membership", type=Path, default=Path("docs/isaac-lab-training/membership.json"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    report = verify(a.manifest, a.membership)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}))


if __name__ == "__main__":
    main()
