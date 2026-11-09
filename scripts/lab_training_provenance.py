
from __future__ import annotations

import hashlib
import json
from pathlib import Path

MEMBERSHIP = Path(__file__).resolve().parents[1] / "docs/isaac-lab-training/membership.json"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dataset_paths(run, dataset=None, verification=None):
    dataset = Path(dataset if dataset is not None else run["args"]["manifest"]).resolve()
    verification = Path(verification if verification is not None else run["dataset_verification"]["path"]).resolve()
    if not dataset.is_file() or not verification.is_file():
        raise ValueError("Recorded dataset is unavailable; supply explicit dataset and verification paths for this run")
    return dataset, verification


def validation_identity(run, dataset, verification, membership=MEMBERSHIP):
    """Bind recorded paths to verified cases by unique metadata hashes, never basenames."""
    from fsop.data.native_replay import NativeEpisode, episode_paths
    from scripts.verify_lab_dataset import verify

    dataset, verification, membership = Path(dataset), Path(verification), Path(membership)
    manifest = json.loads(dataset.read_text())
    receipt = json.loads(verification.read_text())
    recorded = run["dataset_verification"]
    manifest_sha, receipt_sha = digest(dataset), digest(verification)
    if (
        manifest.get("schema_version") != 1
        or run.get("manifest_sha256") != manifest_sha
        or recorded.get("status") != "verified"
        or recorded.get("sha256") != receipt_sha
        or recorded.get("manifest_sha256") != manifest_sha
        or recorded.get("membership_sha256") != digest(membership)
    ):
        raise ValueError("Run dataset verification provenance mismatch")
    actual = verify(dataset, membership)
    for key in ("passed", "manifest_sha256", "membership_sha256", "episodes", "anchors"):
        if receipt.get(key) != actual[key]:
            raise ValueError(f"Dataset verification receipt differs from actual dataset: {key}")
    by_hash, validation, artifacts = {}, {}, {}
    for path, row in zip(episode_paths(manifest=dataset), manifest["episodes"], strict=True):
        meta = json.loads((path / "meta.json").read_text())
        meta_sha = row["meta_sha256"]
        if meta_sha in by_hash:
            raise ValueError("Duplicate episode metadata identity")
        by_hash[meta_sha] = (meta, row)
        artifacts[str(path / "meta.json")] = meta_sha
        for name, value in meta["arrays_sha256"].items():
            artifacts[str(path / name)] = value
        artifacts[str(path / "track.json")] = digest(path / "track.json")
        artifacts[str(path / "source.zip")] = meta["source_archive_sha256"]
        if meta["split"] == "val":
            ep = NativeEpisode(path)
            validation[meta["id"]] = {
                "case": meta["case"],
                "meta_sha256": meta_sha,
                "arrays_sha256": meta["arrays_sha256"],
                "qualified_frames": set(ep.anchors),
            }
    paths, seen = {}, set()
    for episode in run["episodes"]:
        match = by_hash.get(episode["meta_sha256"])
        if match is None:
            raise ValueError("Recorded training episode differs from verified dataset")
        meta, row = match
        identity = meta["id"]
        if episode["path"] in paths or identity in seen:
            raise ValueError("Duplicate recorded training episode")
        if episode["split"] != meta["split"] or episode["accepted"] != row["accepted"]:
            raise ValueError("Recorded training episode qualification differs")
        paths[episode["path"]] = identity
        seen.add(identity)
    if len(seen) != len(by_hash):
        raise ValueError("Recorded training inventory is incomplete")
    selected = set()
    for anchor in run["validation_selection"]:
        identity = paths.get(anchor["path"])
        if identity not in validation or identity in selected:
            raise ValueError("Validation selection has an unknown or duplicate case")
        frames = anchor["frames"]
        if (
            not isinstance(frames, list)
            or not frames
            or any(type(frame) is not int for frame in frames)
            or len(set(frames)) != len(frames)
            or not set(frames).issubset(validation[identity]["qualified_frames"])
        ):
            raise ValueError("Validation selection contains invalid or unqualified frames")
        validation[identity]["frames"] = frames
        selected.add(identity)
    if selected != set(validation):
        raise ValueError("Validation selection does not cover every verified validation case")
    for row in validation.values():
        del row["qualified_frames"]
    artifacts.update(
        {
            str(dataset.resolve()): manifest_sha,
            str(verification.resolve()): receipt_sha,
            str(membership.resolve()): digest(membership),
        }
    )
    return {
        "fingerprint": validation,
        "path_to_case": paths,
        "input_sha256": artifacts,
        "membership_sha256": actual["membership_sha256"],
    }


def run_checkpoint(root, run, recorded_path, filename):
    """Relocate only a declared run's direct checkpoint child when mirroring /work."""
    root, recorded = Path(root).resolve(), Path(recorded_path)
    allowed = {root / filename}
    if run["args"].get("out"):
        allowed.add(Path(run["args"]["out"]) / filename)
    if recorded not in allowed and recorded.resolve() not in allowed:
        raise ValueError("checkpoint path mismatch")
    return root / filename
