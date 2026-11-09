"""Read the sealed conditions needed to compare two native Lab episodes."""

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

FIELDS = (
    "schema_version",
    "backend",
    "fps",
    "label_dt",
    "command_mode",
    "variant",
    "case",
    "realization_sha256",
    "track",
    "start_pose",
    "initial_physical_state",
    "initial_cones_xyz",
    "cone_colors",
    "vehicle_params",
    "expert_params",
    "nominal_detection_calibration",
    "intrinsics",
    "physics_sha256",
    "appearance_sha256",
    "asset_sha256",
    "layout_sha256",
    "render_profile",
    "sim_render_settings",
    "physics_settings",
    "source_sha256",
    "graph_sha256",
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def comparison_conditions(folder, metadata):
    missing = [key for key in FIELDS if key not in metadata]
    if missing:
        raise ValueError(f"Missing comparison fields: {', '.join(missing)}")
    for key in FIELDS:
        if key not in ("asset_sha256", "layout_sha256") and metadata[key] is None:
            raise ValueError(f"Missing comparison field: {key}")
    folder = Path(folder)
    for field, filename in (
        ("realization_sha256", "realization.json"),
        ("appearance_sha256", "appearance.json"),
        ("source_archive_sha256", "source.zip"),
    ):
        if field not in metadata or metadata[field] != digest(folder / filename):
            raise ValueError(f"Episode artifact is not bound to metadata: {filename}")
    with ZipFile(folder / "source.zip") as archive:
        sources = metadata["source_sha256"]
        if not isinstance(sources, dict) or sorted(archive.namelist()) != sorted(sources):
            raise ValueError("Source inventory mismatch")
        for name, expected in sources.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError("Source content mismatch")
    export_path = folder / "export.json"
    if metadata.get("export_sha256") != digest(export_path):
        raise ValueError("Scene export is not bound to metadata")
    export = json.loads(export_path.read_text())
    dependencies, hashes = export["dependencies"], export["sha256"]
    if not isinstance(dependencies, list) or not isinstance(hashes, dict) or sorted(dependencies) != sorted(hashes):
        raise ValueError("Scene asset inventory mismatch")
    for name, expected in hashes.items():
        asset = (folder / "assets" / name).resolve()
        if not asset.is_relative_to((folder / "assets").resolve()) or digest(asset) != expected:
            raise ValueError("Scene asset digest mismatch or path escape")
    return {**{key: metadata[key] for key in FIELDS}, "asset_dependencies": hashes}


def compare_conditions(left, right):
    for key in (*FIELDS, "asset_dependencies"):
        if key not in left or key not in right or left[key] != right[key]:
            raise ValueError(f"Noncomparable episode condition: {key}")
