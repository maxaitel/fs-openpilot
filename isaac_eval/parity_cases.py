"""Frozen custom-simulator cases and their renderer-independent initialization."""

from __future__ import annotations

import hashlib
import json
import tempfile
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from fsop.sim.env import CAMERAS, FSEnv, SimCfg
from fsop.sim.randomization import dr_preset
from fsop.track.fixed_layouts import skidpad
from fsop.track.io import load_json

REFERENCE_ROOT = Path(__file__).with_name("parity_reference")
CHECKPOINT_SHA256 = "cf3a8d253dfd5a417e1967d2d6dc8cdd0263e3d16d3f1f02f15495f490a83b7e"
REFERENCE_SHA256 = "3aba439995027dc70d96ae64c902aa53ae3c4702d94d2052d716681c1785809c"
RETENTION_TRACKS = (
    "trackdrive_020000", "trackdrive_020005", "trackdrive_020015", "trackdrive_020023",
    "trackdrive_030000", "trackdrive_030010", "trackdrive_030020", "trackdrive_030061",
)
LEVELS = ("light", "heavy")
RESULT_SOURCES = {
    "skidpad": "runs/skidpad_clean/fresh_after/driving.json",
    "retention": "runs/skidpad_clean/control_retention/driving.json",
    "zero": "runs/skidpad_clean/commands_zero/driving.json",
    "swapped": "runs/skidpad_clean/commands_swapped/driving.json",
    "prior_failure": "runs/skidpad_clean/video_previous_failure/video.json",
}
ENVIRONMENT_SOURCES = (
    "src/fsop/sim/env.py", "src/fsop/sim/expert.py", "src/fsop/sim/randomization.py",
    "src/fsop/sim/vehicle.py", "src/fsop/sim/referee.py", "src/fsop/sim/route_commands.py",
    "src/fsop/sim/skidpad.py", "src/fsop/profile.py", "src/fsop/track/fixed_layouts.py",
    "src/fsop/track/schema.py", "src/fsop/track/centerline.py", "src/fsop/track/io.py",
    "src/fsop/geometry/camera.py",
)
MATCHING_CONTRACT = {
    "checkpoint_sha256": CHECKPOINT_SHA256,
    "controller": "pp",
    "max_time": 120.0,
    "noise": False,
    "venue_prob": 1.0,
    "fps": 20,
    "substeps": 5,
    "randomize_expert": True,
    "random_start": False,
    "start_speed": 0.0,
    "profile": "built-in TeamProfile defaults",
    "cameras": ["road", "wide"],
    "retention_laps": 1,
    "postprocessing": "original postprocess with env.rng, road then wide, once per policy observation",
    "initialization": "FSEnv reset renders and postprocesses once before the first policy observation",
    "panorama_sampling": "frozen sorted inventory of the 20 original artifacts/panoramas/*.jpg files",
    "commands": "RouteCommands.advance on physical progress; zero or swap channels 1 and 2 at policy boundary",
    "referee": "original Referee using actual randomized cone positions and original mission_done threshold",
}
BACKEND_DIFFERENCES = {
    "renderer": "Isaac RTX replaces Panda3D; RGB, shadows, materials and lighting are not pixel-identical.",
    "venue": "SceneDR values are sampled identically; backend-specific scenery meshes and material responses differ.",
    "physics": "PhysX replaces the custom bicycle model; dynamics, contact response and body motion differ.",
    "camera": "Sampled mount and model intrinsics match; Isaac body heave/roll/pitch may change world extrinsics.",
    "trajectory": "Commands and referee logic match, but different trajectories can change their timing and outcomes.",
    "numeric_runtime": "NumPy, SciPy and platform versions are recorded; cross-platform floating point may differ.",
    "historical_state": ("Realizations are reconstructed from source, seed and archived asset inventory; "
                         "historical runs did not save initialization snapshots."),
}


@dataclass(frozen=True)
class ParityCase:
    case_id: str
    group: str
    discipline: str
    seed: int
    dr: str
    command_mode: str
    track_id: str
    reference_source: str


def all_cases() -> tuple[ParityCase, ...]:
    cases = [ParityCase(f"skidpad-{seed}-{dr}-recorded", "skidpad", "skidpad", seed, dr,
                        "recorded", "skidpad", RESULT_SOURCES["skidpad"])
             for seed in range(72000, 72010) for dr in LEVELS]
    cases += [ParityCase(f"retention-{tid}-{dr}-recorded", "retention", "trackdrive", 0, dr,
                         "recorded", tid, RESULT_SOURCES["retention"])
              for tid in RETENTION_TRACKS for dr in LEVELS]
    cases += [ParityCase(f"ablation-skidpad-72000-{dr}-{mode}", "ablation", "skidpad", 72000, dr,
                         mode, "skidpad", RESULT_SOURCES[mode]) for mode in ("zero", "swapped") for dr in LEVELS]
    cases.append(ParityCase("prior-failure-skidpad-71003-heavy-recorded", "prior_failure", "skidpad", 71003,
                            "heavy", "recorded", "skidpad", RESULT_SOURCES["prior_failure"]))
    return tuple(cases)


def get_case(case_id: str) -> ParityCase:
    for case in all_cases():
        if case.case_id == case_id:
            return case
    raise ValueError(f"Unknown frozen parity case {case_id!r}")


def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_sha256(value: object) -> str:
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(content).hexdigest()


def reference_provenance(reference_root: Path = REFERENCE_ROOT) -> dict:
    path = reference_root / "provenance.json"
    if sha256_file(path) != REFERENCE_SHA256:
        raise ValueError("Frozen custom-simulator provenance manifest changed")
    return json.loads(path.read_text())


def _reference_document(source: str, provenance: dict, reference_root: Path) -> dict:
    artifact = next(row for row in provenance["results"] if row["source"] == source)
    path = reference_root / artifact["local_path"]
    if sha256_file(path) != artifact["sha256"]:
        raise ValueError(f"Reference result changed: {source}")
    return json.loads(path.read_text())


def _matches(case: ParityCase, row: dict) -> bool:
    return (row.get("discipline") == case.discipline and row.get("seed") == case.seed
            and row.get("dr") == case.dr and row.get("track", "skidpad") == case.track_id
            and row.get("command_mode", "recorded") == case.command_mode)


def reference_result(case: ParityCase, reference_root: Path = REFERENCE_ROOT) -> tuple[int, dict]:
    provenance = reference_provenance(reference_root)
    data = _reference_document(case.reference_source, provenance, reference_root)
    rows = data.get("runs", [data])
    matches = [(index, row) for index, row in enumerate(rows) if _matches(case, row)]
    if len(matches) != 1:
        raise ValueError(f"Reference must contain exactly one row for {case.case_id}")
    return matches[0]


def validate_reference(reference_root: Path = REFERENCE_ROOT) -> dict:
    provenance = reference_provenance(reference_root)
    if provenance["checkpoint"]["sha256"] != CHECKPOINT_SHA256:
        raise ValueError("Reference checkpoint identity changed")
    for artifact in provenance["results"] + provenance["tracks"]:
        if sha256_file(reference_root / artifact["local_path"]) != artifact["sha256"]:
            raise ValueError(f"Reference artifact changed: {artifact['source']}")
    if tuple(row["track_id"] for row in provenance["tracks"]) != RETENTION_TRACKS:
        raise ValueError("Retention TrackJSON inventory changed")
    for track in provenance["tracks"]:
        bucket = zlib.crc32(track["track_id"].encode()) % 1000
        if bucket != track["crc32_bucket"] or bucket >= 100:
            raise ValueError("Retention track is outside the original test split")
        if track["matching_episodes"][-1]["episode"] != track["selected_episode"]:
            raise ValueError("Retention selection must reproduce the original last-index assignment")
    for dataset in provenance["datasets"].values():
        if any(row["train"] or row["val"] for row in dataset["retention_membership"].values()):
            raise ValueError("Retention track appears in a training or validation split")
    cases = all_cases()
    for source in RESULT_SOURCES.values():
        data = _reference_document(source, provenance, reference_root)
        rows = data.get("runs", [data])
        expected = [case for case in cases if case.reference_source == source]
        if len(rows) != len(expected) or any(sum(_matches(case, row) for row in rows) != 1 for case in expected):
            raise ValueError(f"Reference does not contain the exact case set: {source}")
        protocol = data.get("protocol", {})
        digest = protocol.get("sha256", data.get("checkpoint_sha256"))
        if digest is not None and digest != CHECKPOINT_SHA256:
            raise ValueError(f"Reference checkpoint mismatch: {source}")
        if any(row.get("controller", "pp") != "pp" for row in rows):
            raise ValueError("Reference controller differs from pp")
    final = _reference_document("runs/skidpad_clean/final-protocol.json", provenance, reference_root)
    if (final["sha256"] != CHECKPOINT_SHA256 or final["seeds"] != list(range(72000, 72010))
            or final["levels"] != list(LEVELS)):
        raise ValueError("Final evaluation protocol changed")
    return provenance


def verify_checkpoint(path: str | Path) -> str:
    digest = sha256_file(path)
    if digest != CHECKPOINT_SHA256:
        raise ValueError(f"Checkpoint must be the frozen custom evaluation checkpoint, got {digest}")
    return digest


def verify_environment_sources(repo_root: Path | None = None) -> dict[str, str]:
    root = repo_root or Path(__file__).resolve().parents[1]
    reference = {row["source"]: row["sha256"] for row in reference_provenance()["source_files"]}
    current = {source: sha256_file(root / source) for source in ENVIRONMENT_SOURCES}
    changed = [source for source, digest in current.items() if digest != reference[source]]
    if changed:
        raise ValueError(f"Environment source differs from the original custom evaluation: {changed}")
    return current


def verify_panorama_assets(directory: str | Path = "artifacts/panoramas") -> dict[str, str]:
    root = Path(directory)
    expected = {row["name"]: row["sha256"] for row in reference_provenance()["panoramas"]}
    current = {path.name: sha256_file(path) for path in sorted(root.glob("*.jpg"))}
    if current != expected:
        raise ValueError("Panorama assets differ from the frozen original directory inventory or contents")
    return current


def case_manifest(case: ParityCase) -> dict:
    index, result = reference_result(case)
    provenance = reference_provenance()
    artifact = next(row for row in provenance["results"] if row["source"] == case.reference_source)
    track = next((row for row in provenance["tracks"] if row["track_id"] == case.track_id), None)
    return {**asdict(case), "checkpoint_sha256": CHECKPOINT_SHA256, "controller": "pp", "max_time": 120,
            "noise": False, "venue_prob": 1.0, "reference_index": index,
            "reference_sha256": artifact["sha256"], "reference_result": result,
            "track_source": track["source"] if track else "fsop.track.fixed_layouts.skidpad()",
            "track_source_sha256": track["sha256"] if track else next(
                row["sha256"] for row in provenance["source_files"]
                if row["source"] == "src/fsop/track/fixed_layouts.py")}


def suite_manifest() -> dict:
    provenance = validate_reference()
    return {"schema_version": 1, "checkpoint_sha256": CHECKPOINT_SHA256, "reference_sha256": REFERENCE_SHA256,
            "case_count": 41, "case_counts": {"skidpad": 20, "retention": 16, "ablation": 4, "prior_failure": 1},
            "matching_contract": MATCHING_CONTRACT, "backend_differences": BACKEND_DIFFERENCES,
            "source_hashes": {row["source"]: row["sha256"] for row in provenance["source_files"]},
            "cases": [case_manifest(case) for case in all_cases()]}


class _InitializationRenderer:
    def load_scene(self, track, cones_xy, cone_colors, scene):
        pass

    def render(self, transforms):
        return {key: np.zeros((CAMERAS[key].height, CAMERAS[key].width, 3), dtype=np.uint8) for key in transforms}


def build_environment(case: ParityCase) -> FSEnv:
    """Return the original initialized environment immediately before its first policy observation.

    Placeholder panorama names replay the original filesystem-dependent sampling without requiring image files.
    Only the discarded reset image is synthetic. Its pixels do not affect random draws in postprocess.
    The caller must postprocess each actual policy observation in road, wide order using env.rng.
    """
    if get_case(case.case_id) != case:
        raise ValueError("Case parameters differ from the frozen registry")
    provenance = validate_reference()
    verify_environment_sources()
    if case.discipline == "trackdrive":
        artifact = next(row for row in provenance["tracks"] if row["track_id"] == case.track_id)
        track = load_json(REFERENCE_ROOT / artifact["local_path"])
        track.laps = 1
    else:
        track = skidpad()
    config = SimCfg(dr=case.dr, max_time=120, venue_prob=1.0)
    config.noise.enabled = False
    dr_config = dr_preset(case.dr)
    with tempfile.TemporaryDirectory(prefix="fsop-parity-panoramas-") as directory:
        for row in provenance["panoramas"]:
            (Path(directory) / row["name"]).touch()
        dr_config.panorama_dir = directory
        env = FSEnv(track, config, seed=case.seed, renderer=_InitializationRenderer(), dr_cfg=dr_config)
    if env.scene.sky_pano:
        env.scene.sky_pano = str(Path("artifacts/panoramas") / Path(env.scene.sky_pano).name)
    env.dr_cfg.panorama_dir = "artifacts/panoramas"
    env.renderer = None
    return env


def realized_environment(env: FSEnv) -> dict:
    """Snapshot the episode realization and current RNG without consuming either."""
    mission = {"L": env.mission.L, "s_end": env.mission.s_end, "ds": env.mission.ds,
               "closed": env.mission.closed, "laps": env.track.laps,
               "s_arr": env.mission.s_arr.tolist(), "kappa": env.mission.kappa.tolist(),
               "v_ref": env.mission.v_ref.tolist()}
    cones = {"xy": env.cones_xy.tolist(), "colors": [str(color) for color in env.cone_colors],
             "source_indices": env.cone_src.tolist()}
    snapshot = {"schema_version": 1, "seed": env.seed, "config": asdict(env.cfg), "dr_config": asdict(env.dr_cfg),
                "scene": env.scene.to_dict(), "mount": asdict(env.mount), "T_ego_cam": env.T_ego_cam().tolist(),
                "vehicle": asdict(env.vp), "expert_params": asdict(env.expert_params), "mission": mission,
                "track_sha256": json_sha256(env.track.to_dict()), "cones": cones,
                "state": asdict(env.state), "frame": env.frame, "s": env.s, "lat": env.lat, "hint": env.hint,
                "noise_timer": env.noise_timer, "frame_desire": env.frame_desire.tolist(),
                "route_decisions": env.route_commands.decisions, "rng_state": env.rng.bit_generator.state}
    # JSON normalization also makes tuples portable without touching the environment.
    snapshot = json.loads(json.dumps(snapshot, allow_nan=False))
    snapshot["realization_sha256"] = json_sha256(snapshot)
    return snapshot
