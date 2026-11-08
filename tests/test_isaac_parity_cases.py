import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from fsop.sim.env import CAMERAS, FSEnv, SimCfg
from fsop.sim.randomization import dr_preset
from fsop.track.fixed_layouts import skidpad
from fsop.track.io import load_json
from isaac_eval.parity_cases import (
    CHECKPOINT_SHA256,
    REFERENCE_ROOT,
    RETENTION_TRACKS,
    all_cases,
    build_environment,
    case_manifest,
    get_case,
    realized_environment,
    reference_provenance,
    reference_result,
    suite_manifest,
    validate_reference,
    verify_checkpoint,
    verify_environment_sources,
    verify_panorama_assets,
)
from isaac_eval.prepare_parity_cases import write_frozen_json


class ConstantRenderer:
    def __init__(self, value=117):
        self.value = value
        self.calls = 0

    def load_scene(self, track, cones_xy, cone_colors, scene):
        pass

    def render(self, transforms):
        self.calls += 1
        return {key: np.full((CAMERAS[key].height, CAMERAS[key].width, 3), self.value, dtype=np.uint8)
                for key in transforms}


@pytest.fixture(scope="module")
def panorama_directory(tmp_path_factory):
    root = tmp_path_factory.mktemp("original-panoramas")
    for row in reference_provenance()["panoramas"]:
        (root / row["name"]).touch()
    return root


def original_environment(case, panorama_directory, renderer):
    track = (load_json(REFERENCE_ROOT / "tracks" / f"{case.track_id}.json")
             if case.discipline == "trackdrive" else skidpad())
    if case.discipline == "trackdrive":
        track.laps = 1
    cfg = SimCfg(dr=case.dr, max_time=120, venue_prob=1.0, render=renderer is not None)
    cfg.noise.enabled = False
    dr = dr_preset(case.dr)
    dr.panorama_dir = str(panorama_directory)
    env = FSEnv(track, cfg, seed=case.seed, renderer=renderer, dr_cfg=dr)
    if env.scene.sky_pano:
        env.scene.sky_pano = str(Path("artifacts/panoramas") / Path(env.scene.sky_pano).name)
    env.dr_cfg.panorama_dir = "artifacts/panoramas"
    return env


def test_exact_case_inventory_and_original_reference_rows():
    cases = all_cases()
    assert len(cases) == len({case.case_id for case in cases}) == 41
    assert Counter(case.group for case in cases) == {"skidpad": 20, "retention": 16, "ablation": 4, "prior_failure": 1}
    assert {(case.seed, case.dr) for case in cases if case.group == "skidpad"} == {
        (seed, dr) for seed in range(72000, 72010) for dr in ("light", "heavy")}
    assert {(case.track_id, case.dr, case.seed) for case in cases if case.group == "retention"} == {
        (tid, dr, 0) for tid in RETENTION_TRACKS for dr in ("light", "heavy")}
    assert {(case.seed, case.dr, case.command_mode) for case in cases if case.group == "ablation"} == {
        (72000, dr, mode) for dr in ("light", "heavy") for mode in ("zero", "swapped")}
    assert validate_reference()["checkpoint"]["sha256"] == CHECKPOINT_SHA256
    for case in cases:
        assert get_case(case.case_id) == case
        manifest = case_manifest(case)
        assert manifest["reference_result"] == reference_result(case)[1]
        assert manifest["controller"] == "pp"
    manifest = suite_manifest()
    assert len(manifest["cases"]) == 41
    assert "not pixel-identical" in manifest["backend_differences"]["renderer"]


@pytest.mark.parametrize("case", all_cases(), ids=lambda case: case.case_id)
def test_matches_original_initialization_and_first_observation(case, panorama_directory):
    original_renderer = ConstantRenderer()
    original = original_environment(case, panorama_directory, original_renderer)
    reproduced = build_environment(case)
    assert original_renderer.calls == 1
    assert reproduced.renderer is None
    assert realized_environment(reproduced) == realized_environment(original)
    reproduced.renderer = ConstantRenderer()
    actual = reproduced.observe()
    expected = original.observe()
    assert original_renderer.calls == 2
    assert reproduced.renderer.calls == 1
    assert reproduced.rng.bit_generator.state == original.rng.bit_generator.state
    for camera in ("road", "wide"):
        np.testing.assert_array_equal(actual.images[camera], expected.images[camera])
    np.testing.assert_array_equal(actual.desire, expected.desire)
    assert actual.state == expected.state


def test_skipping_reset_image_changes_first_policy_noise(panorama_directory):
    case = all_cases()[0]
    omitted = original_environment(case, panorama_directory, renderer=None)
    matched = build_environment(case)
    assert omitted.rng.bit_generator.state != matched.rng.bit_generator.state
    omitted.renderer = ConstantRenderer()
    matched.renderer = ConstantRenderer()
    incorrect, correct = omitted.observe(), matched.observe()
    assert not np.array_equal(incorrect.images["road"], correct.images["road"])


def test_missing_panorama_directory_does_not_silently_change_sampling(tmp_path):
    with pytest.raises(ValueError, match="Panorama assets differ"):
        verify_panorama_assets(tmp_path)
    case = all_cases()[0]
    changed = original_environment(case, tmp_path, ConstantRenderer())
    correct = build_environment(case)
    assert changed.scene != correct.scene
    assert changed.expert_params != correct.expert_params


def test_retention_tracks_are_byte_exact_and_excluded_from_training():
    provenance = validate_reference()
    for artifact in provenance["tracks"]:
        track = load_json(REFERENCE_ROOT / artifact["local_path"])
        assert track.id == artifact["track_id"]
        assert artifact["crc32_bucket"] < 100
        assert artifact["selected_episode"] == artifact["matching_episodes"][-1]["episode"]
        for dataset in provenance["datasets"].values():
            membership = dataset["retention_membership"][track.id]
            assert membership["train"] == membership["val"] == []
    assert "spark_skidpad_intent" in provenance["datasets"]
    assert "spark_full_venue" in provenance["datasets"]


def test_source_guard_and_snapshot_are_read_only():
    assert verify_environment_sources()
    env = build_environment(all_cases()[0])
    first = realized_environment(env)
    second = realized_environment(env)
    assert first == second
    first["rng_state"]["state"]["state"] = 0
    assert env.rng.bit_generator.state == second["rng_state"]


def test_checkpoint_and_manifest_reject_changes(tmp_path):
    wrong = tmp_path / "wrong.pt"
    wrong.write_bytes(b"wrong checkpoint")
    with pytest.raises(ValueError, match="frozen custom evaluation checkpoint"):
        verify_checkpoint(wrong)
    path = tmp_path / "manifest.json"
    write_frozen_json(path, {"fixed": True})
    write_frozen_json(path, {"fixed": True})
    assert json.loads(path.read_text()) == {"fixed": True}
    with pytest.raises(ValueError, match="different frozen parity artifact"):
        write_frozen_json(path, {"fixed": False})
    with pytest.raises(ValueError, match="Unknown frozen parity case"):
        get_case("skidpad-72010-light-recorded")
