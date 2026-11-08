import copy
from collections import Counter

import numpy as np
import pytest
from shapely.geometry import Polygon

from isaac_eval.parity_cases import build_environment, get_case
from isaac_eval.stress_layout import build_layout, layout_sha256, protected_area


@pytest.mark.parametrize(
    "case_id",
    [
        "skidpad-72000-light-recorded",
        "skidpad-72000-heavy-recorded",
        "retention-trackdrive_020000-light-recorded",
        "retention-trackdrive_020000-heavy-recorded",
    ],
)
def test_fixed_layout_clearance_and_rng(case_id):
    env = build_environment(get_case(case_id))
    original = copy.deepcopy(env.rng.bit_generator.state)
    layout = build_layout(env)
    assert env.rng.bit_generator.state == original
    assert layout_sha256(build_layout(env)) == layout_sha256(layout)
    corridor = protected_area(env)
    counts = Counter(item["kind"] for item in layout["objects"])
    assert all(
        counts[kind] > 0
        for kind in (
            "person_business",
            "person_worker",
            "person_police",
            "wall",
            "barrier",
            "worktable",
            "tires",
            "barrel_blue",
            "barrel_yellow",
            "equipment",
        )
    )
    footprints = [Polygon(obj["footprint_xy"]) for obj in layout["objects"]]
    for obj, shape in zip(layout["objects"], footprints, strict=True):
        assert shape.is_valid
        assert shape.distance(corridor) >= 0.15
        assert shape.distance(corridor) == pytest.approx(obj["clearance_m"])
    for i, shape in enumerate(footprints):
        assert all(shape.distance(other) >= 0.2 - 1e-9 for other in footprints[:i])
    surface = layout["surface"]
    assert Counter(item["kind"] for item in surface) == {
        "tar": 28,
        "rubber": 18,
        "paint": 12,
    }
    for item in surface:
        if item["kind"] in ("tar", "rubber", "paint"):
            assert item["size"][1] <= 0.075
    assert np.isfinite(np.array([item["position"] for item in layout["surface"]])).all()
    weather = layout["weather"]
    bounds = np.asarray(weather["bounds_m"])
    assert np.all(bounds[0] <= np.min(env.track.centerline, axis=0) - 30)
    assert np.all(bounds[1] >= np.max(env.track.centerline, axis=0) + 30)
    assert weather["fields"]
    assert weather["gain_limits"][0] < 1 < weather["gain_limits"][1]
