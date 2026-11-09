import copy

import numpy as np

from isaac_lab_fsop.appearance_probe import stability_analysis


def test_repeat_noise_gates_attribution_and_preserves_unresolved_effects():
    names = ("matched", "weather_only", "marks_only", "props_only", "full", "no_props",
             "matched_repeat", "full_repeat")
    results = {name: {"plan": np.zeros((33, 15)), "action": [0.0, 0.0]} for name in names}
    results["full"]["plan"][:, 1] = 0.2
    results["full_repeat"]["plan"][:, 1] = 0.201
    result = stability_analysis(results)
    assert result["interpretable"]
    assert result["contrasts"]["full_minus_no_props"]["effect_above_noise"]["near_xy_m"]
    assert not result["contrasts"]["props_only_minus_matched"]["effect_above_noise"]["near_xy_m"]
    unstable = copy.deepcopy(results)
    unstable["matched_repeat"]["action"][0] = 0.003
    result = stability_analysis(unstable)
    assert not result["interpretable"]
    assert not any(flag for row in result["contrasts"].values() for flag in row["effect_above_noise"].values())
    assert result["contrasts"]["full_minus_no_props"]["magnitude"]["near_xy_m"] == 0.2
