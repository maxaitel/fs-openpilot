import hashlib
import json
from pathlib import Path

from fsop.eval.controller import controller_contract
from scripts.verify_lab_recovery import verify
from test_lab_recovery_gate import RECIPE, episode  # noqa: F401


def sha(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


def test_v2_retains_only_source_bound_legacy_episode(episode, tmp_path):
    original = verify(episode, RECIPE)
    assert original["passed"], original["reasons"]
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(original))
    old = json.loads(RECIPE.read_text())
    old["schema_version"] = 2
    old["membership"] = str((RECIPE.parent / old["membership"]).resolve())
    record = {"checkpoint_sha256": old["checkpoint_sha256"], "controller_contract": controller_contract("pp"),
              "collection": {"mode": "legacy_learner_prefix_v1", **old["collection"]},
              "legacy_receipt": {"path": str(receipt), "sha256": sha(receipt), "recipe": str(RECIPE),
                                 "recipe_sha256": sha(RECIPE)}}
    old["collection_by_case"] = {case: record for case in old["recovery_case_ids"]}
    (tmp_path / "cases").symlink_to(RECIPE.parent / "cases", target_is_directory=True)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps(old))
    result = verify(episode, recipe)
    assert result["passed"], result["reasons"]
    meta_path = episode / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["controller_contract"] = controller_contract("op_tangent_pi_v1")
    meta_path.write_text(json.dumps(meta))
    forged_receipt = {**original, "hashes": {**original["hashes"], "meta_sha256": sha(meta_path)}}
    receipt.write_text(json.dumps(forged_receipt))
    record["legacy_receipt"]["sha256"] = sha(receipt)
    recipe.write_text(json.dumps(old))
    result = verify(episode, recipe)
    assert not result["passed"] and any("controller contract" in reason for reason in result["reasons"])
    meta.pop("controller_contract")
    meta_path.write_text(json.dumps(meta))
    receipt.write_text(json.dumps({**original, "passed": False}))
    result = verify(episode, recipe)
    assert not result["passed"] and any("digest" in reason for reason in result["reasons"])
