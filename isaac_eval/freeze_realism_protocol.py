"""Freeze source/assets after fixed-pose verification, before scored runs."""

import hashlib
import json
from pathlib import Path

from appearance_guard import evidence_sources
from rendering import PROFILE

root = Path(__file__).resolve().parent.parent
sources = evidence_sources()
base = json.loads((root / "isaac_eval/protocol.json").read_text())
verifications = {}
for discipline in ["track", "skidpad"]:
    pairs = []
    for scene in ["plain", "paddock"]:
        path = root / f"isaac_eval/runs/preview_{scene}_{discipline}/verification.json"
        record = json.loads(path.read_text())
        assert record["source_sha256"] == sources, f"Stale preview evidence: {path}"
        pairs.append(record)
        verifications[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert pairs[0]["physics_sha256"] == pairs[1]["physics_sha256"], discipline
    assert pairs[0]["poses"] == pairs[1]["poses"], discipline
gate = root / "isaac_eval/runs/integration_realistic/verification.json"
integration = json.loads(gate.read_text())
assert integration["pass"] and integration["scene"] == "paddock", "Realistic integration gate failed"
assert integration["source_sha256"] == sources, "Stale integration evidence"
base["integration_report_sha256"] = hashlib.sha256(gate.read_bytes()).hexdigest()
base["integration_report"] = str(gate.relative_to(root))
base["render_profile"] = PROFILE
base["render_subframes"] = PROFILE["subframes"]
base["version"] = "realism-1"
base["experiment"] = "paired appearance-only paddock vs plain; frozen model"
base["baseline_eval_commit"] = "b2d78ab"
base["preview_verification_sha256"] = verifications
files = list(base["source_sha256"]) + [
    "isaac_eval/realistic_scene.py",
    "isaac_eval/appearance_guard.py",
    "isaac_eval/preview_realism.py",
    "isaac_eval/export_scene.py",
    "isaac_eval/realistic_car.py",
    "isaac_eval/verify_integration.py",
    "isaac_eval/realism_assets.py",
    "isaac_eval/assets/manifest.json",
    "src/fsop/geometry/camera.py",
]
files += [str(p.relative_to(root)) for p in sorted((root / "src/fsop").rglob("*.py"))]
files += ["isaac_eval/rendering.py", "isaac_eval/run_realism_suite.sh"]
base["source_sha256"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}
base["variants"] = ["plain", "paddock"]
base["ordering"] = "alternate AB/BA by seed within each discipline"
base["limitations"] = "Visual simulation only; unchanged dry friction, ideal cameras, no people or moving traffic."
(root / "isaac_eval/protocol_realism.json").write_text(json.dumps(base, indent=2) + "\n")
print("FROZEN protocol_realism.json", len(files), "source/manifest hashes")
