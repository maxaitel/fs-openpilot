"""Post-replay accounting of the frozen support failure. No parameter changes."""
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from math import cos, hypot, sin
from pathlib import Path
import json

from planner import ConeClass, CorridorPlanner, Recipe, Track

OUT = Path(__file__).resolve().parent
p = Recipe.load()
source = OUT / "rows.json"
all_rows = json.loads(source.read_text())
report = {}
for name, variants in all_rows.items():
    counts = Counter()
    for row in variants["streaming"]:
        tracks = tuple(Track(t["identifier"], tuple(t["xy"]), ConeClass(t["cone_class"]),
                             t["score"], t["last_seen"], t["supported_at"])
                       for t in row["result"]["diagnostics"]["observed_tracks"])
        planner = CorridorPlanner(p)
        planner._tracks = tracks
        gates, _, _, _ = planner._gates(row["timestamp"])
        assert len(gates) == row["result"]["diagnostics"]["gate_count"]
        seeds = []
        for gate in gates:
            longitudinal = gate.midpoint[0] * cos(gate.heading) + gate.midpoint[1] * sin(gate.heading)
            lateral = -gate.midpoint[0] * sin(gate.heading) + gate.midpoint[1] * cos(gate.heading)
            if -p.max_behind_m <= longitudinal <= p.max_origin_extension_m and abs(lateral) <= p.max_origin_offset_m:
                seeds.append(gate)
        counts["no_gates" if not gates else "gates_but_no_nearfield_seed" if not seeds else "nearfield_seed_but_no_accepted_path"] += 1
        retained = [t for t in tracks if t.supported_at is not None and t.cone_class != ConeClass.UNKNOWN
                    and row["timestamp"] - t.supported_at <= p.max_track_age_s]
        counts["frames_with_range_supported_retained_tracks"] += bool(retained)
        counts["frames_with_retained_track_excluded_by_freshness"] += any(row["timestamp"] - t.last_seen > p.max_observation_gap_s for t in retained)
        counts["frames_with_color_conflicted_tracks"] += any(t.cone_class == ConeClass.UNKNOWN for t in tracks)
        counts["frames_with_any_near_retained_track"] += any(hypot(*t.xy) < p.support_range_min_m for t in retained)
    report[name] = dict(counts)
result = {"rows_sha256": sha256(source.read_bytes()).hexdigest(), "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
          "scope": "Post-replay descriptive counts. Reuses frozen gate eligibility exactly. No alternative recipe was evaluated.",
          "counts": report}
encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
out = OUT / "failure_accounting.json"
if out.exists():
    assert out.read_text() == encoded
else:
    out.write_text(encoded)
print(encoded)
