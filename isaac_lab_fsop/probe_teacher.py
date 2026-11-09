# ruff: noqa: E402, I001
import argparse
import json
import time
import hashlib
from dataclasses import asdict
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

p = argparse.ArgumentParser()
p.add_argument("--case", default="retention-trackdrive_020000-heavy-recorded")
p.add_argument("--variant", default="matched")
p.add_argument("--case-file", type=Path)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--render", action="store_true")
p.add_argument("--contact-test", action="store_true")
p.add_argument("--steps", type=int, default=2400)
p.add_argument("--controller", default="pure_pursuit")
p.add_argument("--speed", type=float, default=8)
p.add_argument("--alat", type=float, default=4)
p.add_argument("--lookahead", type=float, default=0.25)
a = p.parse_args()
from isaac_lab_fsop.runtime import launch

launcher = launch(render=a.render)
app = launcher.app
import numpy as np
from isaac_lab_fsop.env import LabDrive
from isaac_lab_fsop.expert import NativeExpert
from isaac_eval.parity_cases import build_environment, get_case
from fsop.sim.expert import ExpertParams

start = time.perf_counter()
a.out.mkdir(parents=True, exist_ok=False)
from isaac_lab_fsop.evaluate import source_inventory

sources = source_inventory()
with ZipFile(a.out / "source.zip", "x", compression=ZIP_DEFLATED) as archive:
    for path in sources:
        archive.write(path, path)
if a.case_file:
    from isaac_lab_fsop.cases import Case, environment

    case = Case(**json.loads(a.case_file.read_text()))
    metadata_env = environment(case)
    a.variant = case.variant
else:
    metadata_env = build_environment(get_case(a.case))
drive = LabDrive(metadata_env, variant=a.variant, render=a.render)
pose = (*[float(x) for x in drive.env.cones_xy[0]], 0.0) if a.contact_test else None
if pose:
    pose = (pose[0] - 3, pose[1], pose[2])
obs = drive.reset(start_pose=pose)
expert = NativeExpert(drive, ExpertParams(v_cap=a.speed, a_lat_max=a.alat,
                                        controller=a.controller, k_lookahead=a.lookahead))
rows = []
for i in range(a.steps):
    steer, accel = (0.0, 3.0) if a.contact_test else expert.act()
    rows.append([*obs.state.as_array(), obs.s, obs.lat, steer, accel, len(drive.hits)])
    obs = drive.step(steer, accel, render=a.render and (i + 1) % 4 == 0)
    if i % 200 == 0:
        print("TEACHER", i, obs.s, obs.lat, len(drive.hits), flush=True)
    if drive.env.referee.done:
        break
summary = drive.summary()
summary.update(
    case=a.case,
    wall_seconds=time.perf_counter() - start,
    physics_backend=type(drive.sim).__module__,
    expert=asdict(expert.params),
    case_config=asdict(case) if a.case_file else None,
    source_sha256=sources,
    source_archive_sha256=hashlib.sha256((a.out / "source.zip").read_bytes()).hexdigest(),
    appearance=drive.appearance,
)
if source_inventory() != sources:
    raise RuntimeError("Source changed during native teacher probe")
np.save(a.out / "states.npy", np.array(rows))
(a.out / "summary.json").write_text(json.dumps(summary, indent=2))
print("TEACHER_RESULT", json.dumps(summary), flush=True)
print("CLOSE_DRIVE", flush=True)
drive.close()
print("CLOSE_APP", flush=True)
app.close(wait_for_replicator=False)

print("CLOSED", flush=True)
