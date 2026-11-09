# ruff: noqa: E402, I001
import argparse
import hashlib
import json
import platform
from zipfile import ZipFile, ZIP_DEFLATED
import time
from dataclasses import asdict
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--case", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--membership", type=Path, default=Path("docs/isaac-lab-training/membership.json"))
p.add_argument("--checkpoint", type=Path)
p.add_argument("--takeover", type=float, default=0.0)
p.add_argument("--speed", type=float, default=8.0)
p.add_argument("--alat", type=float, default=4.0)
p.add_argument("--controller", default="pure_pursuit")
a = p.parse_args()
from isaac_lab_fsop.runtime import launch, shutdown

launcher = launch(render=True)
app = launcher.app
drive = None
try:
    import numpy as np
    import torch
    from fsop.eval.closed_loop import ModelPolicy
    from fsop.geometry.yuv import rgb_to_model
    from fsop.model.temporal import TemporalState
    from fsop.sim.expert import ExpertParams
    from fsop.track.schema import CONE_CLASS_INDEX
    from isaac_lab_fsop.cases import Case, environment
    from isaac_lab_fsop.env import LabDrive
    from isaac_lab_fsop.expert import NativeExpert
    from isaac_lab_fsop.evaluate import source_inventory

    case = Case(**json.loads(a.case.read_text()))
    membership = json.loads(a.membership.read_text())
    if asdict(case) not in membership["cases"]:
        raise ValueError("Case is outside the frozen membership")
    if case.split not in ("train", "val"):
        raise ValueError("Collection permits training/development cases only")
    if a.takeover and a.checkpoint is None:
        raise ValueError("Learner-prefix recovery requires a checkpoint")
    a.out.mkdir(parents=True, exist_ok=False)
    source_hashes = source_inventory()
    with ZipFile(a.out / "source.zip", "x", compression=ZIP_DEFLATED) as archive:
        for path in source_hashes:
            archive.write(path, path)
    started = time.perf_counter()
    drive = LabDrive(environment(case), variant=case.variant)
    x, y, yaw = drive.env.track.start_pose
    start_pose = (x - case.lateral_start * np.sin(yaw), y + case.lateral_start * np.cos(yaw), yaw + case.yaw_start)
    obs = drive.reset(start_pose=start_pose)
    expert = NativeExpert(drive, ExpertParams(v_cap=a.speed, a_lat_max=a.alat, controller=a.controller))
    policy = ModelPolicy.from_checkpoint(a.checkpoint) if a.checkpoint else None
    if policy:
        policy.model.requires_grad_(False)
        policy.reset()
    ts = TemporalState()
    frames = {"road": [], "wide": []}
    ticks, states, controls, teacher, desires, contacts = [], [], [], [], [], []
    latitudes = []
    taken = a.takeover <= 0
    for frame in range(2400):
        states.append(obs.state.as_array())
        contacts.append(len(drive.hits))
        desire = ts.push_desire(obs.desire)
        latitudes.append(obs.lat)
        if not taken and (obs.t >= a.takeover or abs(obs.lat) >= 0.6):
            taken = True
        if frame % 4 == 0 or not taken:
            obs = drive.observe(render=True)
        if frame % 4 == 0:
            ticks.append(frame)
            desires.append(desire)
            for name in frames:
                frames[name].append(rgb_to_model(obs.images[name]))
        if taken:
            action = expert.act()
        else:
            with torch.inference_mode():
                action = policy(obs, drive.env)[:2]
        controls.append(action)
        teacher.append(taken)
        obs = drive.step(*action, render=False)
        if frame % 200 == 0:
            print("COLLECT", case.id, frame, round(obs.s, 2), round(obs.lat, 3), len(drive.hits), flush=True)
        if drive.env.referee.done:
            break
    summary = drive.summary()
    terminal_start = drive.env.frame
    if summary["result"] == "finished":
        for _ in range(200):
            frame = drive.env.frame
            states.append(obs.state.as_array())
            contacts.append(len(drive.hits))
            desire = ts.push_desire(obs.desire)
            if frame % 4 == 0:
                obs = drive.observe(render=True)
                ticks.append(frame)
                desires.append(desire)
                for name in frames:
                    frames[name].append(rgb_to_model(obs.images[name]))
            controls.append((float(drive.act.angle), -2.0))
            teacher.append(True)
            obs = drive.step_terminal_brake(render=False)
    states.append(obs.state.as_array())
    contacts.append(len(drive.hits))
    for name, values in frames.items():
        np.save(a.out / f"{name}.npy", np.asarray(values, dtype=np.uint8))
    for name, values in {
        "ticks": ticks,
        "states": states,
        "controls": controls,
        "teacher": teacher,
        "desires": desires,
        "contacts": contacts,
    }.items():
        np.save(a.out / f"{name}.npy", np.asarray(values))
    classes = np.array([CONE_CLASS_INDEX.get(c, -1) for c in drive.env.cone_colors])
    np.save(a.out / "cones.npy", np.c_[drive.env.cones_xy, classes])
    metadata = {
        "source_sha256": source_hashes,
        "source_archive_sha256": hashlib.sha256((a.out / "source.zip").read_bytes()).hexdigest(),
        "membership_sha256": hashlib.sha256(a.membership.read_bytes()).hexdigest(),
        "calibration": {"mount": asdict(drive.env.mount), "T_ego_cam": drive.env.T_ego_cam().tolist()},
        "scene": drive.env.scene.to_dict(),
        "render_settings": drive.sim.fsop_render_settings,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "physics_dt": drive.sim.cfg.dt,
            "device": drive.sim.cfg.device,
            "use_fabric": drive.sim.cfg.use_fabric,
            "scene_queries": drive.sim.cfg.enable_scene_query_support,
            "physx": drive.sim.cfg.physx.to_dict(),
        },
        "schema_version": 1,
        "backend": "IsaacLab",
        "id": case.id,
        "split": case.split,
        "track_id": drive.env.track.id,
        "discipline": case.discipline,
        "seed": case.seed,
        "case": asdict(case),
        "vehicle": asdict(drive.env.vp),
        "teacher": asdict(expert.params),
        "summary": summary,
        "terminal_continuation": {
            "start_tick": terminal_start,
            "end_tick": drive.env.frame,
            "controller": "hold steering and native braking at -2 m/s2",
            "final_state": asdict(drive.env.state),
            "contacts": len(drive.hits),
        },
        "appearance": drive.appearance,
        "frames": len(ticks),
        "wall_seconds": time.perf_counter() - started,
        "image_hz": 5,
        "physics_hz": 120,
        "control_hz": 20,
        "labels": "future actually executed native expert controls; contact and horizon masks applied by dataset",
        "checkpoint_sha256": hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() if a.checkpoint else None,
        "arrays_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in a.out.glob("*.npy")},
    }
    (a.out / "track.json").write_text(json.dumps(drive.env.track.to_dict()))
    if source_inventory() != source_hashes:
        raise RuntimeError("Source changed during collection")
    (a.out / "meta.json").write_text(json.dumps(metadata, indent=2))
    print("COLLECTION_RESULT", json.dumps({"id": case.id, "frames": len(ticks), **summary}), flush=True)
except BaseException:
    import traceback

    traceback.print_exc()
    raise
finally:
    shutdown(app, drive)
