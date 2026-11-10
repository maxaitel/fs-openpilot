# ruff: noqa: E402, I001
import argparse
import hashlib
import json
import math
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
p.add_argument("--learner-controller", default="pp")
p.add_argument("--phase-schedule", type=Path)
p.add_argument("--takeover", type=float, default=0.0)
p.add_argument("--lateral-takeover", type=float, default=0.6)
p.add_argument("--speed", type=float, default=8.0)
p.add_argument("--alat", type=float, default=4.0)
p.add_argument("--controller", default="pure_pursuit")
p.add_argument("--lookahead", type=float, default=0.25)
a = p.parse_args()
if a.checkpoint and not a.phase_schedule and a.learner_controller != "pp":
    p.error("Checkpoint collection without a phase schedule requires the pp learner controller")
from isaac_lab_fsop.runtime import launch, shutdown

launcher = launch(render=True)
app = launcher.app
drive = None
try:
    import numpy as np
    import torch
    from fsop.eval.closed_loop import ModelPolicy
    from fsop.eval.controller import controller_contract
    from isaac_lab_fsop.recovery_schedule import PhaseRecovery, PhaseSchedule, first_left_decision, phase_control, phase_proposal
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
    if not math.isfinite(a.takeover) or a.takeover < 0:
        raise ValueError("Takeover time must be finite and nonnegative")
    if not math.isfinite(a.lateral_takeover) or a.lateral_takeover <= 0:
        raise ValueError("Lateral takeover threshold must be finite and positive")
    if a.takeover and a.checkpoint is None:
        raise ValueError("Learner-prefix recovery requires a checkpoint")
    schedule = PhaseSchedule.from_dict(json.loads(a.phase_schedule.read_text())) if a.phase_schedule else None
    if schedule and (a.checkpoint is None or a.takeover != 0 or case.discipline != "skidpad"):
        raise ValueError("Phase recovery requires a skidpad checkpoint and no legacy takeover")
    checkpoint_sha = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() if a.checkpoint else None
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
    expert = NativeExpert(drive, ExpertParams(v_cap=a.speed, a_lat_max=a.alat,
                                            controller=a.controller, k_lookahead=a.lookahead))
    policy = ModelPolicy.from_checkpoint(a.checkpoint, a.learner_controller) if a.checkpoint else None
    phase = PhaseRecovery(schedule, first_left_decision(drive.env.track, expert.mission)) if schedule else None
    observation_ticks = []
    activation_snapshot = None
    takeover_snapshot = None
    if policy:
        policy.model.requires_grad_(False)
        policy.reset()
    ts = TemporalState()
    frames = {"road": [], "wide": []}
    ticks, states, controls, teacher, desires, contacts = [], [], [], [], [], []
    latitudes = []
    taken = a.takeover <= 0
    takeover_reason = "expert_from_start" if taken else None
    takeover_tick = 0 if taken else None
    for frame in range(2400):
        states.append(obs.state.as_array())
        contacts.append(len(drive.hits))
        desire = ts.push_desire(obs.desire)
        latitudes.append(obs.lat)
        if phase:
            route = drive.env.referee.skidpad_route.report()
            prefix_clean = (route["valid"] and route["accepted_route"] == ["right"]
                            and not drive.hits and drive.env.referee.n_oc == 0)
            state = phase.advance(frame, obs.s, obs.lat, prefix_clean=prefix_clean)
            if phase.activation_tick == frame or phase.takeover_tick == frame:
                snapshot = {"progress_m": float(obs.s), "lateral_m": float(obs.lat),
                            "state": obs.state.as_array().tolist(),
                            "route": drive.env.referee.skidpad_route.report(),
                            "contacts": len(drive.hits), "off_course": drive.env.referee.n_oc}
                if phase.activation_tick == frame:
                    activation_snapshot = snapshot
                if phase.takeover_tick == frame:
                    takeover_snapshot = snapshot
            taken = state != "learner"
        if not phase and not taken and (obs.t >= a.takeover or abs(obs.lat) >= a.lateral_takeover):
            taken = True
            takeover_tick = frame
            takeover_reason = "lateral_threshold" if abs(obs.lat) >= a.lateral_takeover else "time_limit"
        obs = drive.observe(render=True)
        if frame % 4 == 0:
            ticks.append(frame)
            desires.append(desire)
            for name in frames:
                frames[name].append(rgb_to_model(obs.images[name]))
        if phase:
            with torch.inference_mode():
                action = phase_control(policy, expert, obs, drive.env, state)
            observation_ticks.append(frame)
        elif taken:
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
            obs = drive.observe(render=True)
            if frame % 4 == 0:
                ticks.append(frame)
                desires.append(desire)
                for name in frames:
                    frames[name].append(rgb_to_model(obs.images[name]))
            if phase:
                with torch.inference_mode():
                    phase_proposal(policy, obs, drive.env, teacher=True)
                observation_ticks.append(frame)
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
    if phase:
        takeover_tick, takeover_reason = phase.takeover_tick, phase.takeover_reason
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
        "capture_hz": 20,
        "physics_hz": 120,
        "control_hz": 20,
        "labels": "future actually executed native expert controls; contact and horizon masks applied by dataset",
        "checkpoint_sha256": checkpoint_sha,
        "collection_policy": {
            "mode": "learner_prefix" if a.checkpoint and a.takeover > 0 else "expert",
            "requested_takeover_s": a.takeover,
            "lateral_takeover_threshold_m": a.lateral_takeover,
            "actual_takeover_tick": takeover_tick,
            "takeover_reason": takeover_reason,
        },
        "arrays_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in a.out.glob("*.npy")},
    }
    if a.checkpoint:
        metadata["controller_contract"] = controller_contract(a.learner_controller)
    if phase:
        metadata["collection_policy"] = {
            "mode": "phase_recovery_v1", "schedule": asdict(schedule),
            "controller_state_during_teacher": "reset_after_each_discarded_proposal",
            "actual_activation_tick": phase.activation_tick, "activation_reason": phase.activation_reason,
            "activation": activation_snapshot, "activation_progress_m": phase.activation_progress_m,
            "actual_takeover_tick": takeover_tick, "takeover_reason": takeover_reason,
            "takeover": takeover_snapshot, "policy_observation_ticks": observation_ticks,
        }
    (a.out / "track.json").write_text(json.dumps(drive.env.track.to_dict()))
    if a.checkpoint and hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() != checkpoint_sha:
        raise RuntimeError("Learner checkpoint changed during collection")
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
