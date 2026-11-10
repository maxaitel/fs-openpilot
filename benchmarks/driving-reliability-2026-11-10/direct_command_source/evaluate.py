"""Evaluate one learned policy episode in native Isaac Lab and record its actual inputs.

Run from the repository root with Isaac Lab's Python. The case is a JSON encoding
of isaac_lab_fsop.cases.Case; no historical protocol or baseline is implied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default=None, help="Torch device; default follows NativeActuatorPolicy")
    parser.add_argument("--expected-sha", help="Optional expected complete checkpoint SHA-256")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pilot-steps", type=int, default=0)
    parser.add_argument("--async-evidence", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.pilot_steps <= 2400:
        parser.error("--pilot-steps must be between 0 and 2400")
    if args.expected_sha and (
        len(args.expected_sha) != 64 or any(c not in "0123456789abcdef" for c in args.expected_sha)
    ):
        parser.error("--expected-sha must be a lowercase SHA-256 digest")
    return args


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_digest(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def source_inventory():
    roots = ("src/fsop", "isaac_eval", "isaac_lab_fsop")
    paths = sorted(p for root in roots for p in Path(root).rglob("*.py"))
    paths.append(Path("scripts/audit_isaac_episode.py"))
    paths += [Path(".work/lab/direct_command_trial") / name for name in ("evaluate.py", "verify_command_episode.py", "actuator_trial.py", "process.py", "export_audited_command.py")]
    return {str(p): digest(p) for p in paths}


def main():
    args = parse_args()
    process_start = time.perf_counter()
    checkpoint_sha = digest(args.checkpoint)
    if args.expected_sha and checkpoint_sha != args.expected_sha:
        raise ValueError("Checkpoint SHA-256 does not match --expected-sha")
    from isaac_lab_fsop.runtime import launch, shutdown

    launcher = launch(render=True)
    app = launcher.app
    drive = None
    try:
        import carb
        import numpy as np
        import psutil
        import pynvml
        import torch
        from isaacsim.core.utils.extensions import enable_extension

        from actuator_trial import NativeActuatorPolicy
        from fsop.geometry.camera import CAMERA_MODELS
        from isaac_eval.evidence import EvidenceWriter, ImageHeadCapture, finalize_evidence
        from isaac_eval.parity_cases import get_case, json_sha256, realized_environment
        from isaac_eval.parity_protocol import GRAPH_FILES
        from isaac_eval.rendering import PROFILE
        from isaac_lab_fsop.cases import Case, environment
        from isaac_lab_fsop.env import LabDrive
        from isaac_lab_fsop.evidence import AsyncEvidenceWriter

        enable_extension("isaacsim.sensors.camera")
        case = Case(**json.loads(args.case.read_text()))
        command_mode = get_case(case.frozen_case).command_mode if case.frozen_case else "recorded"
        args.out.mkdir(parents=True, exist_ok=False)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        policy = NativeActuatorPolicy.load(args.checkpoint, device=args.device)
        policy.model.requires_grad_(False)
        state_sha = model_digest(policy.model)
        head_sha = model_digest(policy.head)
        base_parameter_count = sum(p.numel() for p in policy.model.parameters())
        head_parameter_count = sum(p.numel() for p in policy.head.parameters())
        env = environment(case)
        if env.cfg.max_time != 120:
            raise ValueError("Native evaluation requires the fixed 120-second limit")
        realization = realized_environment(env)
        (args.out / "case.json").write_text(json.dumps(asdict(case), indent=2) + "\n")
        (args.out / "realization.json").write_text(json.dumps(realization, indent=2) + "\n")
        (args.out / "track.json").write_text(json.dumps(env.track.to_dict(), indent=2) + "\n")
        drive = LabDrive(env, variant=case.variant, render=True)
        (args.out / "appearance.json").write_text(json.dumps(drive.appearance, indent=2) + "\n")
        x, y, yaw = env.track.start_pose
        start_pose = (x - case.lateral_start * np.sin(yaw), y + case.lateral_start * np.cos(yaw), yaw + case.yaw_start)
        # reset() produces the first cached observation and a single camera capture.
        obs = drive.reset(start_pose=start_pose)
        initial_state = asdict(env.state)
        policy.reset()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        # Flatten the exact physical scene and bundle all authored dependencies.
        from isaac_eval.export_scene import export_scene

        exported = export_scene(drive.stage, args.out / "scene.usda", Path.cwd())
        exported["sha256"] = {name: digest(args.out / "assets" / name) for name in exported["dependencies"]}
        (args.out / "export.json").write_text(json.dumps(exported, indent=2) + "\n")
        sources = source_inventory()
        from zipfile import ZIP_DEFLATED, ZipFile

        with ZipFile(args.out / "source.zip", "x", compression=ZIP_DEFLATED) as archive:
            for path in sources:
                archive.write(path, path)
        graphs = {p: digest(p) for p in GRAPH_FILES}
        settings = carb.settings.get_settings()
        physics_paths = (
            "/physics/disableContactProcessing",
            "/physics/collisionConeCustomGeometry",
            "/physics/collisionCylinderCustomGeometry",
        )
        physics_settings = {path: settings.get(path) for path in physics_paths}
        writer_type = AsyncEvidenceWriter if args.async_evidence else EvidenceWriter
        metadata = {
            "evidence_writer": writer_type.__name__,
            "schema_version": 4,
            "fps": 20,
            "backend": "Isaac Lab PhysX",
            "case": asdict(case),
            "checkpoint_sha256": checkpoint_sha,
            "command_contract": policy.command_contract,
            "controller_contract": {"mode": policy.controller_name, "command_contract": policy.command_contract},
            "controller_name": policy.controller_name,
            "policy_device": str(policy.device),
            "head_state_sha256": head_sha,
            "base_parameter_count": base_parameter_count,
            "head_parameter_count": head_parameter_count,
            "total_parameter_count": base_parameter_count + head_parameter_count,
            "checkpoint_size_bytes": args.checkpoint.stat().st_size,
            "state_sha256": state_sha,
            "track": env.track.to_dict(),
            "initial_cones_xyz": np.c_[env.cones_xy, np.full(len(env.cones_xy), 0.02)].tolist(),
            "cone_colors": [c.value for c in env.cone_colors],
            "expert_params": asdict(env.expert_params),
            "vehicle_params": asdict(env.vp),
            "label_dt": env.cfg.label_dt,
            "nominal_detection_calibration": env.T_ego_cam().tolist(),
            "intrinsics": {name: cam.K.tolist() for name, cam in CAMERA_MODELS.items()},
            "initial_physical_state": initial_state,
            "start_pose": list(start_pose),
            "source_sha256": sources,
            "source_archive_sha256": digest(args.out / "source.zip"),
            "graph_sha256": graphs,
            "variant": case.variant,
            "appearance_sha256": digest(args.out / "appearance.json"),
            "scene_sha256": digest(args.out / "scene.usda"),
            "export_sha256": digest(args.out / "export.json"),
            "realization_sha256": digest(args.out / "realization.json"),
            "asset_sha256": drive.appearance.get("asset_sha256"),
            "layout_sha256": drive.appearance.get("stress", {}).get("layout_sha256"),
            "physics_sha256": drive.appearance["physics_sha256"],
            "velocity_convention": "World XY velocity rotated by planar yaw; full body velocity saved separately",
            "privileged_inputs": [
                "speed",
                "sampled nominal camera calibration",
                "route commands from mission progress",
                "mission progress for launch/finish braking",
            ],
            "prediction_provenance": (
                "Frozen original model diagnostics and direct command head from one native policy call per 20 Hz tick"
            ),
            "uncertainty_semantics": "exp(clamped model second plan output), uncalibrated model scale",
            "image_history": "RGB processed once; image pair max(0,t-4),t; 96-step feature queue; 100-step pulse queue",
            "oracle": "Offline custom bicycle expert diagnostic, never supplied to policy or native PhysX",
            "pilot_only": bool(args.pilot_steps),
            "command_mode": command_mode,
            "render_profile": PROFILE,
            "sim_render_settings": drive.sim.fsop_render_settings,
            "physics_settings": physics_settings,
        }
        # The base exposes policy_raw as a method rather than a module. Wrap it only
        # to copy the exact output of the existing call; never change its arguments/results.
        base = policy.model.base if hasattr(policy.model, "base") else policy.model
        original_policy_raw = base.policy_raw
        raw_output = {}

        def capture_policy_raw(*args, **kwargs):
            output = original_policy_raw(*args, **kwargs)
            raw_output["output"] = [v.detach().cpu().numpy().copy() for v in output]
            return output

        base.policy_raw = capture_policy_raw
        telemetry, rows = [], []
        stop = threading.Event()
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        proc = psutil.Process()

        def monitor():
            while not stop.is_set():
                telemetry.append(
                    {
                        "wall_s": time.perf_counter() - run_start,
                        "gpu_pct": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                        "power_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
                        "temp_c": pynvml.nvmlDeviceGetTemperature(handle, 0),
                        "host_ram_gib": psutil.virtual_memory().used / 2**30,
                        "process_rss_gib": proc.memory_info().rss / 2**30,
                        "cpu_pct": psutil.cpu_percent(),
                    }
                )
                stop.wait(1)

        startup_seconds = time.perf_counter() - process_start
        run_start = time.perf_counter()
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            with (
                writer_type(args.out / "evidence", metadata) as writer,
                ImageHeadCapture(policy.model) as head,
                torch.inference_mode(),
            ):
                for frame in range(args.pilot_steps or 2400):
                    tick = time.perf_counter()
                    started = time.perf_counter()
                    # reset/step already made this tick's cached observation. Capture
                    # only if step was intentionally called without rendering.
                    obs = drive.observe(render=True)
                    raw_images = {
                        name: drive.cameras[name].get_rgba()[:, :, :3].astype(np.uint8).copy()
                        for name in env.cfg.cameras
                    }
                    render_s = time.perf_counter() - started
                    requested = obs.desire.copy()
                    applied = requested.copy()
                    if command_mode == "zero":
                        applied[:] = 0
                    elif command_mode == "swapped":
                        applied = applied[[0, 2, 1, 3, 4, 5, 6, 7]]
                    elif command_mode != "recorded":
                        raise ValueError(f"Unknown command mode: {command_mode}")
                    head.clear()
                    raw_output.clear()
                    started = time.perf_counter()
                    steer, accel, info = policy(replace(obs, desire=applied), env)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    policy_s = time.perf_counter() - started
                    if not np.isfinite(info["plan"]).all() or not np.isfinite([steer, accel]).all():
                        raise ValueError("Nonfinite model output")
                    p, r = drive.pose()
                    positions, quaternions = drive.cones.get_world_poses()
                    velocity = drive.robot.get_linear_velocity()
                    rng_before = json_sha256(env.rng.bit_generator.state)
                    record = {
                        "frame": frame,
                        "images": obs.images,
                        "raw_images": raw_images,
                        "prediction": info,
                        "policy_raw": raw_output["output"],
                        "image_head": head.output,
                        "state": asdict(obs.state),
                        "body": {
                            "position": p,
                            "rotation": r,
                            "velocity_world": velocity,
                            "velocity_body": r.T @ velocity,
                            "angular_velocity_world": drive.robot.get_angular_velocity(),
                        },
                        "progress": {"s": obs.s, "hint": env.hint, "lat": obs.lat},
                        "T_plan_ego_camera": drive.projection,
                        "T_world_camera_body": drive.world_camera,
                        "nominal_detection_calibration": env.T_ego_cam(),
                        "actual_cones_xyz": positions,
                        "actual_cone_quaternions": quaternions,
                        "action": [float(steer), float(accel)],
                        "commands": {"requested": requested, "applied": applied, "mode": command_mode},
                        "counters": {
                            "laps": len(env.referee.lap_times()),
                            "total_laps": env.track.laps,
                            "legacy_doo": env.referee.n_doo,
                            "physical_contacts": len(drive.hits),
                            "off_course": env.referee.n_oc,
                            "result": env.referee.result,
                        },
                        "events": [asdict(event) for event in obs.events],
                        "diagnostics": {"policy_with_capture_ms": policy_s * 1000, "render_ms": render_s * 1000},
                        "temporal": {"older_image_frame": max(0, frame - 4), "current_image_frame": frame},
                    }
                    started = time.perf_counter()
                    writer.append(record)
                    evidence_s = time.perf_counter() - started
                    if json_sha256(env.rng.bit_generator.state) != rng_before:
                        raise RuntimeError("Evidence writing changed scene RNG")
                    row = {
                        "frame": frame,
                        "t": obs.t,
                        "x": obs.state.x,
                        "y": obs.state.y,
                        "yaw": obs.state.yaw,
                        "speed": obs.state.vx,
                        "s": obs.s,
                        "lat": obs.lat,
                        "render_s": render_s,
                        "policy_with_capture_s": policy_s,
                        "evidence_s": evidence_s,
                        "legacy_doo": env.referee.n_doo,
                        "physical_contacts": len(drive.hits),
                    }
                    started = time.perf_counter()
                    terminal_obs = drive.step(steer, accel, render=False)
                    row["physics_s"] = time.perf_counter() - started
                    row["step_wall_s"] = time.perf_counter() - tick
                    rows.append(row)
                    if frame % 100 == 0:
                        print("STEP", case.id, frame, round(env.s, 2), len(drive.hits), flush=True)
                    if env.referee.done:
                        break
            evidence_seconds = writer.write_seconds
        finally:
            stop.set()
            thread.join()
            base.policy_raw = original_policy_raw
            pynvml.nvmlShutdown()
        elapsed = time.perf_counter() - run_start
        after_sha = model_digest(policy.model)
        head_after_sha = model_digest(policy.head)
        final_checkpoint_sha = digest(args.checkpoint)
        summary = drive.summary()
        summary.update(
            case=asdict(case),
            pilot_only=bool(args.pilot_steps),
            events=[asdict(e) for e in env.referee.events],
            pending_events=[asdict(e) for e in terminal_obs.events],
            recorded_frames=len(rows),
            rms_lat=float(np.sqrt(np.mean([row["lat"] ** 2 for row in rows]))),
            wall_seconds=elapsed,
            startup_seconds=startup_seconds,
            evidence_seconds=evidence_seconds,
            seconds_simulated=env.state.t,
            real_time_factor=env.state.t / elapsed,
            command_contract=policy.command_contract,
            controller_contract={"mode": policy.controller_name, "command_contract": policy.command_contract},
            controller_name=policy.controller_name,
            policy_device=str(policy.device),
            head_state_sha256=head_sha,
            head_state_sha256_after=head_after_sha,
            base_parameter_count=base_parameter_count,
            head_parameter_count=head_parameter_count,
            total_parameter_count=base_parameter_count + head_parameter_count,
            checkpoint_sha256=checkpoint_sha,
            checkpoint_sha256_after=final_checkpoint_sha,
            state_sha256=state_sha,
            state_sha256_after=after_sha,
            frozen_unchanged=(checkpoint_sha == final_checkpoint_sha and state_sha == after_sha and head_sha == head_after_sha),
            render_profile=PROFILE,
            physics_sha256=drive.appearance["physics_sha256"],
            torch_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
            torch_peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else None,
            runtime={"python": platform.python_version(), "torch": torch.__version__},
        )
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (args.out / "frames.json").write_text(json.dumps(rows) + "\n")
        (args.out / "telemetry.json").write_text(json.dumps(telemetry, indent=2) + "\n")
        finalize_evidence(args.out / "evidence", args.out / "summary.json")
        if not summary["frozen_unchanged"]:
            raise RuntimeError("Checkpoint or in-memory model state changed during evaluation")
        print("RESULT", json.dumps(summary), flush=True)
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        shutdown(app, drive)


if __name__ == "__main__":
    main()
