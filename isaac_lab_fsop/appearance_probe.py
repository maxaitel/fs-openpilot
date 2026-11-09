"""Replay sealed evidence poses to measure direct visual sensitivity of a frozen policy.

Run with Isaac Lab's Python from the repository root. This is a render-only
counterfactual at one recorded state, not a closed-loop causal experiment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import traceback
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def capture_sources():
    """Snapshot executable source bytes once so the archive and hashes agree."""
    roots = ("src/fsop", "isaac_eval", "isaac_lab_fsop")
    paths = sorted(path for root in roots for path in Path(root).rglob("*.py"))
    return {str(path): path.read_bytes() for path in paths}


def source_hashes(sources):
    return {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()}


def capture_without_advance(sim, capture, baseline):
    before = float(sim.current_time)
    if not math.isfinite(before) or before != baseline:
        raise RuntimeError("Simulation time changed before render-only capture")
    capture(sim)
    after = float(sim.current_time)
    if not math.isfinite(after) or after != before:
        raise RuntimeError("Simulation time advanced during render-only capture")
    return before, after


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode", type=Path, required=True, help="Episode directory or its evidence directory")
    p.add_argument("--frame", type=int, required=True, help="Anchor frame, at least 100")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--settle-captures", type=int, default=1,
                   help="Diagnostic-only captures at each fixed pose before reading RGB (1 to 8)")
    args = p.parse_args()
    if args.frame < 100:
        p.error("--frame must be at least 100 for a complete 25-slot history")
    if not 1 <= args.settle_captures <= 8:
        p.error("--settle-captures must be between 1 and 8")
    return args


def load_records(directory, anchor):
    import numpy as np

    from isaac_eval.evidence import read_evidence, verified_summary

    directory = directory / "evidence" if (directory / "evidence").is_dir() else directory
    if not (directory / "completion.json").is_file() or verified_summary(directory) is None:
        raise ValueError("A sealed complete evidence episode is required")
    selected = set(range(anchor - 100, anchor + 1, 4))
    poses, commands, hashes = {}, [], []
    metadata = None
    for record_metadata, record, entry in read_evidence(directory):
        if metadata is None:
            metadata = record_metadata
        frame = entry["frame"]
        if record["frame"] != frame:
            raise ValueError("Recorded frame does not match manifest")
        commands.append(np.asarray(record["commands"]["applied"], dtype=np.float32).copy())
        hashes.append({"frame": frame, "sha256": entry["sha256"], "input_sha256": entry["input_sha256"]})
        if frame in selected:
            poses[frame] = {
                k: copy.deepcopy(record[k])
                for k in (
                    "body",
                    "state",
                    "progress",
                    "actual_cones_xyz",
                    "actual_cone_quaternions",
                    "T_world_camera_body",
                    "T_plan_ego_camera",
                    "counters",
                )
            }
        if frame == anchor:
            break
    if len(commands) != anchor + 1 or len(poses) != 26:
        raise ValueError("Evidence does not contain the complete requested history")
    if any(x.shape != (8,) or not np.isfinite(x).all() for x in commands):
        raise ValueError("Invalid recorded commands")
    return directory, metadata, poses, commands, hashes


def serial(value):
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value


def stability_analysis(results):
    """Predeclared repeat-noise guard; preserve unresolved and unstable results."""
    import numpy as np

    from fsop.sim.expert import T_IDXS

    limits = {"near_xy_m": 0.02, "steer_rad": 0.002, "accel_mps2": 0.03}

    def distance(left, right):
        a, b = results[left], results[right]
        xy = (np.asarray(a["plan"]) - np.asarray(b["plan"]))[T_IDXS <= 2, :2]
        action = np.abs(np.asarray(a["action"]) - np.asarray(b["action"]))
        return {"near_xy_m": float(np.linalg.norm(xy, axis=1).max()),
                "steer_rad": float(action[0]), "accel_mps2": float(action[1])}

    repeats = {name: distance(name + "_repeat", name) for name in ("matched", "full")}
    noise = {key: max(row[key] for row in repeats.values()) for key in limits}
    stable = all(noise[key] <= limits[key] for key in limits)
    contrasts = {}
    pairs = [(name, "matched") for name in ("weather_only", "marks_only", "props_only", "full", "no_props")]
    pairs.append(("full", "no_props"))
    for left, right in pairs:
        effect = distance(left, right)
        contrasts[left + "_minus_" + right] = {
            "magnitude": effect,
            "effect_above_noise": {key: stable and effect[key] > max(5 * noise[key], limits[key]) for key in limits},
        }
    return {"interpretable": stable, "repeat_limits": limits, "repeat_differences": repeats,
            "repeat_noise_max": noise, "contrast_multiplier": 5, "contrasts": contrasts,
            "limit": "Two repeats estimate a noise floor, not a statistical causal confidence interval."}


def main():
    args = parse_args()
    directory, metadata, poses, commands, hashes = load_records(args.episode, args.frame)
    case_id = metadata["case"].get("case_id") or metadata["case"].get("frozen_case")
    if not case_id:
        raise ValueError("Evidence must identify a frozen case")
    if metadata.get("variant") != "visual_stress_v1":
        raise ValueError("Evidence must be from a visual_stress_v1 episode")
    checkpoint_sha = digest(args.checkpoint)
    if checkpoint_sha != metadata["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the one used for this episode")
    args.out.mkdir(parents=True, exist_ok=False)
    source_start = capture_sources()
    source_start_hashes = source_hashes(source_start)
    (args.out / "source_start.json").write_text(json.dumps(source_start_hashes, indent=2) + "\n")
    with ZipFile(args.out / "source.zip", "x", compression=ZIP_DEFLATED) as archive:
        for name, data in source_start.items():
            archive.writestr(name, data)
    source_archive_sha = digest(args.out / "source.zip")
    from isaac_lab_fsop.runtime import launch, shutdown

    launcher = launch(render=True)
    app, drive = launcher.app, None
    try:
        import cv2
        import numpy as np
        import torch
        from appearance_guard import assert_appearance_only, physical_contract
        from parity_geometry import pose_model_cameras
        from pxr import UsdGeom
        from rendering import capture
        from scipy.spatial.transform import Rotation

        from fsop.eval.closed_loop import ModelPolicy, mission_brake
        from fsop.eval.controller import PlanPurePursuit
        from fsop.geometry.op_frames import op_to_ego_plan
        from fsop.geometry.yuv import rgb_to_model
        from fsop.model.temporal import TemporalState
        from fsop.sim.randomization import postprocess
        from isaac_eval.evidence import input_hash
        from isaac_eval.parity_cases import build_environment, get_case, json_sha256
        from isaac_lab_fsop.env import LabDrive
        from isaac_lab_fsop.evaluate import model_digest

        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        env = build_environment(get_case(case_id))
        if not np.allclose(env.T_ego_cam(), metadata["nominal_detection_calibration"], atol=1e-9, rtol=0):
            raise ValueError("Rebuilt camera calibration does not match recorded calibration")
        initial_rng = copy.deepcopy(env.rng.bit_generator.state)
        drive = LabDrive(env, variant="visual_stress_v1", render=True)
        drive.reset()
        replay_time = float(drive.sim.current_time)
        if not math.isfinite(replay_time):
            raise RuntimeError("Invalid simulation time after reset")
        policy = ModelPolicy.from_checkpoint(args.checkpoint, controller="pp")
        policy.model.requires_grad_(False)
        state_sha = model_digest(policy.model)
        if metadata.get("state_sha256") and state_sha != metadata["state_sha256"]:
            raise ValueError("Loaded model state differs from recorded model state")
        base = policy.model.base if hasattr(policy.model, "base") else policy.model
        physical_before = physical_contract(drive.stage)
        physical_sha = hashlib.sha256(json.dumps(physical_before, sort_keys=True).encode()).hexdigest()
        initial_physical_sha = drive.appearance["physics_sha256"]
        expected_physical = metadata.get("physics_sha256")
        # Backend-specific transform representations may differ; preserve both receipts.
        expected_assets = metadata.get("asset_sha256")
        if expected_assets and drive.appearance.get("asset_sha256") != expected_assets:
            raise ValueError("Stress asset hashes differ from recorded episode")
        expected_layout = metadata.get("layout_sha256")
        if expected_layout and drive.appearance["stress"]["layout_sha256"] != expected_layout:
            raise ValueError("Stress layout differs from recorded episode")
        branches = {
            "weather": "/World/VisualStress/Ground",
            "marks": "/World/VisualStress/Surface",
            "props": "/World/VisualStress/Objects",
        }
        prims = {key: drive.stage.GetPrimAtPath(path) for key, path in branches.items()}
        if any(not p for p in prims.values()):
            raise ValueError("Stress overlay branches are missing")
        # Child mesh authoring leaves two grouping ancestors untyped. A Scope
        # supplies inherited visibility without introducing a transform.
        for key, prim in prims.items():
            if not UsdGeom.Imageable(prim):
                prims[key] = UsdGeom.Scope.Define(drive.stage, prim.GetPath()).GetPrim()
        assert_appearance_only(physical_before, drive.stage)
        repairs = drive.stage.GetPrimAtPath("/World/Appearance/Repairs")
        patches = [p for p in repairs.GetChildren() if p.GetName().startswith("Patch")] if repairs else []
        if len(patches) != 14:
            raise ValueError("Expected the 14 matched-scene repair patches")
        if any(UsdGeom.Imageable(p).ComputeVisibility() != UsdGeom.Tokens.invisible for p in patches):
            raise ValueError("Stress scene did not hide the matched-scene repair patches")
        asphalt = drive.stage.GetPrimAtPath("/World/Appearance/Asphalt")
        if not asphalt or UsdGeom.Imageable(asphalt).ComputeVisibility() == UsdGeom.Tokens.invisible:
            raise ValueError("Matched-scene asphalt is missing or hidden")
        conditions = [
            ("matched", (0, 0, 0)),
            ("weather_only", (1, 0, 0)),
            ("marks_only", (0, 1, 0)),
            ("props_only", (0, 0, 1)),
            ("full", (1, 1, 1)),
            ("no_props", (1, 1, 0)),
            ("matched_repeat", (0, 0, 0)),
            ("full_repeat", (1, 1, 1)),
        ]
        ts = TemporalState()
        for command in commands:
            desire = ts.push_desire(command)
        history_frames = list(range(args.frame - 100, args.frame + 1, 4))
        anchor = poses[args.frame]
        obs = SimpleNamespace(state=SimpleNamespace(**anchor["state"]), s=float(anchor["progress"]["s"]))
        dummy = np.zeros((256, 512, 3), dtype=np.uint8)
        results = {}
        for name, flags in conditions:
            contract_before_toggle = physical_contract(drive.stage)
            for key, visible in zip(("weather", "marks", "props"), flags, strict=True):
                imageable = UsdGeom.Imageable(prims[key])
                imageable.MakeVisible() if visible else imageable.MakeInvisible()
            for patch in patches:
                imageable = UsdGeom.Imageable(patch)
                imageable.MakeInvisible() if flags[0] else imageable.MakeVisible()
            for key, visible in zip(("weather", "marks", "props"), flags, strict=True):
                actual = UsdGeom.Imageable(prims[key]).ComputeVisibility() != UsdGeom.Tokens.invisible
                if actual != bool(visible):
                    raise RuntimeError(f"Visibility mismatch for {key} in {name}")
            for patch in patches:
                actual = UsdGeom.Imageable(patch).ComputeVisibility() != UsdGeom.Tokens.invisible
                if actual != (not flags[0]):
                    raise RuntimeError(f"Repair patch visibility mismatch in {name}")
            assert_appearance_only(contract_before_toggle, drive.stage)
            env.rng.bit_generator.state = copy.deepcopy(initial_rng)
            packed, image_hashes, camera_receipts = [], [], []
            for frame in range(args.frame + 1):
                raw = {"road": dummy, "wide": dummy}
                if frame in poses:
                    record = poses[frame]
                    pos = np.asarray(record["body"]["position"], dtype=float)
                    rot = np.asarray(record["body"]["rotation"], dtype=float)
                    if pos.shape != (3,) or rot.shape != (3, 3) or not np.isfinite(rot).all():
                        raise ValueError(f"Invalid body pose at {frame}")
                    q = Rotation.from_matrix(rot).as_quat()
                    drive.robot.set_world_pose(pos, np.r_[q[3], q[:3]])
                    cone_pos = np.asarray(record["actual_cones_xyz"])
                    cone_q = np.asarray(record["actual_cone_quaternions"])
                    cone_pose = torch.as_tensor(
                        np.concatenate([cone_pos, cone_q], axis=1), dtype=torch.float32, device=drive.cones.asset.device
                    )
                    drive.cones.asset.write_root_pose_to_sim(cone_pose)
                    yaw = float(record["state"]["yaw"])
                    from fsop.geometry.camera import se3

                    planar = se3(Rotation.from_euler("z", yaw).as_matrix(), [pos[0], pos[1], 0])
                    world_camera, projection = pose_model_cameras(drive.cameras, pos, rot, env.T_ego_cam(), planar)
                    if not np.allclose(world_camera, record["T_world_camera_body"], atol=1e-6, rtol=0):
                        raise ValueError(f"Rebuilt camera pose differs from evidence at {frame}")
                    if not np.allclose(projection, record["T_plan_ego_camera"], atol=1e-6, rtol=0):
                        raise ValueError(f"Rebuilt camera projection differs from evidence at {frame}")
                    capture_times = [capture_without_advance(drive.sim, capture, replay_time)
                                     for _ in range(args.settle_captures)]
                    time_before, time_after = capture_times[0][0], capture_times[-1][1]
                    raw = {
                        cam: drive.cameras[cam].get_rgba()[:, :, :3].astype(np.uint8).copy() for cam in ("road", "wide")
                    }
                    camera_receipts.append(
                        {
                            "frame": frame,
                            "world_camera": world_camera.tolist(),
                            "projection": projection.tolist(),
                            "sim_time_before": time_before,
                            "sim_time_after": time_after,
                            "capture_times": capture_times,
                        }
                    )
                # The random draws depend on camera dimensions, not image pixels or yaw rate.
                processed = {
                    cam: postprocess(
                        raw[cam], env.scene, env.rng, float(poses[frame]["state"]["r"]) if frame in poses else 0.0
                    )
                    for cam in ("road", "wide")
                }
                if frame in poses:
                    image_hashes.append({"frame": frame, "raw": input_hash(raw), "processed": input_hash(processed)})
                    packed.append({cam: rgb_to_model(processed[cam]) for cam in ("road", "wide")})
                    if frame == args.frame:
                        for cam in ("road", "wide"):
                            for kind, images in (("raw", raw), ("processed", processed)):
                                path = args.out / f"{name}_{cam}_{kind}.png"
                                if not cv2.imwrite(str(path), cv2.cvtColor(images[cam], cv2.COLOR_RGB2BGR)):
                                    raise OSError(f"Failed to write {path}")
            hidden = []
            with torch.inference_mode():
                for i in range(1, len(packed)):
                    pair = {
                        cam: np.concatenate((packed[i - 1][cam], packed[i][cam]), axis=0) for cam in ("road", "wide")
                    }
                    road = torch.from_numpy(pair["road"][None]).to(policy.device).float()
                    wide = torch.from_numpy(pair["wide"][None]).to(policy.device).float()
                    hidden.append(base.encode(road, wide).hidden)
                hist = torch.stack(hidden, dim=1)
                des = torch.from_numpy(desire[None]).to(policy.device)
                raw_output = base.policy_raw(hist, des, torch.tensor([[1.0, 0.0]], device=policy.device))
                plan_raw = raw_output[0].reshape(1, 2, 33, 15)
                plan = op_to_ego_plan(plan_raw[0, 0].cpu().numpy())
                steer, accel = PlanPurePursuit(env.vp)(plan, obs.state.vx)
                accel = mission_brake(accel, obs, env)
                result = {
                    "raw_model_outputs": [v.detach().cpu().numpy() for v in raw_output],
                    "plan": plan,
                    "plan_std": np.exp(plan_raw[0, 1].clamp(-7, 7).cpu().numpy()),
                    "desire_history": desire,
                    "action": [steer, accel],
                    "image_hashes": image_hashes,
                    "cameras": camera_receipts,
                    "rng_end_sha256": json_sha256(env.rng.bit_generator.state),
                }
            result["anchor_png_sha256"] = {
                f"{cam}_{kind}": digest(args.out / f"{name}_{cam}_{kind}.png")
                for cam in ("road", "wide")
                for kind in ("raw", "processed")
            }
            results[name] = serial(result)
            (args.out / f"{name}.json").write_text(json.dumps(results[name], indent=2, allow_nan=False) + "\n")
        comparisons = {}
        for name, _ in conditions[1:]:
            reference = "full" if name == "full_repeat" else "matched"
            delta = np.abs(np.asarray(results[name]["plan"]) - np.asarray(results[reference]["plan"]))
            actions = np.asarray(results[name]["action"]) - np.asarray(results[reference]["action"])
            comparisons[name] = {
                "reference": reference,
                "plan_max_abs": float(delta.max()),
                "plan_mean_abs": float(delta.mean()),
                "plan_xy_max_abs_m": float(delta[:, :2].max()),
                "plan_xy_mean_abs_m": float(delta[:, :2].mean()),
                "steer_delta": float(actions[0]),
                "accel_delta": float(actions[1]),
            }
        baseline_cameras = results["matched"]["cameras"]
        if any(results[n]["cameras"] != baseline_cameras for n, _ in conditions):
            raise RuntimeError("Camera transforms diverged between conditions")
        if model_digest(policy.model) != state_sha or digest(args.checkpoint) != checkpoint_sha:
            raise RuntimeError("Frozen model or checkpoint changed")
        if len({results[n]["rng_end_sha256"] for n, _ in conditions}) != 1:
            raise RuntimeError("Postprocessing RNG streams diverged")
        source_end_hashes = source_hashes(capture_sources())
        (args.out / "source_end.json").write_text(json.dumps(source_end_hashes, indent=2) + "\n")
        if source_start_hashes != source_end_hashes:
            raise RuntimeError("Executable source changed during the probe")
        receipt = {
            "interpretation": "Direct fixed-pose visual sensitivity only; not closed-loop causal proof.",
            "case_id": case_id,
            "anchor_frame": args.frame,
            "source_evidence": str(directory.resolve()),
            "source_metadata_sha256": digest(directory / "metadata.json"),
            "source_manifest_sha256": digest(directory / "manifest.jsonl"),
            "source_records": hashes,
            "checkpoint_sha256": checkpoint_sha,
            "model_state_sha256": state_sha,
            "physics_sha256_initial": initial_physical_sha,
            "physics_sha256_replay": physical_sha,
            "source_physics_sha256": expected_physical,
            "appearance": drive.appearance,
            "calibration": env.T_ego_cam(),
            "poses": poses,
            "history_frames": history_frames,
            "applied_commands": commands,
            "desire_history": desire,
            "conditions": {n: list(f) for n, f in conditions},
            "comparisons": comparisons,
            "stability": stability_analysis(results),
            "source_sha256_start": source_start_hashes,
            "source_sha256_end": source_end_hashes,
            "source_archive_sha256": source_archive_sha,
            "simulation_time_after_reset": replay_time,
            "capture_calls_per_pose": args.settle_captures,
            "capture_scope": "Diagnostic settling only; scored evaluation keeps the unchanged 16-subframe profile.",
            "no_simulation_advance_after_reset": all(
                camera["sim_time_before"] == replay_time == camera["sim_time_after"]
                for result in results.values()
                for camera in result["cameras"]
            ),
            "temporal_semantics": "25 encoded pairs at anchor-96..anchor; previous images at anchor-100..anchor-4",
            "rng_semantics": "same initial state and road/wide draws for every tick, including skipped frames",
        }
        (args.out / "comparison.json").write_text(json.dumps(serial(receipt), indent=2, allow_nan=False) + "\n")
    except BaseException:
        # SimulationApp shutdown may obscure the exception and its original traceback.
        traceback.print_exc()
        raise
    finally:
        shutdown(app, drive)


if __name__ == "__main__":
    main()
