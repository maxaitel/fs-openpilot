import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from fsop.data.sim_dataset import Episode, split_episodes
from fsop.geometry.op_frames import ego_to_op_plan
from fsop.model.build import load_fs
from fsop.sim.expert import T_IDXS


def command_probe(checkpoint, dataset, output, split="test"):
    model = load_fs(checkpoint).cuda().eval()
    rows = []
    for eid in [e for e in split_episodes(dataset, split) if "skidpad" in e]:
        episode = Episode(Path(dataset) / eid)
        hidden = []
        with torch.no_grad():
            for start in range(0, episode.T, 32):
                indices = range(start, min(episode.T, start + 32))
                frames = {
                    cam: torch.from_numpy(np.stack([episode.frame_pair(cam, k) for k in indices])).cuda().float()
                    for cam in ("road", "wide")
                }
                hidden.append(model.base.encode(frames["road"], frames["wide"]).hidden)
            hidden = torch.cat(hidden)
            features = torch.zeros(episode.T, 25, 512, device="cuda")
            desire = torch.zeros(episode.T, 25, 8, device="cuda")
            for k in range(episode.T):
                count = min(k + 1, 25)
                features[k, -count:] = hidden[k - count + 1 : k + 1]
                desire[k, -count:] = torch.from_numpy(episode.desire[k - count + 1 : k + 1]).cuda()
            modes = {
                "recorded": desire,
                "zero": torch.zeros_like(desire),
                "swapped": desire[:, :, [0, 2, 1, 3, 4, 5, 6, 7]],
            }
            for channel in (1, 2, 7):
                value = torch.zeros_like(desire)
                value[:, -1, channel] = 1
                modes[f"latest_{channel}"] = value
            predictions = {}
            for name, commands in modes.items():
                batches = []
                for start in range(0, episode.T, 64):
                    count = min(64, episode.T - start)
                    plan, _ = model.base.policy_raw(
                        features[start : start + count],
                        commands[start : start + count],
                        torch.tensor([[1.0, 0.0]], device="cuda").expand(count, -1),
                    )
                    batches.append(plan.reshape(-1, 2, 33, 15)[:, 0].cpu().numpy())
                predictions[name] = np.concatenate(batches)
        truth = ego_to_op_plan(episode.plan)
        active = desire.cpu().numpy().sum((1, 2)) > 0
        metrics = {}
        for name, prediction in predictions.items():
            assert np.isfinite(prediction).all(), (eid, name)
            metrics[name] = {}
            for seconds in (0.5, 1, 2):
                index = int(np.argmin(np.abs(T_IDXS - seconds)))
                error = np.abs(prediction[:, index, 1] - truth[:, index, 1])
                metrics[name][f"lateral_{seconds}s_all"] = float(error.mean())
                metrics[name][f"lateral_{seconds}s_active"] = float(error[active].mean()) if active.any() else None
        rows.append({"episode": eid, "frames": episode.T, "active_frames": int(active.sum()), "metrics": metrics})
        np.savez_compressed(
            output / f"{eid}-commands.npz", **predictions, truth=truth, states=episode.states, desire=episode.desire
        )
    (output / "commands.json").write_text(json.dumps({"checkpoint": str(checkpoint), "episodes": rows}, indent=2))
    print(json.dumps(rows, indent=2), flush=True)


def driving_probe(checkpoint, output, seeds, levels, controller, command_mode):
    from fsop.eval.closed_loop import ModelPolicy
    from fsop.sim.env import CAMERAS, FSEnv, SimCfg
    from fsop.sim.render.panda3d_backend import Panda3DRenderer
    from fsop.track.fixed_layouts import skidpad

    policy = ModelPolicy.from_checkpoint(checkpoint, controller=controller)
    renderer = Panda3DRenderer(CAMERAS)
    rows = []
    for seed in seeds:
        for level in levels:
            config = SimCfg(dr=level, venue_prob=1.0, max_time=120)
            config.noise.enabled = False
            env = FSEnv(skidpad(), config, seed=seed, renderer=renderer)
            policy.reset()
            observation = env.observe()
            states, desires, route_desires, plans = [], [], [], []
            vehicle_states, actions = [], []
            while not env.done:
                requested = observation.desire.copy()
                command = requested.copy()
                if command_mode == "zero":
                    command[:] = 0
                elif command_mode == "swapped":
                    command = command[[0, 2, 1, 3, 4, 5, 6, 7]]
                steering, acceleration, info = policy(replace(observation, desire=command), env)
                state = observation.state
                states.append([observation.t, state.x, state.y, state.yaw, state.vx, observation.s, observation.lat])
                vehicle_states.append(state.as_array())
                actions.append([steering, acceleration])
                desires.append(command)
                route_desires.append(requested)
                plans.append(info["plan"])
                observation = env.step(steering, acceleration, add_noise=False)
            result = env.referee.summary()
            result.update(seed=seed, dr=level, controller=controller, command_mode=command_mode,
                          progress=float(env.s / env.mission.s_end))
            if result["result"] == "running":
                result["result"] = "timeout"
            prefix = f"{level}-{seed}"
            np.savez_compressed(output / f"{prefix}.npz", states=states, desire=desires,
                                route_desire=route_desires, plan=plans, vehicle_states=vehicle_states,
                                actions=actions, final_vehicle_state=observation.state.as_array())
            (output / f"{prefix}.json").write_text(json.dumps(result, indent=2))
            rows.append(result)
            print(json.dumps(result), flush=True)
    (output / "driving.json").write_text(json.dumps({"checkpoint": str(checkpoint), "runs": rows}, indent=2))


def command_parity(output):
    import onnxruntime as ort

    from fsop.model.convert import load_graph

    path = "artifacts/models/prep/policy_fp32.onnx"
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    session = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
    model, _ = load_graph(path)
    model.eval()
    rng = np.random.default_rng(10)
    features = rng.normal(size=(1, 25, 512)).astype(np.float32)
    features /= np.linalg.norm(features, axis=-1, keepdims=True)
    traffic = np.array([[1.0, 0.0]], np.float32)
    rows = []
    with torch.no_grad():
        for channel in (1, 2, 7):
            for slot in (0, 8, 16, 24):
                command = np.zeros((1, 25, 8), np.float32)
                command[0, slot, channel] = 1
                reference = session.run(None, {"fb_full": features, "dp_out": command, "tc_out": traffic})
                converted = model(torch.from_numpy(features), torch.from_numpy(command), torch.from_numpy(traffic))
                errors = [float(np.max(np.abs(a - b.numpy()))) for a, b in zip(reference, converted, strict=True)]
                assert np.isfinite(errors).all() and max(errors) < 0.001, errors
                rows.append({"channel": channel, "slot": slot, "plan_max_abs": errors[0], "desire_max_abs": errors[1]})
    (output / "parity.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("commands", "drive", "parity"))
    parser.add_argument("--checkpoint", default="runs/spark_full_refresh/ckpt/best.pt")
    parser.add_argument("--dataset", default="data/sim/spark_full_venue")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--levels", nargs="+", default=["light", "heavy"])
    parser.add_argument("--controller", choices=("pp", "op"), default="pp")
    parser.add_argument("--command-mode", choices=("recorded", "zero", "swapped"), default="recorded")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    if args.mode == "commands":
        command_probe(args.checkpoint, args.dataset, output, args.split)
    elif args.mode == "parity":
        command_parity(output)
    else:
        driving_probe(args.checkpoint, output, args.seeds, args.levels, args.controller, args.command_mode)


if __name__ == "__main__":
    main()
