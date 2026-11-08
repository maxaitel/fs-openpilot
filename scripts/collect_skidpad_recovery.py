import argparse
import hashlib
import json
import math
import multiprocessing as mp
import tempfile
import time
import zlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class GuardSettings:
    threshold: float = 0.6
    heading_threshold: float = 0.20
    release_lat: float = 0.3
    release_heading: float = 0.12
    minimum_teacher_seconds: float = 1.0


@dataclass
class RecoveryGuard:
    settings: GuardSettings
    teacher_since: float | None = None
    interventions: int = 0

    def teacher(self, t: float, lat: float, heading_error: float) -> bool:
        cfg = self.settings
        if self.teacher_since is None:
            if abs(lat) > cfg.threshold or abs(heading_error) > cfg.heading_threshold:
                self.teacher_since = t
                self.interventions += 1
        elif (t - self.teacher_since >= cfg.minimum_teacher_seconds and abs(lat) < cfg.release_lat
              and abs(heading_error) < cfg.release_heading):
            self.teacher_since = None
        return self.teacher_since is not None


def episode_specs(start_seed: int, count: int) -> list[tuple[str, int, str]]:
    specs = []
    seed = start_seed
    while len(specs) < count:
        episode_id = f"recovery_{seed:06d}_skidpad"
        if not (61000 <= seed <= 61002 or 71000 <= seed <= 71004) and zlib.crc32(episode_id.encode()) % 1000 >= 200:
            specs.append((episode_id, seed, "light" if len(specs) % 2 == 0 else "heavy"))
        seed += 1
    return specs


def heading_error(mission, s: float, yaw: float) -> float:
    a, b = mission.point_at(s - 0.3), mission.point_at(s + 0.3)
    angle = yaw - math.atan2(b[1] - a[1], b[0] - a[0])
    return math.atan2(math.sin(angle), math.cos(angle))


def route_stage(env, s: float) -> str:
    segments = env.track.generator["segments"]
    for name in ("exit", "left", "right"):
        if s >= env.mission.s_arr[segments[name][0]]:
            return name
    return "entry"


_POLICY = None
_RENDERER = None


def initialize_worker(checkpoint: str) -> None:
    global _POLICY, _RENDERER
    import torch

    from fsop.eval.closed_loop import ModelPolicy
    from fsop.sim.env import CAMERAS
    from fsop.sim.render.panda3d_backend import Panda3DRenderer

    torch.set_num_threads(2)
    _POLICY = ModelPolicy.from_checkpoint(checkpoint, controller="pp")
    _POLICY.bev = False
    _RENDERER = Panda3DRenderer(CAMERAS)


def collect_one(args) -> dict:
    import numpy as np

    from fsop.eval.closed_loop import mission_brake
    from fsop.sim.env import FSEnv, SimCfg
    from fsop.sim.recorder import EpisodeWriter
    from fsop.track.fixed_layouts import skidpad

    spec, output, settings = args
    started = time.perf_counter()
    episode_id, seed, dr = spec
    cfg = SimCfg(dr=dr, venue_prob=1.0, max_time=90.0)
    cfg.noise.enabled = False
    env = FSEnv(skidpad(), cfg, seed=seed, renderer=_RENDERER)
    _POLICY.reset()
    guard = RecoveryGuard(GuardSettings(**settings["guard"]))
    stage_names = ["entry", "right", "left", "exit"]
    columns = ["t", "teacher", "learner_steer", "learner_accel", "expert_steer", "expert_accel",
               "applied_steer", "applied_accel", "lat", "heading_error", "s", "route_stage"]
    columns += [f"requested_command_{i}" for i in range(8)]
    controls, events = [], []
    stage_ticks = dict.fromkeys(stage_names, 0)
    command_counts = [0] * 8
    desire_acc = np.zeros(8, np.float32)
    with tempfile.TemporaryDirectory(prefix=f".{episode_id}-", dir=output) as temp:
        writer = EpisodeWriter(temp, env, episode_id, max_frames=int(90 * cfg.fps / 4) + 2, record_every=4)
        obs = env.observe()
        while not env.done:
            learner_steer, learner_accel, _ = _POLICY(obs, env)
            expert_steer, expert_accel = env.expert_action()
            expert_accel = mission_brake(expert_accel, obs, env)
            heading = obs.state.yaw
            if settings.get("heading_reference") == "course":
                heading += math.atan2(obs.state.vy, max(obs.state.vx, 0.5))
            error = heading_error(env.mission, obs.s, heading)
            teacher = guard.teacher(obs.t, obs.lat, error)
            steer, accel = (expert_steer, expert_accel) if teacher else (learner_steer, learner_accel)
            stage = route_stage(env, obs.s)
            stage_ticks[stage] += 1
            command_counts = [a + int(b > 0) for a, b in zip(command_counts, obs.desire, strict=True)]
            controls.append([obs.t, int(teacher), learner_steer, learner_accel, expert_steer, expert_accel,
                             steer, accel, obs.lat, error, obs.s, stage_names.index(stage), *obs.desire])
            events.extend({"t": e.t, "kind": e.kind, **e.data} for e in obs.events)
            desire_acc = np.maximum(desire_acc, obs.desire)
            if obs.frame % 4 == 0:
                writer.add(replace(obs, desire=desire_acc), env.label())
                desire_acc = np.zeros(8, np.float32)
            obs = env.step(steer, accel, add_noise=False)
        events.extend({"t": e.t, "kind": e.kind, **e.data} for e in obs.events)
        writer.events = events
        summary = env.referee.summary()
        meta = writer.close(summary)
        np.save(writer.dir / "controls.npy", np.asarray(controls, dtype=np.float64))
        meta["collection"] = {
            **settings, "assisted": True, "control_columns": columns, "control_fps": cfg.fps,
            "route_stage_values": stage_names, "route_stage_ticks": stage_ticks,
            "requested_command_counts": command_counts, "interventions": guard.interventions,
            "model_controlled_fraction": sum(not row[1] for row in controls) / max(len(controls), 1),
            "termination": summary["result"], "dr": dr,
            "wall_seconds": time.perf_counter() - started,
        }
        (writer.dir / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
        writer.dir.rename(Path(output) / episode_id)
    return {"id": episode_id, "track_id": meta["track_id"], "discipline": meta["discipline"],
            "frames": meta["frames"], "dr": dr, "result": summary["result"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect teacher-assisted skidpad recovery examples")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--start-seed", type=int, default=85000)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--heading-threshold", type=float, default=0.20)
    parser.add_argument("--release-lat", type=float, default=0.3)
    parser.add_argument("--release-heading", type=float, default=0.12)
    parser.add_argument("--minimum-teacher-seconds", type=float, default=1.0)
    parser.add_argument("--course-heading", action="store_true")
    args = parser.parse_args()
    guard = GuardSettings(args.threshold, args.heading_threshold, args.release_lat, args.release_heading,
                          args.minimum_teacher_seconds)
    if (args.count < 1 or args.workers < 1 or args.start_seed < 0 or not 0 < guard.release_lat < guard.threshold
            or not 0 < guard.release_heading < guard.heading_threshold or guard.minimum_teacher_seconds < 0
            or not all(math.isfinite(v) for v in asdict(guard).values())):
        parser.error("invalid count, workers, seed, or guard settings")
    checkpoint = Path(args.checkpoint).resolve()
    with checkpoint.open("rb") as f:
        checkpoint_sha256 = hashlib.file_digest(f, "sha256").hexdigest()
    settings = {"checkpoint_sha256": checkpoint_sha256, "guard": asdict(guard),
                "controller": "pp", "max_time": 90.0, "venue_prob": 1.0, "noise": False}
    if args.course_heading:
        settings["heading_reference"] = "course"
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    specs = episode_specs(args.start_seed, args.count)
    index_path = output / "index.json"
    if index_path.exists():
        existing_ids = {row["id"] for row in json.loads(index_path.read_text())["episodes"]}
        if existing_ids - {spec[0] for spec in specs}:
            raise ValueError("existing index contains episodes outside the requested set")
    rows, todo = [], []
    for spec in specs:
        episode_id, seed, dr = spec
        path = output / episode_id
        if path.exists():
            meta_file = path / "meta.json"
            if not meta_file.exists():
                raise ValueError(f"incomplete episode at {path}; inspect before retrying")
            meta = json.loads(meta_file.read_text())
            collection = meta.get("collection", {})
            if (any(collection.get(k) != v for k, v in settings.items())
                    or collection.get("heading_reference", "body") != settings.get("heading_reference", "body")
                    or meta.get("seed") != seed
                    or meta.get("id") != episode_id or meta.get("scene") is None or not (path / "controls.npy").exists()
                    or collection.get("dr") != dr):
                raise ValueError(f"episode settings or checkpoint mismatch at {path}")
            rows.append({"id": episode_id, "track_id": meta["track_id"], "discipline": meta["discipline"],
                         "frames": meta["frames"], "dr": dr, "result": meta["summary"]["result"]})
        else:
            todo.append(spec)
    if args.count > 4 and any(spec in todo for spec in specs[:4]):
        raise ValueError("run --count 4 and inspect the assisted calibration episodes before collecting the full set")
    print(json.dumps({"assisted": True, "settings": settings, "requested": args.count,
                      "completed": len(rows), "remaining": len(todo)}), flush=True)
    if todo:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=initialize_worker, initargs=(str(checkpoint),),
                      maxtasksperchild=8) as pool:
            for row in pool.imap_unordered(collect_one, [(spec, str(output), settings) for spec in todo]):
                rows.append(row)
                print(json.dumps({"assisted_episode": row}), flush=True)
    order = {spec[0]: i for i, spec in enumerate(specs)}
    rows.sort(key=lambda row: order[row["id"]])
    (output / "index.json").write_text(json.dumps({"episodes": rows}, indent=1))


if __name__ == "__main__":
    main()
