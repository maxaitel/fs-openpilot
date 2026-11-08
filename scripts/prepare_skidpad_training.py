import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zarr

from fsop.data.sim_dataset import load_index, split_episodes
from fsop.sim.batch_record import EpisodeSpec, record_dataset
from fsop.sim.expert import ExpertParams, Mission
from fsop.sim.route_commands import RouteCommands
from fsop.track.fixed_layouts import skidpad


def junction_coverage(states, commands, fps=5):
    track = skidpad()
    schedule = RouteCommands.for_mission(track, Mission(track, ExpertParams()), 20)
    speed = np.maximum(states[:, 4], 0)
    lead = np.maximum(schedule.settings.minimum_lead_metres, schedule.settings.lead_seconds * speed)
    counts = []
    for at_s, channel in schedule.decisions:
        window = (states[:, 9] >= at_s - lead) & (
            states[:, 9] <= at_s + schedule.settings.release_metres + speed / fps
        )
        counts.append(int(np.count_nonzero(commands[window, channel])))
    return counts


def audit_episode(dataset, spec):
    episode = dataset / spec.episode_id
    meta = json.loads((episode / "meta.json").read_text())
    plans = np.load(episode / "plan.npy")
    commands = np.load(episode / "desire.npy")
    states = np.load(episode / "states.npy")
    assert np.isfinite(plans).all() and np.isfinite(commands).all(), spec.episode_id
    assert meta["route_commands"]["schema"] == "next_junction_pulse_v1"
    frames = zarr.open_group(str(episode / "frames.zarr"), mode="r")
    for camera in ("road", "wide"):
        array = frames[camera][:]
        assert array.shape == (len(plans), 6, 128, 256), spec.episode_id
        assert np.all(array.reshape(len(array), -1).any(axis=1)), spec.episode_id
    coverage = junction_coverage(states, commands, meta["fps"])
    accepted = bool(meta["summary"]["pass"] and all(coverage))
    return {"id": spec.episode_id, "noise": spec.noise, "command_counts": commands.sum(0).tolist(),
            "junction_pulse_counts": coverage, "summary": meta["summary"], "accepted": accepted}


def main():
    started = time.time()
    output = Path("runs/spark_skidpad")
    output.mkdir(parents=True, exist_ok=True)
    dataset = Path("data/sim/spark_skidpad_intent")
    source = Path("data/sim/spark_full_venue")
    rng = np.random.default_rng(80000)
    specs = [
        EpisodeSpec(
            episode_id=f"intent_{seed:06d}_skidpad",
            track_path="skidpad",
            seed=seed,
            dr=str(rng.choice(["light", "heavy"], p=[0.6, 0.4])),
            random_start=False,
            max_seconds=60.0,
            noise=seed % 3 != 0,
            offset_max=0.6,
            venue_prob=1.0,
        )
        for seed in range(80000, 80320)
    ]
    record_dataset(specs, str(dataset), workers=16, max_tasks_per_child=20)
    rows = load_index(dataset)
    index = {r["id"]: r for r in load_index(source)}
    replay_ids = []
    for split, count in [("train", 256), ("val", 32)]:
        eligible = [eid for eid in split_episodes(source, split) if index[eid]["discipline"] == "trackdrive"]
        selected = sorted(rng.choice(eligible, min(count, len(eligible)), replace=False).tolist())
        for eid in selected:
            destination = dataset / eid
            if not destination.exists():
                destination.symlink_to((source / eid).resolve(), target_is_directory=True)
            rows.append(index[eid])
        replay_ids.extend(selected)
    rows = list({r["id"]: r for r in rows}.values())
    assert not set(replay_ids) & set(split_episodes(source, "test"))
    with ThreadPoolExecutor(4) as pool:
        demonstrations = list(pool.map(lambda spec: audit_episode(dataset, spec), specs))
    (output / "demonstration-audit.json").write_text(json.dumps(demonstrations, indent=2))
    accepted = {row["id"] for row in demonstrations if row["accepted"]}
    assert len(accepted) >= 200, "Too few valid demonstrations; inspect the audit before training"
    rows = [row for row in rows if row["id"] in accepted or row["id"] in replay_ids]
    (dataset / "index.json").write_text(json.dumps({"episodes": rows}, indent=2))
    splits = {split: split_episodes(dataset, split) for split in ("train", "val", "test")}
    report = {
        "seconds": time.time() - started,
        "dataset": str(dataset),
        "episodes": len(rows),
        "frames": sum(r["frames"] for r in rows),
        "skidpad_episodes": len(accepted),
        "skidpad_recorded": len(specs),
        "skidpad_rejected": len(specs) - len(accepted),
        "replay_episodes": len(replay_ids),
        "splits": splits,
        "live_validation_seeds": [61000, 61001, 61002],
        "live_test_seeds": [71000, 71001, 71002, 71003, 71004],
        "video_seed": 71000,
        "scene_levels": ["light", "heavy"],
        "expert_passes": sum(row["summary"]["pass"] for row in demonstrations),
        "noise_free_expert_passes": sum(row["accepted"] and not row["noise"] for row in demonstrations),
        "clean_expert_passes": sum(row["accepted"] and row["summary"]["doo"] == 0
                                   and row["summary"]["oc"] == 0 for row in demonstrations),
        "all_new_frames_read": True,
    }
    (output / "data.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "splits"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
