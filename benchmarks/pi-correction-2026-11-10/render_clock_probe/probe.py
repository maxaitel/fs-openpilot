import argparse
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("sync", "async"), required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
protocol_path = Path(__file__).with_name("protocol.json")
protocol = json.loads(protocol_path.read_text())
args.out.mkdir(exist_ok=False, parents=True)
free = shutil.disk_usage(args.out).free
assert free >= protocol["reserve_bytes"], free

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(name, value):
    (args.out / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

inventory = {str(p): sha(p) for root in ("src/fsop", "isaac_eval", "isaac_lab_fsop") for p in sorted(Path(root).rglob("*.py"))}
save("preflight.json", {"pid": os.getpid(), "free_bytes": free, "protocol_sha256": sha(protocol_path), "probe_sha256": sha(__file__), "source_sha256": inventory})

from isaac_lab_fsop.runtime import launch, shutdown
launcher = launch(render=True)
drive = None
try:
    import carb
    import numpy as np
    import omni.timeline
    import torch
    from isaac_lab_fsop.cases import Case, environment
    from isaac_lab_fsop.env import LabDrive
    from isaac_lab_fsop.evidence import AsyncEvidenceWriter
    from isaac_eval.evidence import EvidenceWriter, finalize_evidence
    from isaac_eval.rendering import PROFILE
    from rendering import capture

    torch.set_num_threads(4)
    case = Case(**json.loads(Path(protocol["case_path"]).read_text()))
    env = environment(case)
    drive = LabDrive(env, variant=case.variant, render=True)
    x, y, yaw = env.track.start_pose
    start_pose = (x-case.lateral_start*np.sin(yaw), y+case.lateral_start*np.cos(yaw), yaw+case.yaw_start)
    drive.reset(start_pose=start_pose)
    timeline = omni.timeline.get_timeline_interface()

    def snapshot():
        view = drive.robot.asset.root_physx_view
        return {"wall_monotonic": time.monotonic(), "sim_time": float(drive.sim.current_time),
                "step_index": int(drive.sim.current_time_step_index), "timeline_time": float(timeline.get_current_time()),
                "timeline_playing": timeline.is_playing(), "timeline_auto_update": timeline.is_auto_updating(),
                "physx_transform_xyzw": view.get_transforms().detach().cpu().numpy().tolist(),
                "physx_velocity": view.get_velocities().detach().cpu().numpy().tolist()}

    def images():
        return {name: drive.cameras[name].get_rgba()[:, :, :3].astype(np.uint8).copy() for name in env.cfg.cameras}

    def hashes(raw):
        return {name: {"shape": list(value.shape), "sha256": hashlib.sha256(value.tobytes()).hexdigest()} for name, value in raw.items()}

    def settings_snapshot():
        settings = carb.settings.get_settings()
        paths = set(drive.sim.fsop_render_settings["base"]) | {
            "/rtx/rendermode", "/rtx/post/aa/op", "/rtx/post/dlss/execMode", "/rtx-transient/dldenoiser/enabled",
            "/rtx/post/motionblur/enabled", "/app/player/playSimulations", "/physics/disableContactProcessing"}
        products = {}
        for prim in drive.stage.Traverse():
            if str(prim.GetPath()).startswith("/Render/"):
                products[str(prim.GetPath())] = {"type": str(prim.GetTypeName()),
                    "attributes": {str(attr.GetName()): str(attr.Get()) for attr in prim.GetAttributes()},
                    "relationships": {str(rel.GetName()): [str(p) for p in rel.GetTargets()] for rel in prim.GetRelationships()}}
        cameras = {}
        for name, camera in drive.cameras.items():
            cameras[name] = {"render_product": str(camera.get_render_product_path()), "resolution": list(camera.get_resolution())}
        return {"global": {p: settings.get(p) for p in sorted(paths)}, "render_prims": products,
                "cameras": cameras, "profile": PROFILE, "sim_render_settings": drive.sim.fsop_render_settings}

    initial = images()
    np.savez_compressed(args.out / "initial_images.npz", **initial)
    metadata = {"pilot_only": True, "mode": args.mode, "case": asdict(case), "case_sha256": sha(protocol["case_path"]),
                "no_model": True, "plan_is_serialization_fixture": True, "protocol": protocol, "appearance": drive.appearance,
                "initial": snapshot(), "initial_images": hashes(initial), "settings": settings_snapshot()}
    save("metadata.json", metadata)
    writer_type = EvidenceWriter if args.mode == "sync" else AsyncEvidenceWriter
    writer = writer_type(args.out / "evidence", {"pilot_only": True, "no_model": True, "plan_is_serialization_fixture": True})
    rows = []
    with (args.out / "trace.jsonl").open("x") as trace:
        for tick in range(protocol["ticks"]):
            row = {"tick": tick, "before_capture": snapshot()}
            capture(drive.sim)
            raw = images()
            row["after_capture"] = snapshot()
            row["raw_images"] = hashes(raw)
            row["camera_frames"] = {name: {key: value for key, value in camera.get_current_frame().items() if key in ("rendering_frame", "rendering_time")} for name, camera in drive.cameras.items()}
            row["before_append"] = snapshot()
            writer.append({"images": raw, "prediction": {"plan": np.zeros((2, 2), dtype=np.float32)},
                           "action": protocol["action"], "fixture": True, "tick": tick})
            row["after_append"] = snapshot()
            row["before_idle"] = snapshot()
            time.sleep(protocol["idle_seconds"])
            row["after_idle"] = snapshot()
            row["before_advance"] = snapshot()
            drive.step(*protocol["action"], render=False)
            row["after_advance"] = snapshot()
            trace.write(json.dumps(row) + "\n")
            trace.flush()
            rows.append(row)
        closing = {"before": snapshot()}
        writer.close()
        closing["after"] = snapshot()
    save("settings_final.json", settings_snapshot())
    save("summary.json", {"ticks": len(rows), "writer_count": writer.count, "writer_close": closing, "result": "completed_fixed_action_probe"})
    finalize_evidence(args.out / "evidence", args.out / "summary.json")
    inventory_after = {p: sha(p) for p in inventory}
    assert inventory_after == inventory
    size = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    save("complete.json", {"ticks": len(rows), "writer_count": writer.count, "source_unchanged": True,
                           "artifact_bytes": size, "free_bytes": shutil.disk_usage(args.out).free,
                           "trace_sha256": sha(args.out / "trace.jsonl"), "completed_before_shutdown": True})
finally:
    shutdown(launcher.app, drive)
