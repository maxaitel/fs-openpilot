"""After the measured suite finishes, audit evidence and render videos on the Spark CPU."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def process_episode(directory):
    import cv2
    import torch

    from isaac_eval.render_debug import export_rrd, render
    from scripts.audit_isaac_episode import audit_episode

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    directory = Path(directory)
    started = time.perf_counter()
    audit = audit_episode(directory)
    if not audit["passed"]:
        raise ValueError(f"Episode audit failed: {directory}")
    render(directory / "evidence", directory / "debug-source.mp4", directory / "research-source.mp4")
    export_rrd(directory / "evidence", directory / "debug.rrd")
    count = audit["frame_count"]
    result = {
        "case_id": directory.name,
        "inference_frames": count,
        "terminal_frames": 40,
        "completion_sha256": digest(directory / "evidence/completion.json"),
        "artifact_audit_sha256": digest(directory / "artifact_audit.json"),
        "producer_sha256": digest(__file__),
        "videos": {},
    }
    for name in ("debug", "research"):
        source, target = directory / f"{name}-source.mp4", directory / f"{name}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-threads",
                "1",
                "-movflags",
                "+faststart",
                str(target),
            ],
            check=True,
        )
        capture = cv2.VideoCapture(str(target))
        fps, decoded = capture.get(cv2.CAP_PROP_FPS), 0
        samples = {0, count // 2, count - 1, count + 20}
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if image.shape != (768, 1280, 3):
                raise ValueError(f"Unexpected video dimensions: {target}")
            if decoded in samples:
                cv2.imwrite(str(directory / f"{name}_{decoded:06d}.jpg"), image)
            decoded += 1
        capture.release()
        if fps != 20 or decoded != count + 40:
            raise ValueError(f"Video decode/cadence mismatch: {target}")
        result["videos"][name] = {
            "sha256": digest(target),
            "source_sha256": digest(source),
            "fps": fps,
            "decoded_frames": decoded,
        }
    result["rrd_sha256"] = digest(directory / "debug.rrd")
    result["postprocess_seconds"] = time.perf_counter() - started
    (directory / "video_audit.json").write_text(json.dumps(result, indent=2))
    return {"case_id": directory.name, "frames": count, "seconds": result["postprocess_seconds"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--install-ffmpeg", action="store_true")
    parser.add_argument("--install-rerun", action="store_true")
    args = parser.parse_args()
    while True:
        status_path = args.run / "status.json"
        status = json.loads(status_path.read_text()) if status_path.is_file() else {}
        if status.get("complete"):
            break
        if any(not row["technical_success"] for row in status.get("cases", [])):
            raise ValueError("Measured suite stopped with a technical failure")
        if not args.wait:
            raise ValueError("Measured suite must finish before CPU postprocessing")
        time.sleep(30)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    if shutil.which("ffmpeg") is None and args.install_ffmpeg:
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], check=True)
    if importlib.util.find_spec("rerun") is None and args.install_rerun:
        subprocess.run([sys.executable, "-m", "pip", "install", "rerun-sdk==0.42.0"], check=True)
    subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, check=True)
    started = time.time()
    outcomes = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_episode, args.run / row["case_id"]) for row in status["cases"]]
        for future in as_completed(futures):
            result = future.result()
            outcomes.append(result)
            (args.run / "postprocess_status.json").write_text(
                json.dumps(
                    {
                        "complete": len(outcomes) == len(futures),
                        "cases": outcomes,
                        "wall_seconds": time.time() - started,
                    },
                    indent=2,
                )
            )
            print("VERIFIED_VIDEO", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
