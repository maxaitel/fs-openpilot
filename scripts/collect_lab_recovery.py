"""Collect a gated recovery dataset without replacing any prior attempt."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shutil
import signal
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsop.data.native_replay import NativeEpisode
from fsop.sim.expert import ExpertParams
from isaac_lab_fsop.evaluate import source_inventory
from scripts.verify_lab_dataset import digest
from scripts.verify_lab_dataset import verify as verify_dataset
from scripts.verify_lab_recovery import verify as verify_recovery

SOURCE_MANIFEST = "df238ccad932d5704f7091cbff1872f580c7d8dce4951468c532e66fb8b85e37"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_new(path, document):
    with path.open("x") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")


def command_for(case_id, recipe, recipe_path, checkpoint, destination):
    teacher = recipe["teachers"][case_id]
    params = ExpertParams(
        v_cap=teacher["v_cap"],
        a_lat_max=teacher["a_lat_max"],
        controller=teacher["controller"],
        k_lookahead=teacher["k_lookahead"],
    )
    require(asdict(params) == teacher, f"Collector cannot reproduce teacher parameters for {case_id}")
    collection = recipe["collection"]
    return [
        sys.executable,
        "-m",
        "isaac_lab_fsop.collect",
        "--case",
        str(recipe_path.parent / "cases" / (case_id + ".json")),
        "--membership",
        str((recipe_path.parent / recipe["membership"]).resolve()),
        "--out",
        str(destination),
        "--checkpoint",
        str(checkpoint),
        "--takeover",
        str(collection["takeover_seconds"]),
        "--lateral-takeover",
        str(collection["lateral_threshold_m"]),
        "--speed",
        str(teacher["v_cap"]),
        "--alat",
        str(teacher["a_lat_max"]),
        "--controller",
        str(teacher["controller"]),
        "--lookahead",
        str(teacher["k_lookahead"]),
    ]


def require_idle_gpu():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        require(
            len(parts) == 2 and parts[0].isdigit() and parts[1] == "/usr/bin/sunshine",
            f"GPU has another compute process: {line}",
        )


def run_child(command, log_path, timeout):
    with log_path.open("x") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            result = child.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
            raise
        if result:
            raise RuntimeError(f"Collector exited {result}; inspect {log_path}")


def retained_matches(source, out):
    source_files = {p.relative_to(source) for p in source.rglob("*") if p.is_file()}
    target_files = {p.relative_to(out) for p in out.rglob("*") if p.is_file()}
    return source_files == target_files and all(os.path.samefile(source / name, out / name) for name in source_files)


def collect(args):
    recipe_path = args.recipe.resolve()
    source = args.source.resolve()
    out = args.out.resolve()
    receipt_path = (args.source_receipt or source / "verification.json").resolve()
    recipe = json.loads(recipe_path.read_text())
    membership_path = (recipe_path.parent / recipe["membership"]).resolve()
    membership = json.loads(membership_path.read_text())
    checkpoint = (args.checkpoint or Path(recipe["checkpoint"])).resolve()
    require(
        math.isfinite(args.timeout) and 0 < args.timeout <= 1200, "Timeout must be positive and at most 1200 seconds"
    )
    require(
        recipe.get("schema_version") == 1
        and recipe.get("collection")
        == {"takeover_seconds": 120, "lateral_threshold_m": 0.4, "capture_hz": 20, "image_hz": 5}
        and recipe.get("gate")
        == {
            "min_lateral_m": 0.3,
            "min_yaw_rad": 0.1,
            "min_anchors": 8,
            "window_seconds": 2,
            "return_within_seconds": 4,
            "settled_lateral_m": 0.2,
            "settled_yaw_rad": 0.1,
            "settled_seconds": 0.5,
        },
        "Coordinator requires the frozen event recovery protocol",
    )
    ids, pilots = recipe["recovery_case_ids"], recipe["pilot_case_ids"]
    cases = {row["id"]: row for row in membership["cases"] if row["split"] in ("train", "val")}
    require(
        len(cases) == 40
        and len(ids) == 8
        and len(set(ids)) == 8
        and len(pilots) == 4
        and pilots == ids[:4]
        and set(ids) <= set(cases)
        and all(cases[key]["split"] == "train" for key in ids),
        "Invalid frozen membership or recovery cases",
    )
    require(digest(membership_path) == recipe["membership_sha256"], "Membership digest differs")
    for key in ids:
        case_file = recipe_path.parent / "cases" / (key + ".json")
        require(
            digest(case_file) == recipe["case_sha256"][key] and json.loads(case_file.read_text()) == cases[key],
            f"Case differs for {key}",
        )
        command_for(key, recipe, recipe_path, checkpoint, out / key)
    require(digest(checkpoint) == recipe["checkpoint_sha256"], "Checkpoint digest differs")
    manifest = source / "manifest.json"
    require(digest(manifest) == SOURCE_MANIFEST, "Source manifest digest differs from frozen lab_v2")
    source_report = verify_dataset(manifest, membership_path)
    require(
        json.loads(receipt_path.read_text()) == source_report,
        "Source verification receipt differs from fresh verification",
    )
    manifest_doc = json.loads(manifest.read_text())
    retained = [row for row in manifest_doc["episodes"] if row["path"] not in ids]
    require(
        len(retained) == 32 and {row["path"] for row in manifest_doc["episodes"]} == set(cases),
        "Source manifest does not contain exact membership",
    )
    require(not out.is_relative_to(source) and not source.is_relative_to(out), "Source and output must be separate")
    require(Path.cwd().resolve() == Path(__file__).resolve().parents[1], "Run the coordinator from the repository root")
    identity = {
        "producer_sources": source_inventory(),
        "python": sys.executable,
        "coordinator_sha256": digest(__file__),
        "recovery_verifier_sha256": digest(Path(__file__).with_name("verify_lab_recovery.py")),
        "dataset_verifier_sha256": digest(Path(__file__).with_name("verify_lab_dataset.py")),
        "recipe": str(recipe_path),
        "recipe_sha256": digest(recipe_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "source": str(source),
        "source_manifest_sha256": digest(manifest),
        "source_receipt": str(receipt_path),
        "source_receipt_sha256": digest(receipt_path),
        "membership_sha256": digest(membership_path),
        "out": str(out),
        "timeout": args.timeout,
        "reserve_gib": 25,
    }
    if not out.exists():
        out.mkdir(parents=True)
        write_new(out / "identity.json", identity)
    require(out.is_dir() and (out / "identity.json").is_file(), "Output already exists without coordinator identity")
    require(json.loads((out / "identity.json").read_text()) == identity, "Coordinator identity or arguments changed")
    with (out / ".collection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (out / "logs").mkdir(exist_ok=True)
        (out / "recovery_gates").mkdir(exist_ok=True)
        (out / "attempts").mkdir(exist_ok=True)
        (out / "seals").mkdir(exist_ok=True)
        known = {
            "identity.json",
            ".collection.lock",
            "logs",
            "recovery_gates",
            "attempts",
            "seals",
            "manifest.json",
            "verification.json",
        } | set(cases)
        require(all(p.name in known for p in out.iterdir()), "Output contains unrelated entries")
        for directory, suffix in (
            ("logs", ".log"),
            ("attempts", ".json"),
            ("recovery_gates", ".json"),
            ("seals", ".json"),
        ):
            allowed = {key + suffix for key in ids}
            require(
                all(p.is_file() and p.name in allowed for p in (out / directory).iterdir()),
                f"Output contains unrelated entries in {directory}",
            )
        for row in retained:
            src, dst = source / row["path"], out / row["path"]
            if not dst.exists():
                shutil.copytree(src, dst, copy_function=os.link)
            require(retained_matches(src, dst), f"Retained episode changed: {row['path']}")

        def check_stable():
            require(
                source_inventory() == identity["producer_sources"]
                and digest(__file__) == identity["coordinator_sha256"]
                and digest(Path(__file__).with_name("verify_lab_recovery.py")) == identity["recovery_verifier_sha256"]
                and digest(Path(__file__).with_name("verify_lab_dataset.py")) == identity["dataset_verifier_sha256"]
                and digest(recipe_path) == identity["recipe_sha256"]
                and digest(checkpoint) == identity["checkpoint_sha256"]
                and digest(manifest) == identity["source_manifest_sha256"]
                and digest(receipt_path) == identity["source_receipt_sha256"]
                and digest(membership_path) == identity["membership_sha256"],
                "Frozen input changed",
            )

        def check_prior(key):
            attempt = out / "attempts" / (key + ".json")
            receipt = out / "recovery_gates" / (key + ".json")
            seal = out / "seals" / (key + ".json")
            require(
                attempt.is_file() and receipt.is_file() and seal.is_file(), f"Missing or failed prior attempt: {key}"
            )
            record = json.loads(attempt.read_text())
            gate = json.loads(receipt.read_text())
            require(
                json.loads((out / key / "meta.json").read_text())["source_sha256"] == identity["producer_sources"]
                and json.loads(seal.read_text()) == {"receipt_sha256": digest(receipt)}
                and record == {"command": command_for(key, recipe, recipe_path, checkpoint, out / key)}
                and gate.get("passed") is True
                and gate == verify_recovery(out / key, recipe_path),
                f"Prior recovery receipt changed or failed: {key}",
            )

        for key in ids:
            attempt = out / "attempts" / (key + ".json")
            receipt = out / "recovery_gates" / (key + ".json")
            if (
                attempt.exists()
                or receipt.exists()
                or (out / "seals" / (key + ".json")).exists()
                or (out / key).exists()
                or (out / "logs" / (key + ".log")).exists()
            ):
                check_prior(key)
        if args.phase == "remaining":
            for key in pilots:
                check_prior(key)
        selected = pilots if args.phase == "pilot" else ids[4:]
        for key in selected:
            check_stable()
            if (out / "attempts" / (key + ".json")).exists():
                continue
            require(shutil.disk_usage(out).free >= 25 * 2**30, "Free disk is below the 25 GiB reserve")
            require_idle_gpu()
            command = command_for(key, recipe, recipe_path, checkpoint, out / key)
            write_new(out / "attempts" / (key + ".json"), {"command": command})
            run_child(command, out / "logs" / (key + ".log"), args.timeout)
            check_stable()
            require(
                json.loads((out / key / "meta.json").read_text())["source_sha256"] == identity["producer_sources"],
                f"Producer source differs for {key}",
            )
            gate = verify_recovery(out / key, recipe_path)
            write_new(out / "recovery_gates" / (key + ".json"), gate)
            require(gate["passed"], f"Recovery gate failed for {key}; attempt and receipt preserved")
            write_new(
                out / "seals" / (key + ".json"), {"receipt_sha256": digest(out / "recovery_gates" / (key + ".json"))}
            )
        if args.phase == "remaining":
            for key in ids:
                check_prior(key)
            rows = []
            for source_row in manifest_doc["episodes"]:
                key = source_row["path"]
                if key not in ids:
                    rows.append(source_row)
                else:
                    ep = NativeEpisode(out / key)
                    rows.append(
                        {
                            "path": key,
                            "meta_sha256": digest(out / key / "meta.json"),
                            "accepted": len(ep.anchors),
                            "rejections": ep.rejections,
                            "wall_seconds": ep.meta["wall_seconds"],
                            "summary": ep.meta["summary"],
                        }
                    )
            final = {
                "schema_version": 1,
                "membership_sha256": recipe["membership_sha256"],
                "episodes": rows,
                "expected_episodes": 40,
                "complete": True,
            }
            final_path = out / "manifest.json"
            if final_path.exists():
                require(json.loads(final_path.read_text()) == final, "Final manifest changed")
            else:
                write_new(final_path, final)
            report = verify_dataset(final_path, membership_path)
            verification = out / "verification.json"
            if verification.exists():
                require(json.loads(verification.read_text()) == report, "Final verification changed")
            else:
                write_new(verification, report)
    print(json.dumps({"phase": args.phase, "passed": True, "out": str(out)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--phase", choices=("pilot", "remaining"), required=True)
    parser.add_argument("--timeout", type=float, default=1200)
    collect(parser.parse_args())


if __name__ == "__main__":
    main()
