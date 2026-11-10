#!/usr/bin/env python3
"""Verify an already copied batch against two remote inventories and write a receipt."""
import argparse
import importlib.util
import json
import os
import re
import sys
import subprocess
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

script = Path(__file__).with_name("archive_lab_batch.py")
spec = importlib.util.spec_from_file_location("archive_lab_batch", script)
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


def verify(host, remote_root, batch, destination):
    if not re.fullmatch(r"[A-Za-z0-9_@.-]+", host) or host.startswith("-"):
        raise ValueError("Invalid SSH host")
    if not PurePosixPath(remote_root).is_absolute():
        raise ValueError("Remote task root must be absolute")
    if not batch or PurePosixPath(batch).is_absolute() or any(p in ("", ".", "..") for p in batch.split("/")):
        raise ValueError("Batch must be a confined relative path")
    destination = Path(destination).absolute()
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("Destination must be a real directory")
    destination = destination.parent.resolve(strict=True) / destination.name
    receipt_path = destination.parent / (destination.name + ".receipt.json")
    if receipt_path.exists() or receipt_path.is_symlink():
        raise ValueError("Receipt already exists")
    first = archive.remote_inventory(host, remote_root, batch)
    local = archive.inventory(destination)
    if local != first["entries"]:
        raise ValueError("Local inventory differs from source")
    bindings = archive.validate_batch(destination, local)
    second = archive.remote_inventory(host, remote_root, batch)
    if second != first:
        raise ValueError("Remote batch changed during verification")
    if archive.inventory(destination) != local:
        raise ValueError("Local batch changed during verification")
    receipt = {
        "version": 1, "verified_at_utc": datetime.now(UTC).isoformat(), "ssh_host": host,
        "remote_hostname": first["hostname"], "remote_root": first["root"], "remote_batch": first["batch"],
        "local_destination": str(destination), "plan_lock_sha256": local["plan.lock.json"]["sha256"],
        "manifest_sha256": local["manifest.json"]["sha256"], "episodes": bindings, "inventory": local,
    }
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return receipt_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-host", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(verify(args.ssh_host, args.remote_root, args.batch, args.destination))
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
