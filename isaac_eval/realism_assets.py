"""Download CC0 surface maps and HDRIs for the Isaac realism evaluation.

Powered by Poly Haven. Run with ``python -m isaac_eval.realism_assets --out DIR``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen

API = "https://api.polyhaven.com"
USER_AGENT = "FSOpenpilotIsaacEval/1.0 (Poly Haven realism asset downloader)"
LICENSE = "https://polyhaven.com/license"
TEXTURES = ("asphalt_track", "asphalt_02", "concrete_floor_01", "gravel_floor_04")
HDRIS = ("derelict_airfield_01", "mall_parking_lot")
MAPS = {"diffuse": "Diffuse", "roughness": "Rough", "normal_gl": "nor_gl"}


def request(url: str):
    return urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=120)


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verified(path: Path, entry: dict) -> bool:
    return (path.is_file() and path.stat().st_size == entry["size"]
            and digest(path, "md5") == entry["md5"]
            and ("sha256" not in entry or digest(path, "sha256") == entry["sha256"]))


def fetch_file(out: Path, entry: dict) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    if verified(out, entry):
        print(f"Verified cache: {out}")
    else:
        print(f"Downloading: {out}", flush=True)
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(dir=out.parent, prefix=".download-", delete=False) as f:
                tmp = Path(f.name)
                with request(entry["url"]) as response:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        f.write(chunk)
            if not verified(tmp, entry):
                raise ValueError(f"Download failed checksum or size verification: {entry['url']}")
            os.replace(tmp, out)
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
    return {"path": str(out), "url": entry["url"], "size": entry["size"],
            "md5": entry["md5"], "sha256": digest(out, "sha256")}


def download(out: Path, lock: Path | None = None) -> Path:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if lock is not None:
        with lock.open() as source:
            manifest = json.load(source)
        for asset in manifest["assets"].values():
            for entry in asset["files"].values():
                relative = Path(entry["path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Invalid locked asset path: {relative}")
                if "sha256" not in entry:
                    raise ValueError(f"Missing SHA-256 for locked asset: {relative}")
                fetch_file(out / relative, entry)
        return write_manifest(out, manifest)

    assets = {}
    for asset_id in (*TEXTURES, *HDRIS):
        with request(f"{API}/info/{asset_id}") as response:
            info = json.load(response)
        with request(f"{API}/files/{asset_id}") as response:
            files = json.load(response)
        texture = asset_id in TEXTURES
        selections = ({name: (files[key]["2k"]["jpg"], f"{name}.jpg")
                       for name, key in MAPS.items()} if texture else
                      {"hdri": (files["hdri"]["4k"]["hdr"], "hdri.hdr")})
        records = {}
        for name, (entry, filename) in selections.items():
            record = fetch_file(out / asset_id / filename, entry)
            record["path"] = str(Path(record["path"]).relative_to(out))
            records[name] = record
        assets[asset_id] = {"name": info["name"], "authors": info.get("authors", {}),
                            "asset_url": f"https://polyhaven.com/a/{asset_id}",
                            "license": "CC0-1.0", "license_url": LICENSE,
                            "resolution": "2k" if texture else "4k",
                            "dimensions_mm": info.get("dimensions") if texture else None,
                            "files": records}
    manifest = {"credit": "Powered by Poly Haven", "api": API, "assets": assets}
    return write_manifest(out, manifest)


def write_manifest(out: Path, manifest: dict) -> Path:
    target = out / "manifest.json"
    with tempfile.NamedTemporaryFile(mode="w", dir=out, prefix=".manifest-", delete=False) as f:
        tmp = Path(f.name)
        try:
            json.dump(manifest, f, indent=2)
            f.write("\n")
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, target)
    print(f"Powered by Poly Haven. Manifest: {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="Asset cache directory")
    parser.add_argument("--lock", type=Path, help="Use exact files and hashes from a manifest without querying the API")
    args = parser.parse_args()
    download(args.out, args.lock)


if __name__ == "__main__":
    main()
