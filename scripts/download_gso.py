#!/usr/bin/env python3
"""E13: Google Scanned Objects from Gazebo Fuel (CC BY 4.0). Records each model's name, size and licence in
index.json. Not for git: the meshes and textures stay in output/."""
import argparse
import io
import json
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://fuel.gazebosim.org/1.0/GoogleResearch/models"


def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception:     # noqa: BLE001 - transient network errors are retried
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def fetch(entry, out):
    name = entry["name"]
    d = out / name
    if (d / "model.sdf").exists():
        return name, True
    try:
        z = zipfile.ZipFile(io.BytesIO(get(f"{API}/{urllib.parse.quote(name)}.zip")))
        root = d.resolve()
        for m in z.infolist():
            target = (d / m.filename).resolve()
            if not str(target).startswith(str(root)):     # refuses paths that leave the folder
                continue
            if m.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(m))
        return name, True
    except Exception as e:     # noqa: BLE001
        print(name, "failed", e, flush=True)
        return name, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("output/e13/gso"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    models, page = [], 1
    while True:
        batch = json.loads(get(f"{API}?per_page=100&page={page}"))
        if not batch:
            break
        models += batch
        page += 1
        if len(batch) < 100:
            break
    keep = [{k: m.get(k) for k in ("name", "filesize", "license_name", "license_url", "description")} for m in models]
    (args.out.parent / "gso_index.json").write_text(json.dumps(keep, indent=1))
    size = sum(m["filesize"] or 0 for m in models)
    print(len(models), "models", round(size / 1e9, 1), "GB compressed", {m["license_name"] for m in models}, flush=True)
    with ThreadPoolExecutor(args.workers) as ex:
        done = list(ex.map(lambda m: fetch(m, args.out), keep))
    print("GSO_DONE", sum(ok for _, ok in done), "of", len(done), flush=True)


if __name__ == "__main__":
    main()
