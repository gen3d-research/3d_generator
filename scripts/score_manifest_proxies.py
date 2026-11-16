#!/usr/bin/env python3
"""Proxy scores of graded objects: run the generator's ObjectScorer on each manifest entry's exported visual mesh.

Writes `<out-root>/<manifest's folder relative to output/>/proxies.json`, never into the result folders
(E10's are sealed): per object, the size, stability and graspability scores and their raw
measures (stability margin, antipodal pair count). Complexity and assembly need the primitive list, which an exported
mesh does not carry, so they are left out; the totals here are therefore not the generator's training score.

    python scripts/score_manifest_proxies.py output/journal_10seed/seed_*/eval_manifest.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

FIELDS = ("size_score", "stability_score", "graspability_score", "validity_score", "stability_margin",
          "n_antipodal_pairs", "is_watertight")


def one(entry: dict) -> dict:
    import trimesh
    from build_neural_manifest import MeshObject
    from scoring import ObjectScorer
    row = {"name": entry["name"], "method": entry.get("method")}
    try:
        m = trimesh.load(entry["visual_mesh"], force="mesh", process=True)
        b = ObjectScorer().score(MeshObject(m, entry["name"]))
        row.update({k: float(getattr(b, k)) for k in FIELDS})
    except Exception as e:     # noqa: BLE001
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifests", nargs="+")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-root", type=Path, default=ROOT / "output/proxies")
    args = ap.parse_args()
    paths = [Path(p) for pat in args.manifests for p in (sorted(glob.glob(pat)) or [pat])]
    with ProcessPoolExecutor(args.workers) as ex:
        for p in paths:
            out = args.out_root / p.resolve().parent.relative_to((ROOT / "output").resolve()) / "proxies.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists():
                continue
            entries = [e for e in json.loads(p.read_text()) if "visual_mesh" in e]
            rows = list(ex.map(one, entries, chunksize=4))
            out.write_text(json.dumps(rows, indent=1))
            print(p, len(rows), "objects,", sum("error" in r for r in rows), "errors", flush=True)


if __name__ == "__main__":
    main()
