#!/usr/bin/env python3
"""E18: per prompt, keep the neural candidate whose best planned grasp has the highest predicted lift.

Reads `<root>/<model>_c<j>/seed_<s>/objects.json` for j = 0..k-1 (built by build_neural_manifest.py with
--table-check --grasp-model, so each grasp carries p_success) and writes `<root>/<model>_sel/seed_<s>/`:
eval_manifest.json (the chosen objects, method `<model>_sel`) and selection.json (every candidate's score and the
choice). Score: max p_success over planned grasps; a candidate with no planned grasp scores -1, one that failed to
build -2. Ties go to the lower j, so candidate 0 (the E12 mesh) wins a tie.

    python scripts/e18_select.py --model shap_e --seeds 65 66 --root output/e18
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def score(row: dict) -> float:
    if "failed" in row or "sdf" not in row:
        return -2.0
    ps = [g.get("p_success") for g in row.get("grasps", []) if g.get("p_success") is not None]
    return max(ps) if ps else -1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--root", type=Path, default=Path("output/e18"))
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()
    for s in args.seeds:
        cands = []
        for j in range(args.k):
            rows = json.loads((args.root / f"{args.model}_c{j}" / f"seed_{s}" / "objects.json").read_text())
            by_idx = {int(re.search(r"_(\d{4})$", r["name"]).group(1)): r for r in rows}
            cands.append(by_idx)
        n = max(max(c) for c in cands) + 1
        chosen, record = [], []
        for i in range(n):
            sc = [score(c[i]) if i in c else -2.0 for c in cands]
            j = max(range(args.k), key=lambda q: (sc[q], -q))
            record.append({"index": i, "scores": sc, "chosen": j,
                           "archetype": next((c[i]["archetype"] for c in cands if i in c), None)})
            row = cands[j].get(i)
            if row is not None and sc[j] > -2.0:
                row = dict(row, method=f"{args.model}_sel", candidate=j)
                chosen.append(row)
        out = args.root / f"{args.model}_sel" / f"seed_{s}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "eval_manifest.json").write_text(json.dumps(chosen, indent=1))
        (out / "selection.json").write_text(json.dumps(record, indent=1))
        picks = [r["chosen"] for r in record]
        print(f"{args.model} seed {s}: {len(chosen)} of {n} built; chosen candidate counts",
              {j: picks.count(j) for j in range(args.k)})


if __name__ == "__main__":
    main()
