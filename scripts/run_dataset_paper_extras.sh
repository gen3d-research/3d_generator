#!/bin/bash
# Grasp dataset paper, added analyses (ROADMAP, design of 16 Nov 2026).
#   ablate   predictor with scalars only / points only, archetype split, training seeds 0-2
#   labels   the 10-seed journal set relabelled under friction 0.3 / 1.0 and grip 10 / 40 N,
#            plus a nominal rerun as a determinism check
# Usage (from 3d_generator/):  bash scripts/run_dataset_paper_extras.sh ablate | labels
set -u
cd "$(dirname "$0")/.."
OUT=output/dataset_paper
PY="${PYTHON:-$HOME/venv/3d_cem/bin/python}"
TPY="${TPYTHON:-$HOME/venv/gen3d_learn/bin/python}"
ARCH=output/grasp_dataset/archetypes.npz
TEST=output/grasp_dataset/journal.npz
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT/runs" "$OUT/labels"

ablate() {
  for s in 0 1 2; do
    for f in scalars points; do
      n="${f}_s$s"
      [ -s "$OUT/runs/$n/results.json" ] && continue
      env -u PYTHONPATH "$TPY" scripts/train_grasp_predictor.py --data "$ARCH" --split archetype \
        --seed $s --features $f --eval-data "$TEST" --out "$OUT/runs/$n" > "$OUT/runs/$n.log" 2>&1
      echo "[$n] exit=$?"
    done
  done
}

labels() {
  for cfg in "0.5 20" "0.3 20" "1.0 20" "0.5 10" "0.5 40"; do
    set -- $cfg
    f="$OUT/labels/journal_mu$1_grip$2.npz"
    [ -s "$f" ] && continue
    nice -n 19 env -u PYTHONPATH "$PY" scripts/build_grasp_dataset.py --out "$f" --mu $1 --grip $2 \
      --manifests output/journal_10seed/seed_*/eval_manifest.json --workers "${WORKERS:-12}" \
      > "${f%.npz}.log" 2>&1
    echo "[mu $1 grip $2] exit=$?"
  done
}

case "${1:-}" in
  ablate) ablate ;;
  labels) labels ;;
  *) echo "usage: $0 ablate|labels"; exit 2 ;;
esac
