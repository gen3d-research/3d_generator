#!/bin/bash
# Predictor scaling study (ROADMAP §3, design of 12 Nov 2026).
#   data   plain-CEM manifests on fresh seeds and their MuJoCo grasp dataset
#   curve  learning curve (A); train: (A), archetype + CEM mixing (B), CEM alone (C); 3 training seeds each
# Usage (from 3d_generator/):  bash scripts/run_predictor_scaling.sh data | curve | train | phase2
set -u
cd "$(dirname "$0")/.."
OUT=output/predictor_scaling
SEEDS=$(seq 100 139)
PY="${PYTHON:-$HOME/venv/3d_cem/bin/python}"
TPY="${TPYTHON:-$HOME/venv/gen3d_learn/bin/python}"
CEM_DATA=output/grasp_dataset/cem_100_139.npz
ARCH=output/grasp_dataset/archetypes.npz
TEST=output/grasp_dataset/journal.npz
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT"

data() {
  # One manifest per seed: plain CEM, top 100 of the trained distribution.
  echo "$SEEDS" | xargs -P "${JOBS:-12}" -I{} bash -c '
    D='"$OUT"'/cem/seed_{}; mkdir -p "$D"
    [ -s "$D/eval_manifest.json" ] && exit 0
    nice -n 19 env -u PYTHONPATH '"$PY"' scripts/build_eval_manifest.py --methods cem --budget 1500 \
      --top-k 100 --seed {} --out "$D/eval_manifest.json" --export-root "$D/manifest_objects" \
      > "$D/build.log" 2>&1; echo "[seed {}] manifest exit=$?"'
  nice -n 19 env -u PYTHONPATH "$PY" scripts/build_grasp_dataset.py --out "$CEM_DATA" \
    --manifests "$OUT"/cem/seed_*/eval_manifest.json --workers "${WORKERS:-16}"
}

train_one() {   # name, then trainer args
  local name=$1; shift
  [ -s "$OUT/runs/$name/results.json" ] && return 0
  env -u PYTHONPATH "$TPY" scripts/train_grasp_predictor.py --eval-data "$TEST" \
    --out "$OUT/runs/$name" "$@" > "$OUT/runs/$name.log" 2>&1
  echo "[$name] exit=$?"
}

curve() {      # (A) needs no CEM data
  mkdir -p "$OUT/runs"
  for s in 0 1 2; do
    for f in 0.0625 0.125 0.25 0.5 1.0; do
      train_one "A_frac${f}_s$s" --data "$ARCH" --split archetype --seed $s --train-frac $f
    done
  done
}

train() {
  curve
  for s in 0 1 2; do
    for k in 5 10 20 40; do          # K = 0 is A_frac1.0
      train_one "B_k${k}_s$s" --data "$ARCH" --split archetype --seed $s \
        --extra-train "$CEM_DATA" --extra-sources $k
    done
    train_one "C_k40_s$s" --data "$ARCH" --split archetype --seed $s --train-frac 0 \
      --extra-train "$CEM_DATA"
  done
}

phase2() {     # selection with each predictor on fresh seeds, MuJoCo nominal (design: ROADMAP §3)

  local P2=$OUT/phase2
  for M in arch mixed; do
    for S in $(seq 140 149); do echo "$M $S"; done
  done | xargs -P "${JOBS:-12}" -n 2 bash -c '
    M=$0; S=$1; D='"$P2"'/$M/seed_$S; mkdir -p "$D"
    [ -s "$D/eval_manifest.json" ] && exit 0
    case $M in arch) MOD=output/grasp_predictor/arch_object_s0/model.npz ;;
               mixed) MOD='"$OUT"'/runs/B_k40_s0/model.npz ;; esac
    nice -n 19 env -u PYTHONPATH '"$PY"' scripts/build_eval_manifest.py --methods cem --budget 1500 \
      --top-k 25 --seed $S --table-check --grasp-model $MOD --cem-pool 4 \
      --out "$D/eval_manifest.json" --export-root "$D/manifest_objects" > "$D/build.log" 2>&1
    echo "[$M seed $S] manifest exit=$?"'
  for M in arch mixed; do
    nice -n 19 env -u PYTHONPATH "$PY" scripts/run_mujoco_eval.py --root "$P2/$M" --seeds $(seq 140 149) \
      --collision coacd --workers "${WORKERS:-12}" > "$P2/$M/mujoco_nominal.log" 2>&1
    echo "[$M] mujoco exit=$?"
  done
}

case "${1:-}" in
  data) data ;;
  curve) curve ;;
  train) train ;;
  phase2) phase2 ;;
  *) echo "usage: $0 data|curve|train|phase2"; exit 2 ;;
esac
