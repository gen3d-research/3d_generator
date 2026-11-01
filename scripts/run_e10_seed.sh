#!/bin/bash
# E10, one seed: builds the four arms' manifests (CoACD concave collision), grades them in MuJoCo
# (true geometry, and feasibility for Gazebo) and with the Gazebo pick-and-lift, writes metrics.
# Usage (from 3d_generator/): OUT=output/e10 MODEL=... bash scripts/run_e10_seed.sh 65
set -u
cd "$(dirname "$0")/.."
S="${1:?seed}"
OUT="${OUT:-output/e10}"
MODEL="${MODEL:-output/grasp_predictor/arch_object_s0/model.npz}"
W="${GZ_WORKERS:-8}"
PY="${PYTHON:-$HOME/venv/3d_cem/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-23}" GZ_PARTITION="${GZ_PARTITION:-gen3d}"
run() { nice -n "${NICE:-10}" env -u PYTHONPATH "$PY" "$@"; }
build() {  # config, method, extra options
  local CFG=$1 M=$2; shift 2
  local D="$OUT/$CFG/seed_$S"; mkdir -p "$D"
  [ -s "$D/eval_manifest.json" ] && return 0
  run scripts/build_eval_manifest.py --budget 1500 --top-k 25 --seed "$S" --methods $M \
      --concave-decomposition coacd --out "$D/eval_manifest.json" \
      --export-root "$D/manifest_objects" "$@" > "$D/build.log" 2>&1
  echo "[e10 $CFG $S] manifest exit=$?"
}
build plain "cem fixed_cad" &
build ranked fixed_cad --table-check --grasp-model "$MODEL" &
build select cem --table-check --grasp-model "$MODEL" --cem-pool 4 &
wait
for CFG in plain ranked select; do
  run scripts/run_mujoco_eval.py --root "$OUT/$CFG" --seeds "$S" --collision coacd \
      --workers "${MJ_WORKERS:-8}" > "$OUT/$CFG/seed_$S/mujoco.log" 2>&1
  echo "[e10 $CFG $S] mujoco exit=$?"
done
set +u
source "${ROS_SETUP:-/opt/ros/jazzy/setup.bash}" 2>/dev/null
set -u
for CFG in plain ranked select; do
  D="$OUT/$CFG/seed_$S"
  [ -s "$D/gazebo_lift.json" ] || python3 scripts/gazebo_lift_eval.py --manifest "$D/eval_manifest.json" \
      --mujoco "$D/mujoco_eval_coacd.json" --out "$D/gazebo_lift.json" --workers "$W" > "$D/gazebo_lift.log" 2>&1
  echo "[e10 $CFG $S] gazebo exit=$?"
done
run scripts/summarize_e10.py --root "$OUT" --seed "$S" --out "$OUT/metrics_seed_$S.json"
