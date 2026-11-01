#!/bin/bash
# E13: real shapes (Google Scanned Objects) in both arms: prepare, build manifests, MuJoCo drop and lift, Gazebo drop and lift, summary.
# Usage (from 3d_generator/): bash scripts/run_e13.sh [root] [--limit N]
ROOT="${1:-output/e13/graded}"; LIMIT="${2:-}"
PY="$HOME/venv/3d_cem/bin/python"
NEURAL="$HOME/venv/gen3d_neural/bin/python"
MODEL=output/grasp_predictor/arch_object_s0/model.npz
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ROS_DOMAIN_ID=23 GZ_PARTITION=gen3d_e13
run() { nice -n 15 env -u PYTHONPATH "$PY" "$@"; }
[ -s "$ROOT/gso_ranked/seed_0/clean.json" ] || nice -n 15 env -u PYTHONPATH "$NEURAL" scripts/e13_prepare_gso.py --root "$ROOT" ${LIMIT:+--limit "$LIMIT"} --workers "${BUILD_WORKERS:-12}"
[ -s "$ROOT/gso_plain/seed_0/eval_manifest.json" ] || run scripts/build_neural_manifest.py --model gso_plain --seed 0 --root "$ROOT" --coacd-root "$ROOT/coacd" --workers "${BUILD_WORKERS:-12}" > "$ROOT/gso_plain/build.log" 2>&1
[ -s "$ROOT/gso_ranked/seed_0/eval_manifest.json" ] || run scripts/build_neural_manifest.py --model gso_ranked --seed 0 --root "$ROOT" --coacd-root "$ROOT/coacd" --table-check --grasp-model "$MODEL" --workers "${BUILD_WORKERS:-12}" > "$ROOT/gso_ranked/build.log" 2>&1
echo "[e13] build done $(date +%H:%M)"
set +u; source /opt/ros/jazzy/setup.bash; set -u
for A in gso_plain gso_ranked; do
  D="$ROOT/$A/seed_0"
  run scripts/run_mujoco_eval.py --root "$ROOT/$A" --seeds 0 --collision coacd --workers "${MJ_WORKERS:-10}" > "$ROOT/$A/mujoco.log" 2>&1
  echo "[e13 $A] mujoco exit=$? $(date +%H:%M)"
  [ -s "$D/gazebo_stability.json" ] || bash scripts/run_drop_test_chunked.sh "$D/eval_manifest.json" "$D/gazebo_stability.json" 25 "${GZ_WORKERS:-10}" > "$D/gazebo_drop.log" 2>&1
  [ -s "$D/gazebo_lift.json" ] || python3 scripts/gazebo_lift_eval.py --manifest "$D/eval_manifest.json" --mujoco "$D/mujoco_eval_coacd.json" --out "$D/gazebo_lift.json" --workers "${GZ_WORKERS:-10}" > "$D/gazebo_lift.log" 2>&1
  echo "[e13 $A] gazebo exit=$? $(date +%H:%M)"
done
run scripts/summarize_e13.py --root "$ROOT" --out "$ROOT/summary.json"
echo E13_DONE
