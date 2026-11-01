#!/bin/bash
# E12: grade one model's raw meshes on the seeds given: clean, build manifest, MuJoCo drop and lift, Gazebo drop and lift.
# Usage (from 3d_generator/): bash scripts/run_e12_grade.sh shap_e 65 66 ...
M="$1"; shift
PY="$HOME/venv/3d_cem/bin/python"
NEURAL="$HOME/venv/gen3d_neural/bin/python"
OUT=output/e12
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ROS_DOMAIN_ID=23 GZ_PARTITION=gen3d_e12
run() { nice -n 15 env -u PYTHONPATH "$PY" "$@"; }
for S in "$@"; do
  D=$OUT/$M/seed_$S
  [ -s "$D/clean.json" ] || nice -n 15 env -u PYTHONPATH "$NEURAL" scripts/e12_clean_meshes.py --model "$M" --seed "$S" --workers "${BUILD_WORKERS:-6}" > "$D/clean.log" 2>&1
  [ -s "$D/eval_manifest.json" ] || run scripts/build_neural_manifest.py --model "$M" --seed "$S" --workers "${BUILD_WORKERS:-6}" > "$D/build.log" 2>&1
  echo "[e12 $M $S] build exit=$?"
done
run scripts/run_mujoco_eval.py --root "$OUT/$M" --seeds "$@" --collision coacd --workers "${MJ_WORKERS:-8}" > "$OUT/$M/mujoco.log" 2>&1
echo "[e12 $M] mujoco exit=$?"
set +u; source /opt/ros/jazzy/setup.bash; set -u
for S in "$@"; do
  D=$OUT/$M/seed_$S
  [ -s "$D/gazebo_stability.json" ] || bash scripts/run_drop_test_chunked.sh "$D/eval_manifest.json" "$D/gazebo_stability.json" 25 "${GZ_WORKERS:-4}" > "$D/gazebo_drop.log" 2>&1
  [ -s "$D/gazebo_lift.json" ] || python3 scripts/gazebo_lift_eval.py --manifest "$D/eval_manifest.json" --mujoco "$D/mujoco_eval_coacd.json" --out "$D/gazebo_lift.json" --workers "${GZ_WORKERS:-4}" > "$D/gazebo_lift.log" 2>&1
  echo "[e12 $M $S] gazebo exit=$?"
done
echo "E12_${M}_DONE"
