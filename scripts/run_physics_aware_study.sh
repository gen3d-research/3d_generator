#!/bin/bash
# Physics-aware selection study (ROADMAP E7): does the planner's table check
# plus the learned grasp-success predictor raise physical lift success?
#
# Per seed, three manifest configs for CEM and Fixed CAD:
#   plain   the published pipeline (planner order, no table check, no pool)
#   grasps  --table-check + --grasp-model     (re-rank each object's grasps)
#   select  grasps + --cem-pool POOL          (CEM keeps the top-k of POOL x k
#                                              objects by best predicted grasp)
# then MuJoCo pick-and-lift on the true geometry (CoACD) under nominal physics
# and four perturbations (mu 0.3 / 1.0, grip 10 / 40 N), so the selection is
# not graded only under the physics its predictor was trained on. Train the
# predictor on objects disjoint from these seeds (e.g. the archetype set).
#
# Usage (from 3d_generator/):
#   MODEL=output/grasp_predictor/arch_object_s0/model.npz \
#   OUT=output/physics_aware bash scripts/run_physics_aware_study.sh 52 53 54
set -u
cd "$(dirname "$0")/.."
MODEL="${MODEL:?set MODEL=path/to/model.npz}"
OUT="${OUT:-output/physics_aware}"
POOL="${POOL:-4}"
PY="${PYTHON:-$HOME/venv/3d_cem/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run() { nice -n "${NICE:-19}" env -u PYTHONPATH "$PY" "$@"; }

for S in "$@"; do
  for CFG in plain grasps select; do
    D="$OUT/$CFG/seed_$S"; mkdir -p "$D"
    [ -s "$D/eval_manifest.json" ] && continue
    case $CFG in
      plain)  EXTRA=() ;;
      grasps) EXTRA=(--table-check --grasp-model "$MODEL") ;;
      select) EXTRA=(--table-check --grasp-model "$MODEL" --cem-pool "$POOL") ;;
    esac
    ( run scripts/build_eval_manifest.py --budget 1500 --top-k 25 --seed "$S" \
          --methods cem fixed_cad --out "$D/eval_manifest.json" \
          --export-root "$D/manifest_objects" "${EXTRA[@]}" > "$D/build.log" 2>&1
      echo "[$CFG seed $S] manifest exit=$?" ) &
  done
done
wait

for CFG in plain grasps select; do
  run scripts/run_mujoco_eval.py --root "$OUT/$CFG" --seeds "$@" --collision coacd \
      --workers "${WORKERS:-12}" > "$OUT/$CFG/mujoco_nominal.log" 2>&1
  for P in "mu 0.3" "mu 1.0" "grip-force-n 10.0" "grip-force-n 40.0"; do
    read -r K V <<< "$P"
    run scripts/run_mujoco_eval.py --root "$OUT/$CFG" --seeds "$@" --collision coacd \
        --"$K" "$V" --tag "coacd_${K%%-*}${V}" --workers "${WORKERS:-12}" \
        > "$OUT/$CFG/mujoco_${K%%-*}${V}.log" 2>&1
  done
done
echo STUDY_DONE
