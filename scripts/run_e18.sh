#!/bin/bash
# E18 (ROADMAP §6, design of 16 Nov 2026): choose among k neural meshes per prompt with the lift predictor.
#   setup MODEL   E12 prompts; candidate 0 = the E12 cleaned mesh (clean.json copied, meshes not)
#   gen MODEL     candidates 1..K-1 on the GPU, one flock ~/.gpu.lock per candidate
#   clean MODEL   repair candidates 1..K-1 (same cleaning as E12)
#   build MODEL   every candidate: planner with the table check, grasps ordered by the predictor, CoACD export
#   select MODEL  per prompt the candidate with the highest best-grasp prediction -> <model>_sel
#   grade MODEL   arms B (<model>_c0) and C (<model>_sel): MuJoCo drop+lift; Gazebo lift (and drop for C;
#                 B's mesh is A's, whose Gazebo drop E12 already has)
# Usage (from 3d_generator/): bash scripts/run_e18.sh STAGE shap_e|trellis
set -u
cd "$(dirname "$0")/.."
STAGE=$1; M=$2
K=${K:-4}
SEEDS="${SEEDS:-65 66 67 68 69 70 71 72 73 74}"
OUT=output/e18
PY="$HOME/venv/3d_cem/bin/python"
NEURAL="$HOME/venv/gen3d_neural/bin/python"
MODEL_NPZ=output/grasp_predictor/arch_object_s0/model.npz
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run() { nice -n 15 env -u PYTHONPATH "$PY" "$@"; }

case $STAGE in
setup)
  mkdir -p $OUT/prompts
  cp --update=none output/e12/prompts/seed_*.json $OUT/prompts/
  for S in $SEEDS; do
    D=$OUT/${M}_c0/seed_$S; mkdir -p "$D"
    cp --update=none output/e12/$M/seed_$S/clean.json "$D/clean.json"
  done ;;
gen)
  # Shap-E (2.5 GB, about 10 min per candidate) is under the lock's thresholds; NOLOCK=1 skips the lock for it.
  LOCK="flock $HOME/.gpu.lock"; [ "${NOLOCK:-0}" = 1 ] && LOCK=""
  for J in $(seq 1 $((K - 1))); do
    PYTHONPATH=output/e12/src/stubs $LOCK nice -n 10 "$NEURAL" scripts/e12_generate.py --model $M \
      --seeds $SEEDS --root $OUT --candidate $J >> $OUT/gen_${M}_c$J.log 2>&1
    echo "[e18 gen $M c$J] exit=$?"
  done ;;
clean)
  for J in $(seq 1 $((K - 1))); do
    for S in $SEEDS; do
      D=$OUT/${M}_c$J/seed_$S
      [ -s "$D/clean.json" ] || nice -n 15 env -u PYTHONPATH "$NEURAL" scripts/e12_clean_meshes.py --model $M \
        --seed $S --root $OUT --candidate $J --workers "${WORKERS:-8}" > "$D/clean.log" 2>&1
      echo "[e18 clean $M c$J $S] exit=$?"
    done
  done ;;
build)
  for J in $(seq 0 $((K - 1))); do
    for S in $SEEDS; do
      D=$OUT/${M}_c$J/seed_$S
      [ -s "$D/objects.json" ] || run scripts/build_neural_manifest.py --model ${M}_c$J --seed $S --root $OUT \
        --table-check --grasp-model $MODEL_NPZ --workers "${WORKERS:-8}" > "$D/build.log" 2>&1
      echo "[e18 build $M c$J $S] exit=$?"
    done
  done ;;
select)
  run scripts/e18_select.py --model $M --seeds $SEEDS --root $OUT --k $K ;;
grade)
  for A in ${M}_c0 ${M}_sel; do
    run scripts/run_mujoco_eval.py --root $OUT/$A --seeds $SEEDS --collision coacd \
      --workers "${MJ_WORKERS:-8}" > $OUT/$A/mujoco.log 2>&1
    echo "[e18 $A] mujoco exit=$?"
  done
  export ROS_DOMAIN_ID=23 GZ_PARTITION=gen3d
  set +u; source /opt/ros/jazzy/setup.bash; set -u
  for A in ${M}_c0 ${M}_sel; do
    for S in $SEEDS; do
      D=$OUT/$A/seed_$S
      if [ "$A" = "${M}_sel" ] && [ ! -s "$D/gazebo_stability.json" ]; then
        bash scripts/run_drop_test_chunked.sh "$D/eval_manifest.json" "$D/gazebo_stability.json" 25 \
          "${GZ_WORKERS:-4}" > "$D/gazebo_drop.log" 2>&1
      fi
      [ -s "$D/gazebo_lift.json" ] || python3 scripts/gazebo_lift_eval.py --manifest "$D/eval_manifest.json" \
        --mujoco "$D/mujoco_eval_coacd.json" --out "$D/gazebo_lift.json" --workers "${GZ_WORKERS:-4}" \
        > "$D/gazebo_lift.log" 2>&1
      echo "[e18 $A $S] gazebo exit=$?"
    done
  done ;;
*) echo "usage: $0 setup|gen|clean|build|select|grade shap_e|trellis"; exit 2 ;;
esac
