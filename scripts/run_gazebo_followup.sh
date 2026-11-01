#!/bin/bash
# Gazebo drop tests after the physics-aware study:
#   1. E7 objects (output/physics_aware): plain CEM + Fixed CAD, then CEM from each
#      physics-aware config, on the seeds given.
#   2. Hulled vs CoACD-decomposed concave collision on identical objects, seeds 42-51.
# Usage (from 3d_generator/, ros2_ws built): bash scripts/run_gazebo_followup.sh 52 53 ...
set -u
cd "$(dirname "$0")/.."
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-23}" GZ_PARTITION="${GZ_PARTITION:-gen3d}"
PY="${PYTHON:-$HOME/venv/3d_cem/bin/python}"
W="${GZ_WORKERS:-4}"
AWARE="${AWARE:-output/physics_aware}"
PAIR="${PAIR:-output/gazebo_coacd}"
for S in "$@"; do
  for CFG in plain select train train_select; do
    D="$AWARE/$CFG/seed_$S"; [ -s "$D/gazebo_stability.json" ] && continue
    M="$D/eval_manifest.json"
    if [ "$CFG" != plain ]; then
      env -u PYTHONPATH "$PY" -c "import json,sys;json.dump([e for e in json.load(open(sys.argv[1])) if e['method']=='cem'],open(sys.argv[2],'w'))" "$M" "$D/manifest_cem.json"
      M="$D/manifest_cem.json"
    fi
    bash scripts/run_drop_test_chunked.sh "$M" "$D/gazebo_stability.json" 25 "$W" > "$D/gazebo.log" 2>&1
    echo "[aware $CFG $S] exit=$?"
  done
done
for S in 42 43 44 45 46 47 48 49 50 51; do
  D="$PAIR/seed_$S"; mkdir -p "$D"
  [ -s "$D/manifest_hull.json" ] || nice -n 19 env -u PYTHONPATH OMP_NUM_THREADS=1 "$PY" \
      scripts/build_collision_pair_manifests.py --seed "$S" --out "$D" > "$D/build.log" 2>&1 &
done
wait
for S in 42 43 44 45 46 47 48 49 50 51; do
  D="$PAIR/seed_$S"
  for ARM in hull coacd; do
    [ -s "$D/gazebo_$ARM.json" ] && continue
    bash scripts/run_drop_test_chunked.sh "$D/manifest_$ARM.json" "$D/gazebo_$ARM.json" 25 "$W" > "$D/gazebo_$ARM.log" 2>&1
    echo "[pair $ARM $S] exit=$?"
  done
done
echo FOLLOWUP_DONE
