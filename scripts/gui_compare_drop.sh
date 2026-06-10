#!/usr/bin/env bash
# Side-by-side GUI drop comparison — spawn N objects in a row in one Gazebo window and
# drop them together, on a loop, so you can record the comparison. Reusable for any
# objects (raw vs optimized, archetype vs generated, method-A vs method-B, ...).
#
# Usage (from 3d_generator/, ros2_ws built):
#   bash scripts/gui_compare_drop.sh /abs/a.sdf=LABEL_A /abs/b.sdf=LABEL_B [/abs/c.sdf=LABEL_C ...]
# Env knobs:
#   DISPLAY (default :1)  DROP_M (drop height, 0.15)  SETTLE_S (5)  LOOPS (60)  SPACING_M (0.18)
set +e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
WORLD=panda_eval_world
DISPLAY="${DISPLAY:-:1}"; export DISPLAY
DROP_M="${DROP_M:-0.15}"; SETTLE_S="${SETTLE_S:-5}"; LOOPS="${LOOPS:-60}"; SPACING_M="${SPACING_M:-0.18}"
LOG="${LOG_DIR:-$ROOT/output/compare}"; mkdir -p "$LOG"

source "${ROS_SETUP:-/opt/ros/jazzy/setup.bash}" 2>/dev/null
source "$ROOT/ros2_ws/install/setup.bash" 2>/dev/null
# Isolated partition/domain so this never collides with other running sims.
export GZ_PARTITION="${GZ_PARTITION:-cmp}" ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-86}" RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Parse "sdf=label" args, lay them out centered along y.
SDFS=(); LABELS=()
for a in "$@"; do SDFS+=("${a%%=*}"); LABELS+=("${a##*=}"); done
NN=${#SDFS[@]}
[ "$NN" -lt 1 ] && { echo "give at least one sdf=label"; exit 1; }
TABLE_Z=0.40; Z=$(awk "BEGIN{print $TABLE_Z + $DROP_M}")

create() { # sdf name y
  gz service -s "/world/$WORLD/create" --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean \
    --timeout 15000 --req "sdf_filename: \"$1\", name: \"$2\", pose: {position: {x: 0.5, y: $3, z: $Z}, orientation: {w: 1.0}}" >/dev/null 2>&1
}
remove() { gz service -s "/world/$WORLD/remove" --reqtype gz.msgs.Entity --reptype gz.msgs.Boolean --timeout 8000 --req "name: \"$1\", type: MODEL" >/dev/null 2>&1; }

echo "[cmp] launching GUI world on DISPLAY=$DISPLAY ($NN objects, drop ${DROP_M}m) ..."
setsid bash -c "exec ros2 launch generated_objects_eval stability_world_gui.launch.py" > "$LOG/cmp_world.log" 2>&1 &
WPID=$!
sleep 18

echo "[cmp] looping drops — Ctrl-C / kill to stop. Labels: ${LABELS[*]}"
for ((loop=0; loop<LOOPS; loop++)); do
  for ((i=0; i<NN; i++)); do
    y=$(awk "BEGIN{print ($i - ($NN-1)/2.0) * $SPACING_M}")
    create "${SDFS[$i]}" "${LABELS[$i]}" "$y"
  done
  sleep "$SETTLE_S"
  for ((i=0; i<NN; i++)); do remove "${LABELS[$i]}"; done
  sleep 1
done

kill -- -"$WPID" 2>/dev/null
echo "[cmp] DONE"
