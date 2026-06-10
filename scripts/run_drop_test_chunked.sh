#!/usr/bin/env bash
# Robust, scalable Gazebo drop test. The gz physics engine (ODE) crashes on a small
# fraction of meshes (trimesh-trimesh collision assertion), which kills a long-lived
# world and — since the evaluator writes its JSON only at the end — loses everything.
# This runner processes the manifest in small CHUNKS, each in a FRESH gz world, across
# N isolated parallel workers (own GZ_PARTITION/ROS_DOMAIN_ID, killed by process-group
# between chunks). A crash loses only one chunk; every chunk is checkpointed; crashed
# chunks are retried once. Merge is by object name.
#
# Usage (from 3d_generator/, with ros2_ws built + sourced):
#   bash scripts/run_drop_test_chunked.sh <manifest.json> <out.json> [chunk=100] [workers=4]
set +e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MAN="${1:?manifest path required}"
OUT="${2:?output json path required}"
CHUNK="${3:-100}"
N="${4:-4}"
SLDIR="$(dirname "$OUT")/_chunks"
PY="${PYTHON:-$HOME/venv/3d_cem/bin/python}"
EVAL="$ROOT/ros2_ws/install/generated_objects_eval/bin/gazebo_stability_eval"
rm -rf "$SLDIR"; mkdir -p "$SLDIR"

NCH=$(env -u PYTHONPATH "$PY" - "$MAN" "$SLDIR" "$CHUNK" <<'PYEOF'
import json,sys,math
man,sldir,ch=sys.argv[1],sys.argv[2],int(sys.argv[3])
m=json.load(open(man)); nch=math.ceil(len(m)/ch)
for j in range(nch):
    json.dump(m[j*ch:(j+1)*ch], open(f"{sldir}/chunk_{j:04d}.json","w"))
print(nch)
PYEOF
)
echo "[drop] $(basename "$MAN"): $NCH chunks of $CHUNK across $N workers"

# Generous gz service timeouts (parallel worlds add transport latency); kept below the
# subprocess timeout so gz returns its own failure first. Abort a chunk after a few
# consecutive spawn failures (= the world died) instead of slow-failing the tail.
export GZ_SVC_TIMEOUT_MS="${GZ_SVC_TIMEOUT_MS:-15000}"
export GZ_SVC_SUBPROC_S="${GZ_SVC_SUBPROC_S:-20}"
export GZ_ABORT_AFTER_FAILS="${GZ_ABORT_AFTER_FAILS:-5}"
pkill -KILL -f "gz sim -s -r" 2>/dev/null; sleep 2

run_worker() {
  local wid=$1
  export GZ_PARTITION="drop${wid}" ROS_DOMAIN_ID=$((90+wid)) RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  local j=$wid
  while [ "$j" -lt "$NCH" ]; do
    local jj; jj=$(printf %04d "$j")
    local cf="$SLDIR/chunk_${jj}.json" of="$SLDIR/out_${jj}.json" nin nout=0
    nin=$(env -u PYTHONPATH "$PY" -c "import json;print(len(json.load(open('$cf'))))")
    for att in 1 2; do
      setsid bash -c "exec ros2 launch generated_objects_eval stability_world.launch.py" \
          > "$SLDIR/world_${wid}.log" 2>&1 &
      local wpid=$!
      sleep 9
      timeout 900 "$EVAL" --manifest "$ROOT/$cf" --out "$ROOT/$of" --max-objects 0 \
          > "$SLDIR/eval_${wid}.log" 2>&1
      kill -- -"$wpid" 2>/dev/null; sleep 1; kill -9 -- -"$wpid" 2>/dev/null; sleep 1
      nout=$(env -u PYTHONPATH "$PY" -c "import json,os;p='$of';print(len(json.load(open(p)).get('results',[])) if os.path.exists(p) else 0)" 2>/dev/null)
      [ "${nout:-0}" -ge "$nin" ] && break
    done
    echo "[drop] worker $wid chunk $j: ${nout:-0}/$nin"
    j=$((j + N))
  done
}

for wid in $(seq 0 $((N-1))); do run_worker "$wid" & done
wait
pkill -KILL -f "gz sim -s -r" 2>/dev/null

env -u PYTHONPATH "$PY" - "$SLDIR" "$OUT" <<'PYEOF'
import json,glob,sys
sldir,out=sys.argv[1],sys.argv[2]
seen={}
for p in sorted(glob.glob(f"{sldir}/out_*.json")):
    try:
        for r in json.load(open(p)).get("results",[]): seen[r.get("name")]=r
    except Exception: pass
json.dump({"results":list(seen.values())}, open(out,"w"))
print(f"[drop] merged {len(seen)} results -> {out}")
PYEOF
echo "[drop] DONE"
