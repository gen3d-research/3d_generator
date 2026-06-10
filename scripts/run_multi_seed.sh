#!/bin/bash
# Run the full evaluation pipeline (manifest build, unified Python eval,
# MoveIt 2 planning eval, Gazebo dynamic-stability eval) for one or more
# random seeds. Outputs land in 3d_generator/output/seed_<seed>/.
#
# Usage:
#   scripts/run_multi_seed.sh 42 43 44

# Intentionally no `set -e`: the moveit_py shutdown can emit a SIGSEGV
# *after* the JSON has been written, which we want to treat as a clean
# completion for this orchestration script.

WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
G_ROOT="$WS_ROOT/3d_generator"
# The ROS 2 workspace lives INSIDE the repo (3d_generator/ros2_ws), not as a
# sibling of it. The old "$WS_ROOT/ros2_ws" pointed at a nonexistent directory,
# so the source below failed silently and the Gazebo stage's eval-binary path
# (stage 4) could not resolve at all.
ROS_WS="$G_ROOT/ros2_ws"

# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source "$ROS_WS/install/setup.bash"

cleanup() {
    # gz transport advertises services by world name; stale gz_sim instances
    # from previous runs steal entity-factory calls non-deterministically.
    # PRECISE pattern: the broad "gz sim" matches any process whose command
    # line merely contains the string (other sessions, even shells).
    pkill -KILL -f "gz sim -s -r" 2>/dev/null
    pkill -KILL -f "gazebo_stability_eval" 2>/dev/null
    pkill -KILL -f "moveit_planning_eval" 2>/dev/null
    pkill -KILL -f "home_joint_state_publisher" 2>/dev/null
    pkill -KILL -f "robot_state_publisher" 2>/dev/null
    pkill -KILL -f "ros2 launch.*stability_world" 2>/dev/null
    pkill -KILL -f "ros2 launch.*moveit_planning_eval" 2>/dev/null
    sleep 2
}

for SEED in "$@"; do
    SEED_DIR="$G_ROOT/output/seed_${SEED}"
    mkdir -p "$SEED_DIR"
    MANIFEST="$SEED_DIR/eval_manifest.json"
    UNIFIED="$SEED_DIR/unified_eval.json"
    MOVEIT="$SEED_DIR/moveit_results.json"
    GAZEBO="$SEED_DIR/gazebo_stability.json"

    cleanup

    echo "============================================================"
    echo "  Seed $SEED  ->  $SEED_DIR"
    echo "============================================================"

    if [ ! -s "$MANIFEST" ]; then
        echo "[$SEED] building manifest"
        python3 "$G_ROOT/scripts/build_eval_manifest.py" \
            --budget 1500 --top-k 25 --seed "$SEED" --out "$MANIFEST" \
            --export-root "$SEED_DIR/manifest_objects"
        python3 "$G_ROOT/scripts/patch_sdf_collision.py" --manifest "$MANIFEST"
    fi

    if [ ! -s "$UNIFIED" ]; then
        echo "[$SEED] unified Python eval"
        python3 "$G_ROOT/scripts/run_unified_eval.py" \
            --budget 1500 --top-k 100 --seed "$SEED" --out "$UNIFIED"
    fi

    if [ ! -s "$MOVEIT" ]; then
        echo "[$SEED] MoveIt 2 planning eval"
        # The moveit_py shutdown segfaults *after* the JSON is written, but
        # the ros2 launch parent waits 15 min for the orphan children to die.
        # Spawn the launch, poll for the result file, then kill the launch.
        ros2 launch generated_objects_eval moveit_planning_eval.launch.py \
            manifest:="$MANIFEST" out:="$MOVEIT" max_objects:=0 \
            > "$SEED_DIR/moveit.log" 2>&1 &
        LAUNCH_PID=$!
        for _ in $(seq 1 600); do
            sleep 1
            if [ -s "$MOVEIT" ]; then
                # Give it a few more seconds to flush the per-grasp records.
                sleep 4
                break
            fi
            if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
                break
            fi
        done
        pkill -KILL -f "moveit_planning_eval" 2>/dev/null
        pkill -KILL -f "home_joint_state_publisher" 2>/dev/null
        pkill -KILL -f "robot_state_publisher" 2>/dev/null
        pkill -KILL -P "$LAUNCH_PID" 2>/dev/null
        kill -KILL "$LAUNCH_PID" 2>/dev/null
        wait "$LAUNCH_PID" 2>/dev/null
        sleep 2
    fi

    if [ ! -s "$GAZEBO" ]; then
        echo "[$SEED] Gazebo stability eval"
        ros2 launch generated_objects_eval stability_world.launch.py \
            > "$SEED_DIR/gz_world.log" 2>&1 &
        sleep 6
        timeout 900 "$ROS_WS/install/generated_objects_eval/bin/gazebo_stability_eval" \
            --manifest "$MANIFEST" --out "$GAZEBO" --max-objects 0 \
            > "$SEED_DIR/gazebo.log" 2>&1 || true
        pkill -f "ros2 launch.*stability" 2>/dev/null || true
        pkill -f "gz sim -s -r" 2>/dev/null || true
        sleep 2
    fi

    echo "[$SEED] done"
done

echo
echo "All seeds finished. Aggregate with:"
echo "  python3 $G_ROOT/scripts/aggregate_seeds.py $*"
