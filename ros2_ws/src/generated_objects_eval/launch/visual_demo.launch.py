"""Full visual demo for the project-page videos.

Brings up, in order:
  1. ``gz sim`` *with* the GUI on the empty panda_eval_world (the user can
     screen-record this window).
  2. ``robot_state_publisher`` + ``home_joint_state_publisher`` so the Panda
     description is on the bus.
  3. RViz2 with the bundled ``demo.rviz`` config: RobotModel + planning scene
     + Trajectory display + MotionPlanning panel.
  4. ``demo_plan_driver`` which spawns one generated object in gz_sim and
     loops MoveIt 2 plans to its grasp candidates, so RViz visualises the
     trajectory continuously.

Usage::

    ros2 launch generated_objects_eval visual_demo.launch.py \\
        manifest:=/abs/path/eval_manifest.json method:=cem

Recording: a separate ``recording.md`` in this package documents three
options (ffmpeg+x11grab, OBS, gz sim's video-recorder GUI button).
"""

import os
import tempfile
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            OpaqueFunction, SetEnvironmentVariable)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def _ros_plugin_path() -> str:
    ros_lib = "/opt/ros/jazzy/lib"
    existing = os.environ.get("GZ_SIM_SYSTEM_PLUGIN_PATH", "")
    if existing and ros_lib not in existing.split(os.pathsep):
        return ros_lib + os.pathsep + existing
    return ros_lib if not existing else existing


def _default_manifest() -> str:
    """Find a sensible default manifest so the demo runs with no args.

    Looks for the seed_42 manifest under the project's ``output/`` tree
    (relative to the workspace root that hosts this ros2_ws), then any
    ``output/**/eval_manifest.json``, then ``""`` if nothing is found.
    The caller is expected to ``DeclareLaunchArgument("manifest",
    default_value=_default_manifest())`` and surface a friendly error
    when the path is empty.
    """
    candidates = [
        Path("/home/asmbatati/text2geometry_ws/3d_generator/output/seed_42/eval_manifest.json"),
        Path("/home/asmbatati/text2geometry_ws/3d_generator/output/eval_manifest.json"),
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    # Last-ditch glob.
    root = Path("/home/asmbatati/text2geometry_ws/3d_generator/output")
    if root.is_dir():
        for m in root.rglob("eval_manifest.json"):
            return str(m)
    return ""


def _build_params_file(use_sim_time: bool = False) -> str:
    moveit_config = (
        MoveItConfigsBuilder("moveit_resources_panda")
        .robot_description(
            file_path="config/panda.urdf.xacro",
            mappings={"ros2_control_hardware_type": "mock_components"},
        )
        .robot_description_semantic(file_path="config/panda.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .moveit_cpp(file_path=str(
            Path(get_package_share_directory("generated_objects_eval"))
            / "config" / "moveit_cpp.yaml"))
        .to_moveit_configs()
    )
    cfg = moveit_config.to_dict()
    # MoveItPy spawns its own internal node which doesn't inherit the
    # launching process's ``-p use_sim_time:=...`` flag, so we have to
    # bake the clock domain into the params YAML under the /** wildcard.
    cfg["use_sim_time"] = use_sim_time
    # Loosen the execution start-state tolerance. Even when the driver plans
    # from the current state, the controller's reported state lags by a few
    # millirad, and MoveIt's 0.01 rad default rejects every trajectory with
    # "start point deviates from current robot state" — so the arm never moves
    # in physics. 0.1 rad removes the spurious rejections.
    cfg["trajectory_execution.allowed_start_tolerance"] = 0.1
    if isinstance(cfg.get("trajectory_execution"), dict):
        cfg["trajectory_execution"]["allowed_start_tolerance"] = 0.1
    params = {"/**": {"ros__parameters": cfg}}
    fd, path = tempfile.mkstemp(prefix="visual_demo_params_", suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(params, f, default_flow_style=False)
    return path, moveit_config


def _launch_setup(context):
    share = Path(get_package_share_directory("generated_objects_eval"))
    world = share / "worlds" / "panda_eval_world.sdf"
    rviz_cfg = share / "config" / "demo.rviz"

    manifest = LaunchConfiguration("manifest").perform(context)
    if not manifest or not Path(manifest).is_file():
        raise RuntimeError(
            f"manifest path '{manifest}' does not exist. Pass an absolute "
            "path to an eval_manifest.json, e.g.\n"
            "  ros2 launch generated_objects_eval visual_demo.launch.py \\\n"
            "      manifest:=/abs/path/output/seed_42/eval_manifest.json")
    method = LaunchConfiguration("method").perform(context)
    loop = LaunchConfiguration("loop").perform(context).lower() in ("1", "true", "yes")
    render_engine = LaunchConfiguration("render_engine").perform(context)
    headless = LaunchConfiguration("headless").perform(context).lower() in ("1", "true", "yes")
    use_gz_control = LaunchConfiguration("use_gz_control").perform(context).lower() in (
        "1", "true", "yes")

    params_file, moveit_config = _build_params_file(use_sim_time=use_gz_control)

    # When wiring up gz_ros2_control we need the augmented URDF (world
    # anchor + inertials + ros2_control + gazebo plugin block) on
    # /robot_description so both ros_gz_sim create and the gz plugin see
    # it.  In RViz-only mode we keep the lighter MoveIt URDF.
    if use_gz_control:
        from generated_objects_eval.build_panda_urdf import build_panda_urdf
        rsp_robot_description = {"robot_description": build_panda_urdf()}
    else:
        rsp_robot_description = moveit_config.robot_description

    gz_cmd = ["gz", "sim", "-r"]
    if headless:
        gz_cmd.append("-s")
    gz_cmd += ["--render-engine", render_engine, str(world)]
    gz = ExecuteProcess(cmd=gz_cmd, output="screen")
    # Clock domain depends on the integration mode:
    #
    # * RViz-only animation (use_gz_control=false): wall-clock everywhere.
    #   DemoNode publishes /joint_states stamped from time.time(),
    #   robot_state_publisher and static_transform_publisher echo those
    #   stamps onto /tf, RViz looks them up at wall-clock now().  If we
    #   let RViz auto-switch to sim time here the TF buffer has nothing
    #   to match and the RobotModel silently fails to position any link.
    #
    # * gz_ros2_control mode (use_gz_control=true): sim-time everywhere.
    #   joint_state_broadcaster inside gz publishes /joint_states with
    #   /clock stamps (starting at 0 and lagging wall-clock by ~10 s
    #   while gz_sim boots).  MoveItPy's planning_scene_monitor compares
    #   the latest /joint_states stamp against its node clock at
    #   construction; if those clocks disagree it FATALs out with
    #   "Unable to configure planning scene monitor" -- which is exactly
    #   what happened before this switch was conditional.
    use_sim_time = {"use_sim_time": use_gz_control}

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="log",
        parameters=[rsp_robot_description, use_sim_time],
    )
    # RViz needs a world -> panda_link0 TF for the RobotModel display.
    world_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_panda_link0",
        output="log",
        parameters=[use_sim_time],
        arguments=["--x", "0", "--y", "0", "--z", "0",
                   "--roll", "0", "--pitch", "0", "--yaw", "0",
                   "--frame-id", "world", "--child-frame-id", "panda_link0"],
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_demo",
        output="screen",
        arguments=["-d", str(rviz_cfg)],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            use_sim_time,
        ],
    )
    driver_cmd = [
        "demo_plan_driver",
        "--manifest", manifest,
        "--method", method,
    ]
    if loop:
        driver_cmd.append("--loop")
    # When the gz_ros2_control stack is wired in, hand each plan to
    # MoveItPy.execute() so the FollowJointTrajectory controller drives
    # the Panda in physics.  Without --execute the driver only animates
    # the plan in its own /joint_states publisher, which is the right
    # mode for RViz-only recording but does nothing in Gazebo.
    if use_gz_control:
        driver_cmd.append("--execute")
    # Match the driver's clock to the rest of the stack (see the
    # use_sim_time comment above).  In gz_ros2_control mode this also
    # gives the planning_scene_monitor a /clock-stamped /joint_states
    # message to anchor on; in RViz-only mode it forces wall-clock so
    # DemoNode's first /joint_states is not stamped with sim-time 0.
    driver_cmd += [
        "--ros-args",
        "--params-file", params_file,
        "-p", f"use_sim_time:={'true' if use_gz_control else 'false'}",
    ]
    driver = ExecuteProcess(cmd=driver_cmd, output="screen")

    # In gz_ros2_control mode the driver MUST start AFTER joint_state_broadcaster
    # is publishing /joint_states with sim-time stamps, otherwise MoveItPy's
    # planning_scene_monitor fails its "recent joint state" check and the
    # whole driver crashes during MoveItPy() construction.
    if use_gz_control:
        from launch.actions import TimerAction
        actions = [gz, rsp, world_tf, rviz, TimerAction(period=15.0, actions=[driver])]
    else:
        actions = [gz, rsp, world_tf, rviz, driver]

    if use_gz_control:
        from launch.actions import TimerAction
        spawn = Node(
            package="ros_gz_sim",
            executable="create",
            output="screen",
            arguments=[
                "-name", "panda",
                "-topic", "/robot_description",
                "-x", "0", "-y", "0", "-z", "0",
            ],
        )
        clock_bridge = Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
            output="log",
        )
        # gz_ros2_control's controller_manager comes up inside gz_sim
        # several seconds after the model spawns; the spawners need both
        # a generous TimerAction delay and an explicit
        # --controller-manager-timeout so they don't FATAL out before
        # the service appears.
        def _spawner(controller):
            return Node(
                package="controller_manager", executable="spawner",
                arguments=[
                    controller,
                    "--controller-manager", "/controller_manager",
                    "--controller-manager-timeout", "60",
                ],
                output="screen",
                parameters=[use_sim_time],
            )
        controllers = [
            TimerAction(period=8.0, actions=[_spawner("joint_state_broadcaster")]),
            TimerAction(period=10.0, actions=[_spawner("panda_arm_controller")]),
            TimerAction(period=12.0, actions=[_spawner("panda_hand_controller")]),
        ]
        actions += [spawn, clock_bridge] + controllers
    return actions


def generate_launch_description():
    return LaunchDescription([
        # gz_sim only searches GZ_SIM_SYSTEM_PLUGIN_PATH for its system
        # plugins; the ROS plugin libs (incl. gz_ros2_control-system) live
        # in /opt/ros/jazzy/lib, so prepend that.  rmw_zenoh_cpp (Jazzy
        # default) drops some controller_manager services intermittently,
        # so pin everything this launch starts to rmw_fastrtps_cpp.
        SetEnvironmentVariable("GZ_SIM_SYSTEM_PLUGIN_PATH", _ros_plugin_path()),
        SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
        DeclareLaunchArgument(
            "manifest",
            default_value=_default_manifest(),
            description="Absolute path to an eval_manifest.json. Defaults "
                        "to output/seed_42/eval_manifest.json under the "
                        "project root if present."),
        DeclareLaunchArgument("method", default_value="cem",
                              description="Which method to pull an object from"),
        DeclareLaunchArgument("loop", default_value="true",
                              description="Loop the plan driver indefinitely "
                                          "for screen-recording"),
        DeclareLaunchArgument("render_engine", default_value="ogre2",
                              description="gz_sim render engine: ogre2 (default) "
                                          "or ogre (use ogre on NVIDIA hybrid-"
                                          "graphics laptops where EGL fails)"),
        DeclareLaunchArgument("headless", default_value="false",
                              description="Run gz_sim server-only (no window); "
                                          "useful when only RViz is being recorded"),
        DeclareLaunchArgument("use_gz_control", default_value="false",
                              description="If true, spawn the Panda in gz_sim "
                                          "with gz_ros2_control plus joint_state_broadcaster "
                                          "+ panda_arm_controller + panda_hand_controller, "
                                          "so trajectories execute in physics. Default false "
                                          "(RViz-only animation)."),
        OpaqueFunction(function=_launch_setup),
    ])
