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
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def _build_params_file() -> str:
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
    method = LaunchConfiguration("method").perform(context)
    loop = LaunchConfiguration("loop").perform(context).lower() in ("1", "true", "yes")
    render_engine = LaunchConfiguration("render_engine").perform(context)
    headless = LaunchConfiguration("headless").perform(context).lower() in ("1", "true", "yes")

    params_file, moveit_config = _build_params_file()

    gz_cmd = ["gz", "sim", "-r"]
    if headless:
        gz_cmd.append("-s")
    gz_cmd += ["--render-engine", render_engine, str(world)]
    gz = ExecuteProcess(cmd=gz_cmd, output="screen")
    # Force wall-clock time everywhere so the TF chain published by
    # static_transform_publisher (wall-clock) and the per-frame transforms
    # published by robot_state_publisher (driven by /joint_states stamps,
    # also wall-clock) line up with what RViz looks up.  Without this,
    # RViz auto-detects gz_sim's /clock and switches to sim time, the TF
    # buffer has nothing for the wall-clock stamp it queries with, and the
    # RobotModel display silently fails to position any link.
    use_sim_time = {"use_sim_time": False}

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="log",
        parameters=[moveit_config.robot_description, use_sim_time],
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
    # Pass --ros-args --params-file <yaml> AND -p use_sim_time:=false so the
    # first /joint_states message stamped by DemoNode has a wall-clock time
    # (not the sim-time 0.0 gz_sim publishes on /clock).  Without this the
    # MoveItPy planning_scene_monitor's wait_for_initial_state_timeout fires
    # because the first joint state has stamp 0 -> driver crashes with
    # "Unable to configure planning scene monitor".
    driver_cmd += [
        "--ros-args",
        "--params-file", params_file,
        "-p", "use_sim_time:=false",
    ]
    driver = ExecuteProcess(cmd=driver_cmd, output="screen")

    actions = [gz, rsp, world_tf, rviz, driver]

    # Optional full gz_ros2_control integration: spawn the Panda model into
    # the running gz world, bridge /clock, and bring up the controllers.
    use_gz_control = LaunchConfiguration("use_gz_control").perform(context).lower() in (
        "1", "true", "yes")
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
        # The controller_manager is spawned by the gz_ros2_control plugin
        # baked into the URDF.  Wait a few seconds for it to be reachable
        # before launching the controller spawners.
        controllers = [
            TimerAction(period=5.0, actions=[Node(
                package="controller_manager", executable="spawner",
                arguments=["joint_state_broadcaster",
                           "--controller-manager", "/controller_manager"],
                output="screen",
            )]),
            TimerAction(period=6.0, actions=[Node(
                package="controller_manager", executable="spawner",
                arguments=["panda_arm_controller",
                           "--controller-manager", "/controller_manager"],
                output="screen",
            )]),
            TimerAction(period=7.0, actions=[Node(
                package="controller_manager", executable="spawner",
                arguments=["panda_hand_controller",
                           "--controller-manager", "/controller_manager"],
                output="screen",
            )]),
        ]
        actions += [spawn, clock_bridge] + controllers
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("manifest"),
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
