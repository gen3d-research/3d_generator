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

    params_file, moveit_config = _build_params_file()

    gz = ExecuteProcess(
        cmd=["gz", "sim", "-r", "--render-engine", "ogre2", str(world)],
        output="screen",
    )
    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="log",
        parameters=[moveit_config.robot_description],
    )
    # The demo driver publishes /joint_states itself (replaying each planned
    # trajectory at its native time scale so the Panda animates in RViz).
    # RViz still needs a world -> panda_link0 TF for the RobotModel display.
    world_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_panda_link0",
        output="log",
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
        ],
    )
    driver_cmd = [
        "demo_plan_driver",
        "--manifest", manifest,
        "--method", method,
    ]
    if loop:
        driver_cmd.append("--loop")
    driver_cmd += ["--ros-args", "--params-file", params_file]
    driver = ExecuteProcess(cmd=driver_cmd, output="screen")

    return [gz, rsp, world_tf, rviz, driver]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("manifest"),
        DeclareLaunchArgument("method", default_value="cem",
                              description="Which method to pull an object from"),
        DeclareLaunchArgument("loop", default_value="true",
                              description="Loop the plan driver indefinitely "
                                          "for screen-recording"),
        OpaqueFunction(function=_launch_setup),
    ])
