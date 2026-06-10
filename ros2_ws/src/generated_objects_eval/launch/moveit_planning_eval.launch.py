"""Launch the MoveIt 2 planning-success evaluator with the Panda config.

This builds the full moveit_config via MoveItConfigsBuilder, dumps the
parameters into a temp YAML, and invokes the evaluator binary with
``--params-file`` so MoveItPy reads the configuration through the ROS
parameter server (the supported path).

Usage::

    ros2 launch generated_objects_eval moveit_planning_eval.launch.py \\
        manifest:=/abs/path/manifest.json out:=/abs/path/results.json \\
        max_objects:=10
"""

import os
import tempfile
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from moveit_configs_utils import MoveItConfigsBuilder


def _build_params_file() -> str:
    # mock_components keeps the URDF parseable by the standard URDFXMLParser
    # (the alternative ros2_control hardware values inject elements that the
    # parser silently rejects, leaving the planning_scene_monitor unable to
    # configure itself).
    share = Path(get_package_share_directory("generated_objects_eval"))
    moveit_cpp_yaml = share / "config" / "moveit_cpp.yaml"
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
        .moveit_cpp(file_path=str(moveit_cpp_yaml))
        .to_moveit_configs()
    )
    cfg = moveit_config.to_dict()
    # MoveItPy creates two nodes that both need these parameters: the cpp
    # initializer with an auto-generated name (reads planning_pipelines) and
    # the wrapper node named via the node_name= kwarg passed to MoveItPy()
    # (reads robot_description).  Use the /** wildcard so the params bind
    # regardless of name.
    params = {"/**": {"ros__parameters": cfg}}
    fd, path = tempfile.mkstemp(prefix="moveit_eval_params_", suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(params, f, default_flow_style=False)
    return path


def _launch_setup(context):
    default_config = Path(get_package_share_directory("generated_objects_eval")) \
        / "config" / "eval_config.yaml"
    params_file = _build_params_file()

    manifest = LaunchConfiguration("manifest").perform(context)
    out = LaunchConfiguration("out").perform(context)
    max_objects = LaunchConfiguration("max_objects").perform(context)
    kinematic_only = LaunchConfiguration("kinematic_only").perform(context)

    exec_cmd = [
        "moveit_planning_eval",
        "--manifest", manifest,
        "--out", out,
        "--config", str(default_config),
        "--max-objects", str(max_objects),
    ]
    # collision-aware planning is the default; kinematic_only:=true reproduces
    # the as-submitted metric (no table / object collision objects in the scene).
    if str(kinematic_only).lower() in ("1", "true", "yes"):
        exec_cmd.append("--kinematic-only")
    exec_cmd += ["--ros-args", "--params-file", params_file]
    # robot_state_publisher + joint_state_publisher provide the /joint_states
    # topic that MoveIt's planning_scene_monitor blocks on at startup.
    from launch_ros.actions import Node
    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="log",
        parameters=[params_file],
    )
    # Publish a valid Panda home configuration; the stock
    # joint_state_publisher emits all zeros which puts the arm in
    # self-collision and blocks every plan via CheckStartStateCollision.
    jsp = ExecuteProcess(
        cmd=["home_joint_state_publisher"],
        output="log",
    )
    # The Panda SRDF declares a virtual_joint between "world" and "panda_link0";
    # without a TF for that pair the planning_scene_monitor logs "Missing
    # virtual_joint" every second.
    world_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_panda_link0",
        output="log",
        arguments=["--x", "0", "--y", "0", "--z", "0",
                   "--roll", "0", "--pitch", "0", "--yaw", "0",
                   "--frame-id", "world", "--child-frame-id", "panda_link0"],
    )
    return [rsp, jsp, world_tf, ExecuteProcess(cmd=exec_cmd, output="screen")]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("manifest"),
        DeclareLaunchArgument("out"),
        DeclareLaunchArgument("max_objects", default_value="0"),
        DeclareLaunchArgument("kinematic_only", default_value="false"),
        OpaqueFunction(function=_launch_setup),
    ])
