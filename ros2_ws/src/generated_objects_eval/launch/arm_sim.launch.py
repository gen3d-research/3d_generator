"""Minimal Panda-in-RViz launch.

No MoveIt, no Gazebo, no animation --- just the bare components needed for
RViz to draw the Panda's RobotModel:

    robot_state_publisher       : URDF on /robot_description + /tf
    home_joint_state_publisher  : Panda 'ready' pose on /joint_states
    static_transform_publisher  : world -> panda_link0
    rviz2                       : with the bundled demo.rviz config

If the arm renders here but not in visual_demo, the bug is in the demo
driver / MoveIt wiring, not in the URDF or RViz config.

Implementation note: ``home_joint_state_publisher`` is an ament_python
console_scripts entry point, so it is installed under ``install/.../bin/``
rather than the ``lib/<pkg>/`` directory that ``launch_ros.Node`` searches.
We therefore launch it via ``ExecuteProcess`` (the entry-point script is
on ``PATH`` once the workspace is sourced).
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    share = Path(get_package_share_directory("generated_objects_eval"))
    rviz_cfg = share / "config" / "demo.rviz"

    moveit_config = (
        MoveItConfigsBuilder("moveit_resources_panda")
        .robot_description(
            file_path="config/panda.urdf.xacro",
            mappings={"ros2_control_hardware_type": "mock_components"},
        )
        .robot_description_semantic(file_path="config/panda.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .to_moveit_configs()
    )

    use_sim_time = {"use_sim_time": False}

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="log",
        parameters=[moveit_config.robot_description, use_sim_time],
    )
    jsp = ExecuteProcess(
        cmd=["home_joint_state_publisher",
             "--ros-args", "-p", "use_sim_time:=false"],
        output="log",
    )
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
        name="rviz2_arm_sim",
        output="screen",
        arguments=["-d", str(rviz_cfg)],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            use_sim_time,
        ],
    )

    return LaunchDescription([rsp, jsp, world_tf, rviz])
