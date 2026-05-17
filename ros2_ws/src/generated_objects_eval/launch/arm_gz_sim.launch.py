"""Spawn the Panda in gz_sim with full ros2_control integration.

Layout::

    gz sim (panda_eval_world, windowed)
        +-- gz_ros2_control plugin (from URDF) -> ros2 controller_manager
    robot_state_publisher (full URDF on /robot_description, /tf)
    ros_gz_sim create (spawns Panda model into the running world)
    spawner joint_state_broadcaster
    spawner panda_arm_controller         (JointTrajectoryController)
    spawner panda_hand_controller        (GripperActionController)
    static_transform_publisher world -> panda_link0
    rviz2 (demo.rviz)

No MoveIt is started here --- this is the "is the Panda physically alive
in Gazebo with working controllers?" smoke test.  visual_demo.launch.py
adds MoveIt 2 on top once this works.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from generated_objects_eval.build_panda_urdf import build_panda_urdf


def _launch_setup(context):
    share = Path(get_package_share_directory("generated_objects_eval"))
    world = share / "worlds" / "panda_eval_world.sdf"
    rviz_cfg = share / "config" / "demo.rviz"

    render_engine = LaunchConfiguration("render_engine").perform(context)
    urdf_string = build_panda_urdf()

    use_sim_time = {"use_sim_time": True}  # gz_sim drives the clock here

    gz = ExecuteProcess(
        cmd=["gz", "sim", "-r", "--render-engine", render_engine, str(world)],
        output="screen",
    )

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="log",
        parameters=[{"robot_description": urdf_string}, use_sim_time],
    )

    # Spawn the Panda into the running gz world from /robot_description.
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

    # /clock bridge so ros2 nodes see gz's sim time.
    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
        output="log",
    )

    # controller_manager is spawned inside gz_sim by the plugin; spawners
    # come up after that's alive.
    js_broadcaster = TimerAction(
        period=5.0,
        actions=[Node(
            package="controller_manager",
            executable="spawner",
            arguments=["joint_state_broadcaster",
                       "--controller-manager", "/controller_manager"],
            output="screen",
        )],
    )
    arm_ctrl = TimerAction(
        period=6.0,
        actions=[Node(
            package="controller_manager",
            executable="spawner",
            arguments=["panda_arm_controller",
                       "--controller-manager", "/controller_manager"],
            output="screen",
        )],
    )
    hand_ctrl = TimerAction(
        period=7.0,
        actions=[Node(
            package="controller_manager",
            executable="spawner",
            arguments=["panda_hand_controller",
                       "--controller-manager", "/controller_manager"],
            output="screen",
        )],
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
        name="rviz2_arm_gz",
        output="screen",
        arguments=["-d", str(rviz_cfg)],
        parameters=[{"robot_description": urdf_string}, use_sim_time],
    )

    return [gz, clock_bridge, rsp, spawn,
            js_broadcaster, arm_ctrl, hand_ctrl,
            world_tf, rviz]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "render_engine", default_value="ogre2",
            description="gz_sim render engine: ogre2 (default) or ogre"),
        OpaqueFunction(function=_launch_setup),
    ])
