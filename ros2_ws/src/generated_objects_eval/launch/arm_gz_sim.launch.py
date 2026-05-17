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

Environment requirements
------------------------
gz_sim only searches its own plugin path for system plugins, not
``LD_LIBRARY_PATH``.  ``libgz_ros2_control-system.so`` lives in
``/opt/ros/jazzy/lib`` (the ROS install prefix), so this launch prepends
that directory to ``GZ_SIM_SYSTEM_PLUGIN_PATH`` before invoking ``gz sim``.
``rmw_zenoh_cpp`` (Jazzy default) intermittently fails to expose ROS
service endpoints to nodes started shortly after each other, which made
``/controller_manager/list_controllers`` invisible to the spawners; this
launch therefore pins the entire stack to ``rmw_fastrtps_cpp`` for
reliable service discovery.  Both overrides apply only to subprocesses
this file starts; they don't leak to other shells.

No MoveIt is started here --- this is the "is the Panda physically alive
in Gazebo with working controllers?" smoke test.  visual_demo.launch.py
adds MoveIt 2 on top once this works.
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetEnvironmentVariable, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from generated_objects_eval.build_panda_urdf import build_panda_urdf


def _ros_plugin_path() -> str:
    ros_lib = "/opt/ros/jazzy/lib"
    existing = os.environ.get("GZ_SIM_SYSTEM_PLUGIN_PATH", "")
    if existing and ros_lib not in existing.split(os.pathsep):
        return ros_lib + os.pathsep + existing
    return ros_lib if not existing else existing


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
    # gz_sim needs a few seconds to spawn the model and bring up the
    # gz_ros2_control plugin's controller_manager; the spawners then
    # wait for /controller_manager/list_controllers themselves with a
    # generous --controller-manager-timeout.
    _spawner = lambda controller: Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            controller,
            "--controller-manager", "/controller_manager",
            "--controller-manager-timeout", "30",
        ],
        output="screen",
        parameters=[use_sim_time],
    )
    js_broadcaster = TimerAction(period=8.0,
                                 actions=[_spawner("joint_state_broadcaster")])
    arm_ctrl = TimerAction(period=10.0,
                           actions=[_spawner("panda_arm_controller")])
    hand_ctrl = TimerAction(period=12.0,
                            actions=[_spawner("panda_hand_controller")])

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
        SetEnvironmentVariable("GZ_SIM_SYSTEM_PLUGIN_PATH", _ros_plugin_path()),
        SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
        DeclareLaunchArgument(
            "render_engine", default_value="ogre2",
            description="gz_sim render engine: ogre2 (default) or ogre"),
        OpaqueFunction(function=_launch_setup),
    ])
