"""Launches the headless gz_sim world used by gazebo_stability_eval."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    share = Path(get_package_share_directory("generated_objects_eval"))
    world = share / "worlds" / "panda_eval_world.sdf"

    return LaunchDescription([
        DeclareLaunchArgument("headless", default_value="true"),
        ExecuteProcess(
            cmd=[
                "gz", "sim",
                "-s",                                # server only
                "-r",                                # start unpaused
                "--render-engine", "ogre2",
                str(world),
            ],
            output="screen",
        ),
    ])
