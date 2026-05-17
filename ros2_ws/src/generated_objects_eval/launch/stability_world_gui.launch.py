"""Launches the panda_eval_world with the gz_sim GUI (no -s flag) so the
physics can be screen-recorded. Used by the project-page video pipeline."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    share = Path(get_package_share_directory("generated_objects_eval"))
    world = share / "worlds" / "panda_eval_world.sdf"
    return LaunchDescription([
        ExecuteProcess(
            cmd=["gz", "sim", "-r", "--render-engine", "ogre2", str(world)],
            output="screen",
        ),
    ])
