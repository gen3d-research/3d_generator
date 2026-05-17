from setuptools import find_packages, setup
from glob import glob

package_name = "generated_objects_eval"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/worlds", glob("worlds/*.sdf")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Abdulrahman S. Al-Batati",
    maintainer_email="aalbatati@psu.edu.sa",
    description="Downstream evaluation of generated objects via MoveIt 2 + gz_sim.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "moveit_planning_eval = generated_objects_eval.moveit_planning_eval:main",
            "gazebo_stability_eval = generated_objects_eval.gazebo_stability_eval:main",
            "home_joint_state_publisher = generated_objects_eval.home_joint_state_publisher:main",
            "demo_plan_driver = generated_objects_eval.demo_plan_driver:main",
        ],
    },
)
