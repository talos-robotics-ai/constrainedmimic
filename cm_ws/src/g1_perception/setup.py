from glob import glob

from setuptools import find_packages, setup

package_name = "g1_perception"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="g1ack52",
    maintainer_email="gabriel.voss01@gmail.com",
    description="Perception for the Unitree G1: RealSense ZMQ bridge and SAM 3 scene capture",
    license="MIT",
    entry_points={
        "console_scripts": [
            "realsense_bridge = g1_perception.realsense_bridge:main",
            "scene_capture = g1_perception.scene_capture:main",
        ],
    },
)
