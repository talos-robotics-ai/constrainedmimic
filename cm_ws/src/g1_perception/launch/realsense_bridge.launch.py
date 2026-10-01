import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    rviz_config = os.path.join(get_package_share_directory("g1_perception"), "rviz", "realsense.rviz")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "zmq_addr",
                default_value="tcp://192.168.123.164:5556",
                description="RealSense server: PC2 directly, or tcp://192.168.1.112:5556 via the Orin relay",
            ),
            DeclareLaunchArgument("rviz", default_value="true", description="Open RViz2 with realsense.rviz"),
            DeclareLaunchArgument(
                "sam3",
                default_value="true",
                description="Start scene_capture (SAM 3, ~4 GB GPU) with the /g1_perception/capture_scene service",
            ),
            Node(
                package="g1_perception",
                executable="realsense_bridge",
                name="realsense_zmq_bridge",
                output="screen",
                emulate_tty=True,
                parameters=[{"zmq_addr": LaunchConfiguration("zmq_addr")}],
            ),
            Node(
                package="g1_perception",
                executable="scene_capture",
                name="scene_capture",
                output="screen",
                emulate_tty=True,
                condition=IfCondition(LaunchConfiguration("sam3")),
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="own_log",
                arguments=["-d", rviz_config],
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
        ]
    )
