import glob
import os

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


OBSTACLES_TOPIC = "/g1_control/karate_chop/obstacles/reference"


def capture_count(bag):
    """Number of scene captures recorded in a bag (0 if it has none or is unreadable)"""
    try:
        with open(os.path.join(bag, "metadata.yaml")) as f:
            info = yaml.safe_load(f)["rosbag2_bagfile_information"]
    except (OSError, KeyError, TypeError, yaml.YAMLError):
        return 0
    for topic in info.get("topics_with_message_count", []):
        if topic["topic_metadata"]["name"] == OBSTACLES_TOPIC:
            return topic["message_count"]
    return 0


def latest_capture_bag(bag_dir="bags"):
    """Newest recorded bag (relative to the launch directory) holding a scene capture"""
    for bag in sorted(glob.glob(os.path.join(bag_dir, "karate_chop_perception_*")), reverse=True):
        if capture_count(bag) > 0:
            return bag
    return ""


def generate_launch_description():
    default_bag = latest_capture_bag()
    # Robot model for RViz: the G1 URDF installed with g1_control, its relative
    # mesh paths made absolute so that RViz can load them
    share = get_package_share_directory("g1_control")
    with open(os.path.join(share, "unitree_g1", "g1_29dof_rev_1_0.urdf")) as f:
        robot_description = f.read().replace(
            'filename="meshes/', f'filename="file://{share}/unitree_g1/meshes/'
        )

    return LaunchDescription(
        [
            DeclareLaunchArgument("add_kinematic_cbf", default_value="false"),
            DeclareLaunchArgument("add_dynamic_cbf", default_value="false"),
            # For karate chop, we can use our simple double-support root state
            # estimator. Other options include "mimic" for using the root state
            # from the reference motion, or "topic" if in sim or using mocap
            DeclareLaunchArgument("root_state_source", default_value="estimator"),
            # Where the obstacle boxes come from: "bag" (default in sim: the capture
            # recorded in obstacles_bag, used at the capture advance), "camera" (live
            # capture, needs the robot network) or "topic" (replayed on
            # /g1_control/karate_chop/obstacles/reference)
            DeclareLaunchArgument("obstacles_source", default_value="bag"),
            # Default: the newest bag with a capture in ./bags
            DeclareLaunchArgument("obstacles_bag", default_value=default_bag),
            LogInfo(
                msg=["Obstacles from the bag: ", LaunchConfiguration("obstacles_bag")],
                condition=IfCondition(
                    PythonExpression(["'", LaunchConfiguration("obstacles_source"), "' == 'bag'"])
                ),
            ),
            # Perception (camera mode only): RealSense bridge + scene_capture
            # (loads SAM 3, ~4 GB GPU). RViz shows the robot with the scene
            DeclareLaunchArgument("zmq_addr", default_value="tcp://192.168.123.164:5556"),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("sam3", default_value="true"),
            # Bag of the perception data (see karate_chop_perception_record.launch.py)
            DeclareLaunchArgument("record", default_value="false"),
            DeclareLaunchArgument("bag_dir", default_value="bags"),
            DeclareLaunchArgument("record_camera", default_value="false"),
            IncludeLaunchDescription(
                PathJoinSubstitution(
                    [FindPackageShare("g1_control"), "launch", "karate_chop_perception_record.launch.py"]
                ),
                launch_arguments={
                    "bag_dir": LaunchConfiguration("bag_dir"),
                    "record_camera": LaunchConfiguration("record_camera"),
                }.items(),
                condition=IfCondition(LaunchConfiguration("record")),
            ),
            # Scoped: the arguments passed to the include (rviz:=false) must not
            # overwrite this file's own launch arguments
            GroupAction(
                scoped=True,
                condition=IfCondition(
                    PythonExpression(["'", LaunchConfiguration("obstacles_source"), "' == 'camera'"])
                ),
                actions=[
                    IncludeLaunchDescription(
                        PathJoinSubstitution(
                            [FindPackageShare("g1_perception"), "launch", "realsense_bridge.launch.py"]
                        ),
                        launch_arguments={
                            "zmq_addr": LaunchConfiguration("zmq_addr"),
                            "rviz": "false",  # the RViz below shows the robot too
                            "sam3": LaunchConfiguration("sam3"),
                        }.items(),
                    ),
                ],
            ),
            # Robot + scene in the karate_chop_reference frame (the karate node publishes
            # /joint_states and karate_chop_reference -> pelvis)
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="own_log",
                parameters=[{"robot_description": robot_description}],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="own_log",
                arguments=["-d", os.path.join(share, "rviz", "karate_chop_perception.rviz")],
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
            Node(
                package="g1_control",
                executable="sim_node",
                name="g1_sim_node",
                output="screen",
                emulate_tty=True,
            ),
            Node(
                package="g1_control",
                executable="hardware_interface_node",
                name="hardware_interface_node",
                output="screen",
                emulate_tty=True,
            ),
            Node(
                package="g1_control",
                executable="karate_chop_perception_node.py",
                name="karate_chop_perception_node",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {
                        "add_kinematic_cbf": LaunchConfiguration("add_kinematic_cbf"),
                        "add_dynamic_cbf": LaunchConfiguration("add_dynamic_cbf"),
                        "root_state_source": LaunchConfiguration("root_state_source"),
                        "obstacles_source": LaunchConfiguration("obstacles_source"),
                        "obstacles_bag": LaunchConfiguration("obstacles_bag"),
                    }
                ],
            ),
        ]
    )
