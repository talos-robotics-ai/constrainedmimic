"""Record a bag of the karate chop perception phase, to replay it in sim.

Records the scene captures (snapshot frames, masks, boxes in the camera frame
and in both filter frames), the TF tree they need and the robot state and
commands. The capture topics are latched, so a capture made before the
recording started is recorded too. With record_camera:=true the live camera
streams are added (~6 MB/s), e.g. to re-run SAM 3 offline.

The bag is written to <bag_dir>/karate_chop_perception_<date>_<time>.

The recorder's output goes to the launch log (~/.ros/log), not the terminal:
on the robot network rosbag2 logs an error for every discovery pass about the
robot's own multi-type topics (/lf/lowstate, /lf/sportmodestate), whatever
topics it is asked to record.
"""

import time

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

PERCEPTION_TOPICS = [
    "/g1_perception/boxes",
    "/g1_perception/obstacles",
    "/g1_perception/overlay",
    "/g1_perception/snapshot/color",
    "/g1_perception/snapshot/color_info",
    "/g1_perception/snapshot/depth",
    "/g1_perception/snapshot/depth_info",
    "/g1_perception/snapshot/cloud",
    "/g1_perception/snapshot/label_image",
    "/g1_control/karate_chop/obstacles/reference",
    "/g1_control/karate_chop/obstacles/dynamic",
    "/tf",
    "/tf_static",
    "/g1_control/robot_state",
    "/g1_control/root_state",
    "/g1_control/robot_command",
]
CAMERA_TOPICS = [
    "/camera/color/image_raw/compressed",
    "/camera/color/camera_info",
    "/camera/depth/image_rect_raw",
    "/camera/depth/camera_info",
]


def generate_launch_description():
    output = PathJoinSubstitution(
        [LaunchConfiguration("bag_dir"), time.strftime("karate_chop_perception_%Y%m%d_%H%M%S")]
    )

    def record(topics, condition):
        return ExecuteProcess(
            cmd=["ros2", "bag", "record", "-s", "mcap", "-o", output, "--topics", *topics],
            output="own_log",
            condition=condition,
        )

    return LaunchDescription(
        [
            DeclareLaunchArgument("bag_dir", default_value="bags"),
            DeclareLaunchArgument("record_camera", default_value="false"),
            record(PERCEPTION_TOPICS, UnlessCondition(LaunchConfiguration("record_camera"))),
            record(PERCEPTION_TOPICS + CAMERA_TOPICS, IfCondition(LaunchConfiguration("record_camera"))),
        ]
    )
