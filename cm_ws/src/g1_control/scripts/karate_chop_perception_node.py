#!/usr/bin/env python3
"""
Karate chop demo node with a perception phase

Copy of karate_chop_node.py with a perception phase: once the policy is
standing, the next advance captures the scene with g1_perception (SAM 3 boxes)
and expresses the obstacle boxes in the frames the safety filters work in.
Both filters use the boxes in place of the fixed Z_MIN plane, keeping the
right hand collision sphere outside every box:
- kinematic (add_kinematic_cbf): the reference motion is not retargeted at
  startup but after every capture, in a background process (~15 s) so the
  control loop is never stalled
- dynamic (add_dynamic_cbf): compiled once at startup with MAX_OBSTACLES box
  slots; a capture only changes the box data it is called with. Unlike
  karate_chop_node.py it filters all arm motion, RAISE_ARM, HOLD_RAISED,
  RUN_MOTION and LOWER_ARM, since the raise and lower transitions are joint
  space blends that the kinematic filter never checked

State machine:
    WAIT_FOR_ROBOT -> IDLE -(advance)-> MOVE_TO_DEFAULT -> HOLD_DEFAULT
    -(advance)-> TRACK_STAND -(advance)-> CAPTURE_SCENE -(boxes)-> RETARGET
    -> SCENE_READY -(advance)-> RAISE_ARM -> HOLD_RAISED -(advance)->
    RUN_MOTION -> LOWER_ARM -> TRACK_STAND (replay with a new capture, or
    stop). CAPTURE_SCENE, RETARGET and SCENE_READY keep the stand;
    CAPTURE_SCENE and RETARGET fall back to TRACK_STAND on failure or timeout
    (advance retries). RETARGET is skipped without add_kinematic_cbf, in which
    case the motion is retargeted (unfiltered) at startup.

Obstacle frames (the camera pose comes from FK of the joints measured at the
capture, assuming the left foot is planted):
- reference: the frame of the reference motion, where the standing pelvis is
  at (0, 0, 0.8). Used by the kinematic filter (retargeting) and by the
  dynamic filter with root_state_source=mimic
- dynamic: the dynamic filter's world for the chosen root_state_source:
  estimator -> left foot frame (origin at the left foot), mimic -> reference,
  topic -> the /g1_control/root_state world
For RViz the node publishes /joint_states and the pelvis pose in the reference
frame on TF (karate_chop_reference -> pelvis, left foot planted like for the
boxes), so robot_state_publisher shows the robot together with the scene.
The camera pose in the reference frame is also broadcast on TF
(karate_chop_reference -> camera depth frame, and -> karate_chop_left_foot) to
check the boxes in RViz. Both box sets are published latched as
g1_control_msgs/BoxArray on /g1_control/karate_chop/obstacles/{reference,dynamic},
so a bag of a capture can be replayed to test the filters in sim.

obstacles_source selects where the boxes come from at the capture advance:
- camera (default): a live capture with g1_perception
- topic: the last g1_control/BoxArray on /g1_control/karate_chop/obstacles/reference,
  e.g. replayed from a bag; no camera or SAM 3 needed. The dynamic filter's
  boxes are derived from them (estimator and mimic root state sources only)
- bag: the last capture recorded in the bag at obstacles_bag (see
  karate_chop_perception_record.launch.py), read at startup but used only at
  the capture advance, like topic. The capture's point cloud, box markers, mask
  overlay and transforms are published then, to see the scene in RViz

Inputs (either via terminal or by pressing the unitree remote buttons):
- Unitree remote (via /lowstate): A = advance, B = stop (hold the current
  commanded position), SELECT = emergency stop (robot goes limp)
- Topics:
    ros2 topic pub --once /g1_control/karate_chop/advance std_msgs/msg/Empty
    ros2 topic pub --once /g1_control/karate_chop/stop std_msgs/msg/Empty
    ros2 topic pub --once /g1_control/emergency_stop std_msgs/msg/Empty
"""

import multiprocessing
import os
import queue
import signal
import time
import traceback
from enum import Enum
from pathlib import Path

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

import rclpy
import rosbag2_py
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from unitree_hg.msg import LowState
from g1_control_msgs.msg import Box, BoxArray, RobotState, RobotCommand, FrameState
from g1_control_msgs.srv import CaptureScene

import cm_control
from cbfpy import CBF
from frax import load_g1
from frax.robots.unitree_g1 import fixed_root_joint_ordering
from cm_control.config import g1_config
from cm_control.core.actuators import RobotActuatorModel
from cm_control.core.dynamic_filter_utils import PDFilter
from cm_control.core.dynamic_configs import BaseDynamicConfig
from cm_control.core.kinematic_configs import BaseKinematicConfig
from cm_control.utils.free_floating_utils import pose_and_twist_to_virtual_joints
from cm_control.core.simple_state_estimation import (
    FunctionalEstimator,
    StatefulEstimator,
)
from cm_control.retargeting.motion_processing import load_and_retarget_motion
from cm_control.retargeting.pico_to_g1_retarget import (
    PicoToG1Retargeter,
    StatefulRetargeter,
)
from cm_control.twist2_utils import twist2_config
from cm_control.twist2_utils.motion_tracking import (
    FunctionalMotionTracker,
    StatefulMotionTracker,
)
from cm_control.utils.rotation_utils import slerp

jax.config.update("jax_enable_x64", True)

DEFAULT_MOTION_FILE = str(
    Path(cm_control.__file__).parents[2]
    / "cm_control"
    / "cm_control"
    / "assets"
    / "data"
    / "rosbag2_2026_05_06-13_49_25_crop_2046_to_2450.pkl"
)

# RealSense mount from the G1 URDF (d435_joint): torso_link -> d435_link
D435_IN_TORSO = np.eye(4)
D435_IN_TORSO[:3, :3] = Rotation.from_euler("y", 0.8307767239493009).as_matrix()
D435_IN_TORSO[:3, 3] = [0.0576235, 0.01753, 0.42987]
# d435_link (x forward, y left, z up) -> optical frame (z forward, x right, y down)
OPTICAL_IN_D435 = np.eye(4)
OPTICAL_IN_D435[:3, :3] = [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
REFERENCE_FRAME = "karate_chop_reference"
# Frame of the dynamic filter's world for each root_state_source
DYNAMIC_FRAMES = {
    "estimator": "karate_chop_left_foot",
    "mimic": REFERENCE_FRAME,
    "topic": "karate_chop_root_state_world",
}
# Right hand collision sphere, the one the karate chop CBFs constrain
RIGHT_HAND_SPHERE = 44
# Box slots of the dynamic filter (fixed so that it compiles once)
MAX_OBSTACLES = 8
FAR_BOX = {
    "label": "far",
    "score": 0.0,
    "centre": np.array([0.0, 0.0, -10.0]),
    "rotation": np.eye(3),
    "half_extents": np.full(3, 0.1),
}

# Recorded capture topics: the boxes the node uses, and what RViz shows of the scene
BAG_OBSTACLES_TOPIC = "/g1_control/karate_chop/obstacles/reference"
BAG_SCENE_TOPICS = (
    "/g1_perception/snapshot/cloud",
    "/g1_perception/obstacles",
    "/g1_perception/overlay",
)

# Unitree remote bit indices. See unitree_ros2 gamepad.hpp
BUTTON_SELECT_BIT = 3
BUTTON_A_BIT = 8
BUTTON_B_BIT = 9


class State(Enum):
    WAIT_FOR_ROBOT = 0
    IDLE = 1
    MOVE_TO_DEFAULT = 2
    HOLD_DEFAULT = 3
    TRACK_STAND = 4
    CAPTURE_SCENE = 5
    RETARGET = 6
    SCENE_READY = 7
    RAISE_ARM = 8
    HOLD_RAISED = 9
    RUN_MOTION = 10
    LOWER_ARM = 11
    FINISHED = 12
    KILLED = 13


class KarateChopNode(Node):
    def __init__(self):
        super().__init__("karate_chop_perception_node")

        # Tunable parameters
        self.declare_parameter("motion_file", DEFAULT_MOTION_FILE)
        self.declare_parameter("add_kinematic_cbf", False)
        self.declare_parameter("add_dynamic_cbf", False)
        # The root state can either be from 'topic', 'estimator', or 'mimic'
        # topic: use ground truth from sim or mocap on hardware
        # estimator: kinematic estimate assuming double-support contact
        # mimic: assume that the root state matches the reference motion
        self.declare_parameter("root_state_source", "estimator")
        self.declare_parameter("move_to_default_duration", 3.0)
        self.declare_parameter("transition_duration", 3.0)
        # Perception phase: SAM 3 prompts and how long to wait for the boxes
        self.declare_parameter("capture_prompts", ["table"])
        self.declare_parameter("capture_timeout", 10.0)
        # "camera": live capture, "topic": replayed boxes (see the module docstring)
        self.declare_parameter("obstacles_source", "camera")
        self.declare_parameter("obstacles_bag", "")  # with obstacles_source=bag
        motion_file = self.get_parameter("motion_file").value
        add_kinematic_cbf = self.get_parameter("add_kinematic_cbf").value
        add_dynamic_cbf = self.get_parameter("add_dynamic_cbf").value
        self.root_state_source = self.get_parameter("root_state_source").value
        if self.root_state_source not in ("topic", "estimator", "mimic"):
            raise ValueError(f"Invalid root state source: {self.root_state_source}")
        self.move_to_default_duration = self.get_parameter(
            "move_to_default_duration"
        ).value
        self.transition_duration = self.get_parameter("transition_duration").value
        self.capture_prompts = list(self.get_parameter("capture_prompts").value)
        self.capture_timeout = self.get_parameter("capture_timeout").value
        self.obstacles_source = self.get_parameter("obstacles_source").value
        if self.obstacles_source not in ("camera", "topic", "bag"):
            raise ValueError(f"Invalid obstacles source: {self.obstacles_source}")
        if self.obstacles_source == "bag":
            # Read now so the capture advance does not stall the control loop
            obstacles_bag = self.get_parameter("obstacles_bag").value
            if not obstacles_bag:
                raise ValueError(
                    "obstacles_source=bag needs obstacles_bag:=<path to a recorded capture>"
                )
            self.bag_capture = read_bag_capture(obstacles_bag)
            self.get_logger().info(
                f"Obstacles from {obstacles_bag}: "
                f"{len(self.bag_capture['obstacles'].boxes)} boxes, used at the capture advance"
            )

        self.control_freq = 50
        self.control_dt = 1 / self.control_freq

        # PD constants and default (standing) reference
        self.stiffness = np.asarray(twist2_config.sim_kps)
        self.damping = np.asarray(twist2_config.sim_kds)
        self.default_q_actuated = np.asarray(twist2_config.default_joint_position)
        self.default_root_pos = np.array([0.0, 0.0, 0.8])
        self.default_root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.zero_vec3 = np.zeros(3)
        self.default_contact_mode = 3  # Both feet

        # Safety filters. The kinematic one is built after each scene capture,
        # from the obstacle boxes (see retarget_with_obstacles)
        robot = load_g1()
        self.robot = robot
        self.add_kinematic_cbf = add_kinematic_cbf
        self.motion_file = motion_file
        # Boxes the dynamic filter is called with (dynamic filter frame)
        self.dynamic_obstacles = obstacle_arrays([])
        if add_dynamic_cbf:
            dyn_cbf = CBF.from_config(BoxObstaclesDynamicCBFConfig(robot))
            actuator_model = RobotActuatorModel(
                self.stiffness, self.damping, g1_config.joint_max_torques
            )
            self.pd_filter = ObstaclePDFilter(dyn_cbf, actuator_model)
        else:
            self.pd_filter = None
        if self.pd_filter is not None and self.root_state_source == "estimator":
            self.estimator = StatefulEstimator(FunctionalEstimator(robot))
        else:
            self.estimator = None

        # Camera pose from the measured joints, jitted here so that a capture
        # does not stall the control loop
        self.camera_fk = jax.jit(partial(torso_in_feet_and_pelvis, robot))
        self.camera_fk(self.default_q_actuated)
        # Where the left foot sits in the reference frame while standing
        stand_q = np.concatenate([self.default_root_pos, np.zeros(3), self.default_q_actuated])
        self.left_foot_in_reference = np.asarray(
            robot._left_foot_transform(robot.joint_to_world_transforms(stand_q))
        )
        self.pelvis_fk = jax.jit(partial(pelvis_in_left_foot, robot))
        self.pelvis_fk(self.default_q_actuated)

        # Load and retarget the reference motion.
        # Note: this demo doesn't run the constrained retargeter online, it instead
        # uses it to precompute the safe reference motion from human data. With
        # the kinematic CBF this happens after the scene capture, so there is no
        # motion (and the arm cannot be raised) until then
        self.motion = None
        if add_kinematic_cbf:
            # Started now so that its imports are done by the first capture
            context = multiprocessing.get_context("spawn")
            self.retarget_tasks = context.Queue()
            self.retarget_tasks.cancel_join_thread()  # never block the node's exit
            self.retarget_results = context.Queue()
            self.retarget_process = context.Process(
                target=retarget_worker,
                args=(self.retarget_tasks, self.retarget_results),
                daemon=True,
            )
            self.retarget_process.start()
        else:
            self.get_logger().info(f"Loading and retargeting motion: {motion_file}")
            self.set_motion(retarget_motion(robot, motion_file, self.control_freq, None))

        self.motion_tracker = StatefulMotionTracker(FunctionalMotionTracker())
        if self.pd_filter is not None:
            self.get_logger().info("JIT compiling the PD filter...")
            if self.estimator is not None:
                jax.block_until_ready(
                    self.pd_filter.filter_from_virtual_joints(
                        self.default_q_actuated,
                        np.zeros(6),
                        np.zeros(6),
                        self.default_q_actuated,
                        np.zeros(29),
                        self.default_contact_mode,
                        self.dynamic_obstacles,
                    )
                )
            else:
                jax.block_until_ready(
                    self.pd_filter.filter(
                        self.default_q_actuated,
                        self.default_root_pos,
                        self.default_root_quat,
                        self.zero_vec3,
                        self.zero_vec3,
                        self.default_q_actuated,
                        np.zeros(29),
                        self.default_contact_mode,
                        self.dynamic_obstacles,
                    )
                )

        # Warm up the eager JAX math of the policy steps (quaternion blend, and the
        # action update after filtering). Compiled on first use, it would stall the
        # first stand tick (~0.4 s, as the policy takes over) and the first raise tick
        for pct in (0.0, 0.5):
            jax.block_until_ready(slerp(self.default_root_quat, self.default_root_quat, pct))
        if self.pd_filter is not None:
            tracker = self.motion_tracker.functional_tracker
            jax.block_until_ready(
                (1 / tracker.action_scale)
                * (jnp.asarray(self.default_q_actuated) - tracker.default_dof_pos)
            )

        # State machine management
        self.state = State.WAIT_FOR_ROBOT
        self.advance_requested = False
        self.stop_requested = False
        self.estop_requested = False
        self.interp_start_time = None
        self.interp_start_q = None
        self.motion_step = 0
        self.hold_pd_target = None
        self.capture_future = None
        self.capture_start_time = None
        self.capture_joints = None
        self.capture_root_state = None
        # Retarget job counter: results of abandoned jobs (e.g. after a reset) are dropped
        self.retarget_job = 0
        self.retarget_start_time = None
        self.last_loop_time = None
        self.max_loop_interval = 0.0
        # Obstacle boxes from the last capture, see the module docstring
        self.obstacles_reference = []
        self.obstacles_dynamic = []
        self.replayed_obstacles = None  # last BoxArray with obstacles_source=topic

        # Latest message caches
        self.last_robot_state = None
        self.last_root_state = None
        self.prev_remote_keys = 0

        # QoS settings
        qos_best_effort_keep_last_volatile_depth_1 = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )
        qos_reliable_keep_last_volatile_depth_1 = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Publishers and subscribers
        self.robot_state_sub = self.create_subscription(
            RobotState,
            "/g1_control/robot_state",
            self.robot_state_callback,
            qos_best_effort_keep_last_volatile_depth_1,
        )
        self.root_state_sub = self.create_subscription(
            FrameState,
            "/g1_control/root_state",
            self.root_state_callback,
            qos_best_effort_keep_last_volatile_depth_1,
        )
        self.low_state_sub = self.create_subscription(
            LowState,
            "/lowstate",
            self.low_state_callback,
            qos_best_effort_keep_last_volatile_depth_1,
        )
        self.advance_sub = self.create_subscription(
            Empty,
            "/g1_control/karate_chop/advance",
            self.advance_callback,
            qos_reliable_keep_last_volatile_depth_1,
        )
        self.stop_sub = self.create_subscription(
            Empty,
            "/g1_control/karate_chop/stop",
            self.stop_callback,
            qos_reliable_keep_last_volatile_depth_1,
        )
        self.reset_sub = self.create_subscription(
            Empty,
            "/g1_control/reset",
            self.reset_callback,
            qos_reliable_keep_last_volatile_depth_1,
        )
        self.robot_command_pub = self.create_publisher(
            RobotCommand,
            "/g1_control/robot_command",
            qos_best_effort_keep_last_volatile_depth_1,
        )
        self.estop_pub = self.create_publisher(
            Empty, "/g1_control/emergency_stop", qos_reliable_keep_last_volatile_depth_1
        )
        qos_reliable_keep_last_transient_depth_1 = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.capture_client = self.create_client(
            CaptureScene, "/g1_perception/capture_scene"
        )
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        if self.obstacles_source == "camera":
            self.obstacles_reference_pub = self.create_publisher(
                BoxArray,
                "/g1_control/karate_chop/obstacles/reference",
                qos_reliable_keep_last_transient_depth_1,
            )
            self.obstacles_dynamic_pub = self.create_publisher(
                BoxArray,
                "/g1_control/karate_chop/obstacles/dynamic",
                qos_reliable_keep_last_transient_depth_1,
            )
        elif self.obstacles_source == "bag":
            self.bag_scene_pubs = {
                topic: self.create_publisher(
                    type(msg), topic, qos_reliable_keep_last_transient_depth_1
                )
                for topic, msg in self.bag_capture["scene"].items()
            }
        else:
            # Same topic the camera mode publishes on, so a recorded bag plays as is
            self.replayed_obstacles_sub = self.create_subscription(
                BoxArray,
                "/g1_control/karate_chop/obstacles/reference",
                self.replayed_obstacles_callback,
                qos_reliable_keep_last_transient_depth_1,
            )
        # Virtual gantry is only for simulation
        self.gantry_pub = self.create_publisher(
            Bool, "/g1_control/sim_gantry", qos_reliable_keep_last_transient_depth_1
        )
        # Keep the robot supported until the policy takes over
        self.set_gantry(True)

        # Keep a message prepared to send in an emergency kill state
        kill_msg = RobotCommand()
        kill_msg.q = [0.0] * 29
        kill_msg.dq = [0.0] * 29
        kill_msg.kp = [0.0] * 29
        kill_msg.kd = [1.0] * 29  # Slight damping
        kill_msg.tau = [0.0] * 29
        self.kill_msg = kill_msg

        self.timer = self.create_timer(self.control_dt, self.control_loop)

        # Robot pose and joints for RViz (robot_state_publisher)
        self.joint_state_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.robot_tf_broadcaster = TransformBroadcaster(self)
        self.robot_viz_timer = self.create_timer(1 / 30, self.publish_robot_viz)

        # fmt: off
        self.get_logger().info("====================================================")
        self.get_logger().info("Karate chop node initialized")
        self.get_logger().info(f"{add_kinematic_cbf=}")
        self.get_logger().info(f"{add_dynamic_cbf=}")
        self.get_logger().info(f"root_state_source={self.root_state_source}")
        self.get_logger().info("Controls: A = advance, B = stop (hold), SELECT = emergency stop")
        self.get_logger().info("(or publish to /g1_control/karate_chop/{advance,stop})")
        self.get_logger().info("====================================================")
        # fmt: on

    def robot_state_callback(self, msg):
        self.last_robot_state = msg

    def root_state_callback(self, msg):
        self.last_root_state = msg

    def low_state_callback(self, msg):
        # Parse the unitree remote button data
        keys = int(msg.wireless_remote[2]) | (int(msg.wireless_remote[3]) << 8)
        new_presses = keys & ~self.prev_remote_keys
        self.prev_remote_keys = keys
        if new_presses & (1 << BUTTON_A_BIT):
            self.advance_requested = True
        if new_presses & (1 << BUTTON_B_BIT):
            self.stop_requested = True
        if new_presses & (1 << BUTTON_SELECT_BIT):
            self.estop_requested = True

    def replayed_obstacles_callback(self, msg):
        self.replayed_obstacles = msg
        self.get_logger().info(
            f"Received {len(msg.boxes)} replayed obstacle boxes", throttle_duration_sec=5.0
        )

    def advance_callback(self, msg):
        self.advance_requested = True

    def stop_callback(self, msg):
        self.stop_requested = True

    def reset_callback(self, msg):
        # Restart the state machine from scratch
        self.get_logger().info("Reset received: restarting the state machine")
        self.state = State.WAIT_FOR_ROBOT
        self.advance_requested = False
        self.stop_requested = False
        self.estop_requested = False
        self.motion_step = 0
        self.hold_pd_target = None
        if self.capture_future is not None:
            self.capture_future.cancel()
        self.capture_future = None
        self.obstacles_reference = []
        self.obstacles_dynamic = []
        # A retarget still running finishes in the background; its result is dropped
        self.retarget_job += 1
        self.dynamic_obstacles = obstacle_arrays([])
        self.last_robot_state = None
        functional_tracker = self.motion_tracker.functional_tracker
        self.motion_tracker._last_action = np.zeros(functional_tracker.num_actions)
        self.motion_tracker._obs_history = np.zeros(
            (functional_tracker.history_len, functional_tracker.n_obs_single)
        )
        if self.estimator is not None:
            self.estimator.reset()
        self.set_gantry(True)

    def publish_robot_viz(self):
        """Joints and pelvis pose (reference frame, left foot planted) for RViz"""
        if self.last_robot_state is None:
            return
        q = np.array(self.last_robot_state.q)
        stamp = self.get_clock().now().to_msg()
        joints = JointState(name=list(fixed_root_joint_ordering), position=q.tolist())
        joints.header.stamp = stamp
        self.joint_state_pub.publish(joints)
        pelvis_in_reference = self.left_foot_in_reference @ np.asarray(self.pelvis_fk(q))
        self.robot_tf_broadcaster.sendTransform(
            transform_msg(pelvis_in_reference, REFERENCE_FRAME, "pelvis", stamp)
        )

    def set_gantry(self, enabled):
        self.gantry_pub.publish(Bool(data=enabled))

    def get_current_time_in_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def interpolation_pct(self, duration):
        elapsed = self.get_current_time_in_seconds() - self.interp_start_time
        return min(elapsed / duration, 1.0)

    def publish_pd_command(self, q_target):
        q_target = np.asarray(q_target)
        msg = RobotCommand()
        msg.q = q_target.tolist()
        msg.dq = [0.0] * 29
        msg.kp = self.stiffness.tolist()
        msg.kd = self.damping.tolist()
        msg.tau = [0.0] * 29
        self.robot_command_pub.publish(msg)
        self.hold_pd_target = q_target

    def step_policy(
        self,
        mimic_q,
        mimic_pos,
        mimic_quat,
        mimic_vel,
        mimic_omega,
        contact_mode,
        apply_filter=False,
    ):
        robot_q = np.array(self.last_robot_state.q)
        robot_qd = np.array(self.last_robot_state.dq)
        robot_quat_wxyz = np.array(self.last_robot_state.quaternion)
        robot_omega_body = np.array(self.last_robot_state.gyroscope)
        # Keep the estimator's state up to date at every policy step, even
        # when the filter is not applied
        if self.estimator is not None:
            q_root, qd_root = self.estimator.step(robot_q, robot_qd)
        pd_target = self.motion_tracker.step(
            robot_quat_wxyz=robot_quat_wxyz,
            robot_omega_body=robot_omega_body,
            robot_q=robot_q,
            robot_qd=robot_qd,
            mimic_pos=mimic_pos,
            mimic_quat=mimic_quat,
            mimic_vel=mimic_vel,
            mimic_omega=mimic_omega,
            mimic_q=mimic_q,
        )
        # Note: (if the dynamic filter is enabled,) this runs the dynamic filter
        # at the same rate as the policy (50 Hz). This is totally fine for this demo
        # but other demos may require a higher frequency and should be decoupled
        if self.pd_filter is not None and apply_filter:
            if self.root_state_source == "estimator":
                pd_target = self.pd_filter.filter_from_virtual_joints(
                    pd_target, q_root, qd_root, robot_q, robot_qd, contact_mode,
                    self.dynamic_obstacles,
                )
            else:
                if self.root_state_source == "mimic":
                    pos, quat = mimic_pos, mimic_quat
                    vel, omega = mimic_vel, mimic_omega
                else:  # topic
                    pos = np.array(self.last_root_state.position)
                    quat = np.array(self.last_root_state.quaternion)
                    vel = np.array(self.last_root_state.velocity)
                    omega = np.array(self.last_root_state.omega)
                pd_target = self.pd_filter.filter(
                    pd_target, pos, quat, vel, omega, robot_q, robot_qd, contact_mode,
                    self.dynamic_obstacles,
                )
            # HACK -- update the last action state after filtering
            # (same as the karate chop sim example)
            self.motion_tracker._last_action = (
                1 / self.motion_tracker.functional_tracker.action_scale
            ) * (pd_target - self.motion_tracker.functional_tracker.default_dof_pos)
        self.publish_pd_command(pd_target)

    def track_interpolated_reference(self, start_step, end_step, pct, apply_filter=False):
        q0, pos0, quat0 = start_step
        q1, pos1, quat1 = end_step
        q = linear_interp(q0, q1, pct)
        pos = linear_interp(pos0, pos1, pct)
        quat = np.asarray(slerp(quat0, quat1, pct))  # TODO: use numpy slerp!
        self.step_policy(
            q, pos, quat, self.zero_vec3, self.zero_vec3, self.default_contact_mode,
            apply_filter=apply_filter,
        )

    @property
    def stand_step(self):
        return (self.default_q_actuated, self.default_root_pos, self.default_root_quat)

    def motion_keyframe(self, i):
        return (
            self.motion_q_actuated[i],
            self.motion["pos"][i],
            self.motion["quat"][i],
        )

    def start_capture(self):
        """Ask g1_perception for the obstacle boxes (asynchronous)"""
        if self.obstacles_source == "topic":
            self.use_replayed_obstacles(self.replayed_obstacles)
            return
        if self.obstacles_source == "bag":
            for topic, msg in self.bag_capture["scene"].items():
                self.bag_scene_pubs[topic].publish(msg)
            if self.bag_capture["transforms"]:
                self.tf_broadcaster.sendTransform(self.bag_capture["transforms"])
            self.use_replayed_obstacles(self.bag_capture["obstacles"])
            return
        if not self.capture_client.service_is_ready():
            self.get_logger().warn(
                "/g1_perception/capture_scene is not available, is g1_perception "
                "running? Press A to retry"
            )
            return
        if self.root_state_source == "topic" and self.last_root_state is None:
            self.get_logger().warn(
                "No root state received yet: the obstacle boxes need "
                "/g1_control/root_state with root_state_source=topic"
            )
            return
        # The robot stands still, so the joints now match the captured frame
        self.capture_joints = np.array(self.last_robot_state.q)
        self.capture_root_state = self.last_root_state
        self.capture_future = self.capture_client.call_async(
            CaptureScene.Request(prompts=self.capture_prompts)
        )
        self.capture_start_time = self.get_current_time_in_seconds()
        self.state = State.CAPTURE_SCENE
        self.get_logger().info(f"Capturing the scene {self.capture_prompts}...")

    def check_capture(self):
        """Advance once the boxes are in, fall back to the stand on failure"""
        elapsed = self.get_current_time_in_seconds() - self.capture_start_time
        if not self.capture_future.done():
            if elapsed > self.capture_timeout:
                self.capture_future.cancel()
                self.capture_failed(f"no reply after {self.capture_timeout:.0f} s")
            return
        response = self.capture_future.result()
        if response is None or not response.success:
            message = response.message if response is not None else "no response"
            self.capture_failed(message)
            return
        if not response.boxes:
            self.capture_failed(f"no obstacle boxes ({response.message})")
            return
        self.set_obstacles(response.boxes)
        self.obstacles_ready(f"Scene captured in {elapsed:.1f} s.")

    def use_replayed_obstacles(self, msg):
        """obstacles_source=topic or bag: use recorded boxes (reference frame)"""
        if msg is None or not msg.boxes:
            self.get_logger().warn(
                f"No recorded obstacle boxes ({self.obstacles_source} source: play the bag "
                "on /g1_control/karate_chop/obstacles/reference, or record a capture with "
                "boxes). Press A to retry"
            )
            return
        if msg.header.frame_id != REFERENCE_FRAME:
            self.get_logger().warn(
                f"Replayed boxes are in {msg.header.frame_id!r}, expected {REFERENCE_FRAME!r}"
            )
            return
        self.obstacles_reference = [box_in_frame(b, np.eye(4)) for b in msg.boxes]
        # Dynamic filter frame, with the same planted left foot assumption as a capture
        if self.root_state_source == "estimator":
            reference_in_dynamic = np.linalg.inv(self.left_foot_in_reference)
            self.tf_broadcaster.sendTransform(
                transform_msg(self.left_foot_in_reference, REFERENCE_FRAME,
                              DYNAMIC_FRAMES["estimator"], msg.header.stamp)
            )
        elif self.root_state_source == "mimic":
            reference_in_dynamic = np.eye(4)
        else:
            reference_in_dynamic = None
            self.get_logger().warn(
                "Replayed boxes cannot be placed in the root_state topic world: no "
                "boxes for the dynamic filter"
            )
        self.obstacles_dynamic = (
            [box_in_frame(b, reference_in_dynamic) for b in msg.boxes]
            if reference_in_dynamic is not None
            else []
        )
        self.log_obstacles()
        self.obstacles_ready("Using the replayed obstacle boxes.")

    def obstacles_ready(self, what):
        """Continue after new obstacles: hand them to the filters"""
        self.dynamic_obstacles = obstacle_arrays(self.obstacles_dynamic)
        if self.pd_filter is not None and not self.obstacles_dynamic:
            self.get_logger().warn("The dynamic filter has no obstacle boxes")
        if self.add_kinematic_cbf:
            self.start_retarget()
            self.get_logger().info(
                f"{what} Retargeting the motion with the obstacle boxes (~15 s)..."
            )
            return
        self.scene_ready()

    def scene_ready(self):
        self.state = State.SCENE_READY
        self.get_logger().info(
            f"Scene ready ({len(self.obstacles_reference)} obstacle boxes). "
            "Press A to raise the arm"
        )

    def capture_failed(self, reason):
        # Fail closed: without obstacles the motion is not started
        self.capture_future = None
        self.state = State.TRACK_STAND
        self.get_logger().warn(
            f"Scene capture failed: {reason}. Still standing, press A to retry"
        )

    def set_obstacles(self, boxes):
        """Express the boxes (camera depth optical frame) in the filter frames"""
        foot_torso, pelvis_torso = (
            np.asarray(T) for T in self.camera_fk(self.capture_joints)
        )
        camera_in_foot = foot_torso @ D435_IN_TORSO @ OPTICAL_IN_D435
        camera_in_reference = self.left_foot_in_reference @ camera_in_foot
        if self.root_state_source == "estimator":
            camera_in_dynamic = camera_in_foot
        elif self.root_state_source == "mimic":
            camera_in_dynamic = camera_in_reference
        else:  # topic
            root = self.capture_root_state
            pelvis = np.eye(4)
            pelvis[:3, :3] = Rotation.from_quat(root.quaternion, scalar_first=True).as_matrix()
            pelvis[:3, 3] = root.position
            camera_in_dynamic = pelvis @ pelvis_torso @ D435_IN_TORSO @ OPTICAL_IN_D435

        self.obstacles_reference = [box_in_frame(b, camera_in_reference) for b in boxes]
        self.obstacles_dynamic = [box_in_frame(b, camera_in_dynamic) for b in boxes]

        stamp = boxes[0].header.stamp  # the captured camera frame
        dynamic_frame = DYNAMIC_FRAMES[self.root_state_source]
        self.obstacles_reference_pub.publish(
            box_array_msg(self.obstacles_reference, REFERENCE_FRAME, stamp)
        )
        self.obstacles_dynamic_pub.publish(
            box_array_msg(self.obstacles_dynamic, dynamic_frame, stamp)
        )
        transforms = [
            transform_msg(camera_in_reference, REFERENCE_FRAME, boxes[0].header.frame_id, stamp)
        ]
        if self.root_state_source == "estimator":
            transforms.append(
                transform_msg(self.left_foot_in_reference, REFERENCE_FRAME, dynamic_frame, stamp)
            )
        self.tf_broadcaster.sendTransform(transforms)
        self.log_obstacles()

    def log_obstacles(self):
        self.get_logger().info(
            f"{len(self.obstacles_reference)} obstacle boxes, reference frame "
            f"(dynamic filter frame: {self.root_state_source}):"
        )
        for box in self.obstacles_reference:
            yaw = np.degrees(np.arctan2(box["rotation"][1, 0], box["rotation"][0, 0]))
            self.get_logger().info(
                f"  {box['label']} ({box['score']:.2f}): centre "
                f"{np.round(box['centre'], 3).tolist()} m, size "
                f"{np.round(2 * box['half_extents'], 3).tolist()} m, yaw {yaw:.0f} deg"
            )

    def set_motion(self, motion):
        self.motion = motion
        self.motion_q_actuated = motion["q"][:, 6:]
        self.num_motion_steps = motion["q"].shape[0]

    def start_retarget(self):
        """Hand the obstacle boxes (reference frame) to the retarget process"""
        self.retarget_job += 1
        self.retarget_tasks.put(
            (self.retarget_job, self.motion_file, self.control_freq, list(self.obstacles_reference))
        )
        self.motion = None  # the previous motion ignored the new scene
        self.retarget_start_time = self.get_current_time_in_seconds()
        self.max_loop_interval = 0.0
        self.state = State.RETARGET

    def check_retarget(self):
        if not self.retarget_process.is_alive():
            self.capture_failed("the retarget process died")
            return
        try:
            job, result = self.retarget_results.get_nowait()
        except queue.Empty:
            return
        if job != self.retarget_job:
            return  # abandoned job
        elapsed = self.get_current_time_in_seconds() - self.retarget_start_time
        if not isinstance(result, dict):
            self.capture_failed(f"retargeting failed:\n{result}")
            return
        # The retargeting runs in its own process, so this should stay nominal
        if self.max_loop_interval > 2 * self.control_dt:
            self.get_logger().warn(
                f"Control loop interval up to {1e3 * self.max_loop_interval:.0f} ms "
                f"while retargeting (nominal {1e3 * self.control_dt:.0f} ms)"
            )
        self.set_motion(result)
        self.get_logger().info(f"Motion retargeted in {elapsed:.1f} s")
        self.scene_ready()

    def control_loop(self):
        now = time.monotonic()
        if self.last_loop_time is not None:
            self.max_loop_interval = max(self.max_loop_interval, now - self.last_loop_time)
        self.last_loop_time = now

        # Consume any pending operator inputs
        advance, self.advance_requested = self.advance_requested, False
        stop, self.stop_requested = self.stop_requested, False
        estop, self.estop_requested = self.estop_requested, False

        # Handle a kill command immediately
        if estop and self.state != State.KILLED:
            self.get_logger().warn("Emergency stop requested! Killing robot")
            self.estop_pub.publish(Empty())
            self.state = State.KILLED
            # In sim, catch the limp robot on the virtual gantry
            self.set_gantry(True)
        if self.state == State.KILLED:
            return

        if self.last_robot_state is None:
            self.get_logger().info(
                "Waiting for robot state...", throttle_duration_sec=1.0
            )
            return
        if self.state == State.WAIT_FOR_ROBOT:
            self.state = State.IDLE
            self.get_logger().info(
                "Robot state received. Press A to move to the default pose"
            )
            return

        # A stop request (short of an estop) freezes the robot at the last
        # commanded position, with the policy turned off
        if stop and self.hold_pd_target is not None and self.state != State.FINISHED:
            self.get_logger().info(
                "Stop requested: holding the last commanded position"
            )
            self.state = State.FINISHED
            # The position hold is not actively balanced, so re-engage the
            # (virtual or real) gantry support
            self.set_gantry(True)

        if self.state == State.IDLE:
            if advance:
                self.interp_start_q = np.array(self.last_robot_state.q)
                self.interp_start_time = self.get_current_time_in_seconds()
                self.state = State.MOVE_TO_DEFAULT
                self.get_logger().info("Moving to the default pose...")
        elif self.state == State.MOVE_TO_DEFAULT:
            pct = self.interpolation_pct(self.move_to_default_duration)
            self.publish_pd_command(
                linear_interp(self.interp_start_q, self.default_q_actuated, pct)
            )
            if pct >= 1.0:
                self.state = State.HOLD_DEFAULT
                self.get_logger().info(
                    "Holding the default pose. Press A to activate the policy"
                )
        elif self.state == State.HOLD_DEFAULT:
            self.publish_pd_command(self.default_q_actuated)
            if advance:
                self.state = State.TRACK_STAND
                # The policy balances the robot from here on
                self.set_gantry(False)
                self.get_logger().info(
                    "Policy active (standing). Press A to raise the arm"
                )
        elif self.state == State.TRACK_STAND:
            self.track_interpolated_reference(self.stand_step, self.stand_step, 0.0)
            if advance:
                self.start_capture()
        elif self.state == State.CAPTURE_SCENE:
            self.track_interpolated_reference(self.stand_step, self.stand_step, 0.0)
            self.check_capture()
        elif self.state == State.RETARGET:
            self.track_interpolated_reference(self.stand_step, self.stand_step, 0.0)
            self.check_retarget()
        elif self.state == State.SCENE_READY:
            self.track_interpolated_reference(self.stand_step, self.stand_step, 0.0)
            if advance:
                # The dynamic filter runs from the raise on
                if (
                    self.pd_filter is not None
                    and self.root_state_source == "topic"
                    and self.last_root_state is None
                ):
                    self.get_logger().warn(
                        "No root state received yet: the dynamic CBF needs "
                        "/g1_control/root_state (or another root_state_source)"
                    )
                else:
                    self.interp_start_time = self.get_current_time_in_seconds()
                    self.state = State.RAISE_ARM
                    self.get_logger().info("Raising the arm to the start of the motion...")
        elif self.state == State.RAISE_ARM:
            pct = self.interpolation_pct(self.transition_duration)
            self.track_interpolated_reference(
                self.stand_step, self.motion_keyframe(0), pct, apply_filter=True
            )
            if pct >= 1.0:
                self.state = State.HOLD_RAISED
                self.get_logger().info("Arm raised. Press A to run the motion")
        elif self.state == State.HOLD_RAISED:
            self.track_interpolated_reference(
                self.motion_keyframe(0), self.motion_keyframe(0), 0.0, apply_filter=True
            )
            if advance:
                # The root state was checked before the raise
                self.motion_step = 0
                self.state = State.RUN_MOTION
                self.get_logger().info("Running the reference motion...")
        elif self.state == State.RUN_MOTION:
            i = self.motion_step
            self.step_policy(
                self.motion_q_actuated[i],
                self.motion["pos"][i],
                self.motion["quat"][i],
                self.motion["vel"][i],
                self.motion["omega"][i],
                int(self.motion["contact_mode"][i]),
                apply_filter=True,
            )
            self.motion_step += 1
            if self.motion_step >= self.num_motion_steps:
                self.interp_start_time = self.get_current_time_in_seconds()
                self.state = State.LOWER_ARM
                self.get_logger().info("Motion complete. Returning to stand...")
        elif self.state == State.LOWER_ARM:
            pct = self.interpolation_pct(self.transition_duration)
            self.track_interpolated_reference(
                self.motion_keyframe(-1), self.stand_step, pct, apply_filter=True
            )
            if pct >= 1.0:
                self.state = State.TRACK_STAND
                self.get_logger().info(
                    "Standing. Press A to replay the motion, or B to stop and hold"
                )
        elif self.state == State.FINISHED:
            self.publish_pd_command(self.hold_pd_target)
        else:
            raise RuntimeError("Unreachable state")


def linear_interp(start, end, pct):
    return start + pct * (end - start)


def retarget_motion(robot, motion_file, control_freq, obstacles):
    """Retarget the reference motion, keeping the right hand outside the
    obstacle boxes (reference frame) if given"""
    if obstacles:
        kin_cbf = CBF.from_config(BoxObstaclesKinematicCBFConfig(robot, obstacles))
    else:
        kin_cbf = None
    retargeter = StatefulRetargeter(
        PicoToG1Retargeter(
            kin_cbf, robot, use_constrained_integration=True, force_double_support=True
        )
    )
    motion = load_and_retarget_motion(motion_file, control_freq, retargeter)
    return {k: np.asarray(v) if hasattr(v, "shape") else v for k, v in motion.items()}


def retarget_worker(tasks, results):
    """Background process: retarget the motion for each (job, motion_file,
    control_freq, obstacles) task. Being a separate process, it cannot stall the
    control loop the way a thread does (JAX tracing holds the GIL for seconds)"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the parent stops us on exit
    os.nice(10)  # the control loop comes first
    # The retargeting prints its progress; errors are sent back instead
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    robot = load_g1()
    while True:
        job, motion_file, control_freq, obstacles = tasks.get()
        try:
            results.put((job, retarget_motion(robot, motion_file, control_freq, obstacles)))
        except Exception:
            results.put((job, traceback.format_exc()))


def obstacle_arrays(obstacles):
    """Obstacle dicts -> fixed-size (centres, rotations, half_extents, valid)
    arrays for the dynamic filter, keeping the MAX_OBSTACLES best scores"""
    obstacles = sorted(obstacles, key=lambda box: -box["score"])[:MAX_OBSTACLES]
    centres = np.zeros((MAX_OBSTACLES, 3))
    rotations = np.tile(np.eye(3), (MAX_OBSTACLES, 1, 1))
    half_extents = np.full((MAX_OBSTACLES, 3), 0.01)
    valid = np.zeros(MAX_OBSTACLES, bool)
    for i, box in enumerate(obstacles):
        centres[i], rotations[i] = box["centre"], box["rotation"]
        half_extents[i], valid[i] = box["half_extents"], True
    return centres, rotations, half_extents, valid


class BoxObstaclesDynamicCBFConfig(BaseDynamicConfig):
    """Dynamic CBF keeping the right hand collision sphere outside the obstacle
    boxes. Same settings as KarateChopDynamicCBFConfig, with the box signed
    distances in place of the Z_MIN plane. The boxes are a runtime argument
    (obstacles=obstacle_arrays(...)), so the filter compiles only once"""

    def __init__(self, robot):
        super().__init__(
            constrained=True,
            underactuated=True,
            robot=robot,
            include_jdot=True,
            use_naive_objective=False,
            solver_tol=1e-5,
            init_args=None,
            # A box far below the robot: with no box the constraint is constant and
            # cbfpy's setup check complains that Lgh is zero
            init_kwargs={"contact_mode": 3, "obstacles": obstacle_arrays([FAR_BOX])},
        )

    def h_2(self, z, *args, **kwargs):
        q = z[: self.robot.num_joints]
        collision_pos, collision_rad = self.robot.link_collision_data(q)
        centres, rotations, half_extents, valid = kwargs["obstacles"]
        distances = box_signed_distances(
            collision_pos[RIGHT_HAND_SPHERE], centres, rotations, half_extents
        )
        # Unused slots get a constant, inactive constraint
        return jnp.where(valid, distances - collision_rad[RIGHT_HAND_SPHERE], 1.0)

    def alpha(self, h, *args, **kwargs):
        return 10.0 * h

    def alpha_2(self, h, *args, **kwargs):
        return 10.0 * h


@jax.tree_util.register_static
class ObstaclePDFilter(PDFilter):
    """PDFilter that passes the obstacle boxes on to the dynamic CBF"""

    @jax.jit
    def filter(self, pd_target, pos, quat, vel, omega, q_act, qd_act, contact_mode, obstacles):
        q_ff, qd_ff = pose_and_twist_to_virtual_joints(pos, quat, vel, omega)
        return self.filter_from_virtual_joints(
            pd_target, q_ff, qd_ff, q_act, qd_act, contact_mode, obstacles
        )

    @jax.jit
    def filter_from_virtual_joints(
        self, pd_target, q_ff, qd_ff, q_act, qd_act, contact_mode, obstacles
    ):
        z = jnp.concatenate([q_ff, q_act, qd_ff, qd_act])
        u = self.actuator_model.compute_joint_torques_from_only_positional_target(
            q_act, qd_act, pd_target
        )
        u_safe = self.cbf.safety_filter(z, u, contact_mode=contact_mode, obstacles=obstacles)
        return self.actuator_model.get_positional_target_from_torques(q_act, qd_act, u_safe)


class BoxObstaclesKinematicCBFConfig(BaseKinematicConfig):
    """Kinematic CBF keeping the right hand collision sphere outside every
    obstacle box. Same settings as KarateChopKinematicCBFConfig, with the box
    signed distances in place of the Z_MIN plane. The boxes (reference frame)
    are baked in, so the CBF is rebuilt for every capture"""

    def __init__(self, robot, obstacles):
        # Set before the base init, which evaluates h_1 once
        self.box_centres = np.array([box["centre"] for box in obstacles])
        self.box_rotations = np.array([box["rotation"] for box in obstacles])
        self.box_half_extents = np.array([box["half_extents"] for box in obstacles])
        super().__init__(
            constrained=True,
            underactuated=False,
            robot=robot,
            use_naive_objective=False,
            solver_tol=1e-7,
            init_args=None,
            init_kwargs={"contact_mode": 3},
        )

    def h_1(self, z, *args, **kwargs):
        collision_pos, collision_rad = self.robot.link_collision_data(z)
        distances = box_signed_distances(
            collision_pos[RIGHT_HAND_SPHERE],
            self.box_centres,
            self.box_rotations,
            self.box_half_extents,
        )
        return distances - collision_rad[RIGHT_HAND_SPHERE]

    def alpha(self, h, *args, **kwargs):
        return 4.0 * h


def box_signed_distances(point, centres, rotations, half_extents):
    """Signed distance from a point to each oriented box (negative inside).
    Outside a box this is the Euclidean distance to it, which is smooth"""
    local = jnp.einsum("kji,kj->ki", rotations, point - centres)  # R^T (p - c)
    d = jnp.abs(local) - half_extents
    # Small offset keeps the gradient finite when the point is inside
    outside = jnp.sqrt(jnp.sum(jnp.maximum(d, 0.0) ** 2, axis=-1) + 1e-12)
    inside = jnp.minimum(jnp.max(d, axis=-1), 0.0)
    return outside + inside


def torso_in_feet_and_pelvis(robot, q_actuated):
    """torso_link pose in the left foot frame and in the pelvis frame"""
    joint_transforms = robot.joint_to_world_transforms(
        jnp.concatenate([jnp.zeros(6), q_actuated])
    )
    torso = joint_transforms[6 + fixed_root_joint_ordering.index("waist_pitch_joint")]
    left_foot = robot._left_foot_transform(joint_transforms)
    pelvis = joint_transforms[5]
    return jnp.linalg.inv(left_foot) @ torso, jnp.linalg.inv(pelvis) @ torso


def pelvis_in_left_foot(robot, q_actuated):
    """Pelvis pose in the left foot frame"""
    joint_transforms = robot.joint_to_world_transforms(
        jnp.concatenate([jnp.zeros(6), q_actuated])
    )
    left_foot = robot._left_foot_transform(joint_transforms)
    return jnp.linalg.inv(left_foot) @ joint_transforms[5]


def box_in_frame(box, camera_in_frame):
    """g1_control_msgs/Box in the camera frame -> dict in the frame of the transform"""
    p, q, h = box.pose.position, box.pose.orientation, box.half_extents
    rotation = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    return {
        "label": box.label,
        "score": box.score,
        "centre": camera_in_frame[:3, :3] @ [p.x, p.y, p.z] + camera_in_frame[:3, 3],
        "rotation": camera_in_frame[:3, :3] @ rotation,
        "half_extents": np.array([h.x, h.y, h.z]),
    }


def read_bag_capture(path):
    """Last recorded capture in a bag -> {"obstacles": BoxArray (reference frame),
    "scene": {topic: last message} for RViz, "transforms": static transforms}"""
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=path),  # storage format detected from the bag
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if BAG_OBSTACLES_TOPIC not in types:
        raise RuntimeError(f"{path} has no {BAG_OBSTACLES_TOPIC}, record it in camera mode")
    reader.set_filter(
        rosbag2_py.StorageFilter(
            topics=[t for t in (BAG_OBSTACLES_TOPIC, "/tf_static", *BAG_SCENE_TOPICS) if t in types]
        )
    )
    last, transforms = {}, {}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        msg = deserialize_message(data, get_message(types[topic]))
        if topic == "/tf_static":
            transforms.update({t.child_frame_id: t for t in msg.transforms})
        else:
            last[topic] = msg
    if BAG_OBSTACLES_TOPIC not in last:
        raise RuntimeError(f"{path} has no scene capture (no message on {BAG_OBSTACLES_TOPIC})")
    return {
        "obstacles": last.pop(BAG_OBSTACLES_TOPIC),
        "scene": last,
        "transforms": list(transforms.values()),
    }


def box_array_msg(obstacles, frame_id, stamp):
    msg = BoxArray()
    msg.header.frame_id, msg.header.stamp = frame_id, stamp
    for box in obstacles:
        b = Box(header=msg.header, label=box["label"], score=float(box["score"]))
        p, q, h = b.pose.position, b.pose.orientation, b.half_extents
        p.x, p.y, p.z = (float(v) for v in box["centre"])
        q.x, q.y, q.z, q.w = (float(v) for v in Rotation.from_matrix(box["rotation"]).as_quat())
        h.x, h.y, h.z = (float(v) for v in box["half_extents"])
        msg.boxes.append(b)
    return msg


def transform_msg(child_in_parent, parent, child, stamp):
    msg = TransformStamped()
    msg.header.stamp, msg.header.frame_id, msg.child_frame_id = stamp, parent, child
    t, r = msg.transform.translation, msg.transform.rotation
    t.x, t.y, t.z = (float(v) for v in child_in_parent[:3, 3])
    r.x, r.y, r.z, r.w = (float(v) for v in Rotation.from_matrix(child_in_parent[:3, :3]).as_quat())
    return msg


# NOTE: upon trying a few methods, this seemed to be the best way to get the kill command
# sent out on a ctrl+c. But, sometimes there are still race condition issues, so maybe
# there is a better way to handle it (maybe with the C++ command processor node?)
def main(args=None):
    rclpy.init(args=args)
    node = KarateChopNode()

    shutting_down = False

    def sigint_handler(signum, frame):
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True

        node.get_logger().warn("SIGINT received, killing robot")

        for _ in range(3):
            node.robot_command_pub.publish(node.kill_msg)
            rclpy.spin_once(node, timeout_sec=0.05)

        rclpy.shutdown()

    signal.signal(signal.SIGINT, sigint_handler)

    try:
        rclpy.spin(node)
    finally:
        if rclpy.ok():
            node.destroy_node()


if __name__ == "__main__":
    main()
