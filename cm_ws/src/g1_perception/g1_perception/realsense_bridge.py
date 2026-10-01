"""Bridge the G1's RealSense ZMQ stream into ROS 2.

realsense_server.py on PC2 answers each ZMQ request with four parts:
colour JPEG, side-by-side IR-left|IR-right JPEG, raw z16 depth and the IR-left
intrinsics as JSON. realsense_server_calib.py adds a fifth part, the colour
intrinsics and the depth->colour extrinsics as JSON. This node polls the server
and republishes the frames as standard sensor_msgs topics.

Depth is registered to IR-left on a D4xx, so depth, infra1 and the point cloud
share one optical frame and one CameraInfo. With the fifth part, colour gets a
CameraInfo and the depth->colour extrinsics are broadcast as a static transform.
The server sends no stereo baseline or capture time, so infra2 has no CameraInfo
and every message is stamped when the reply arrives.
"""

import io
import json
import threading
import time

import numpy as np
import rclpy
import zmq
from geometry_msgs.msg import TransformStamped
from PIL import Image as PILImage
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2, PointField
from tf2_ros import StaticTransformBroadcaster

DEPTH_FRAME = "camera_depth_optical_frame"
COLOR_FRAME = "camera_color_optical_frame"
INFRA2_FRAME = "camera_infra2_optical_frame"

RECV_TIMEOUT_MS = 3000
CLOUD_FIELDS = [
    PointField(name=name, offset=4 * i, datatype=PointField.FLOAT32, count=1)
    for i, name in enumerate(("x", "y", "z", "intensity"))
]


class RealSenseBridge(Node):
    def __init__(self):
        super().__init__("realsense_zmq_bridge")
        self.zmq_addr = self.declare_parameter("zmq_addr", "tcp://192.168.123.164:5556").value
        self.period = 1.0 / self.declare_parameter("rate", 30.0).value
        self.cloud_step = self.declare_parameter("cloud_step", 2).value
        self.z_max = self.declare_parameter("z_max", 6.0).value
        self.publish_cloud = self.declare_parameter("publish_cloud", True).value

        def pub(msg_type, topic):
            return self.create_publisher(msg_type, topic, 2)

        self.pub_color = pub(Image, "/camera/color/image_raw")
        self.pub_color_jpeg = pub(CompressedImage, "/camera/color/image_raw/compressed")
        self.pub_color_info = pub(CameraInfo, "/camera/color/camera_info")
        self.pub_depth = pub(Image, "/camera/depth/image_rect_raw")
        self.pub_depth_info = pub(CameraInfo, "/camera/depth/camera_info")
        self.pub_infra1 = pub(Image, "/camera/infra1/image_rect_raw")
        self.pub_infra1_info = pub(CameraInfo, "/camera/infra1/camera_info")
        self.pub_infra2 = pub(Image, "/camera/infra2/image_rect_raw")
        self.pub_cloud = pub(PointCloud2, "/camera/depth/points")

        self.rays = None  # per-pixel (x/z, y/z), built from the first frame's intrinsics
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.color_calib = None  # last colour calibration JSON, re-broadcast when it changes
        self.frames_since_report = 0
        self.last_report = time.monotonic()

        self.stop_event = threading.Event()
        self.poll_thread = threading.Thread(target=self.poll, daemon=True)
        self.poll_thread.start()
        self.get_logger().info(f"Polling {self.zmq_addr} at up to {1.0 / self.period:.0f} Hz")

    def shutdown(self):
        self.stop_event.set()
        self.poll_thread.join(timeout=RECV_TIMEOUT_MS / 1000 + 1.0)

    def poll(self):
        context = zmq.Context()
        while not self.stop_event.is_set():
            # A REQ socket that timed out cannot send again, so each retry gets a new one.
            socket = context.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, RECV_TIMEOUT_MS)
            socket.connect(self.zmq_addr)
            try:
                self.request_frames(socket)
            except zmq.Again:
                self.get_logger().warn(f"No reply from {self.zmq_addr}, retrying", throttle_duration_sec=5.0)
            finally:
                socket.close()
        context.term()

    def request_frames(self, socket):
        while not self.stop_event.is_set():
            start = time.monotonic()
            socket.send(b"get")
            parts = socket.recv_multipart()
            if len(parts) < 4:
                self.get_logger().warn("Server has no frames yet", throttle_duration_sec=5.0)
                time.sleep(0.2)
                continue
            try:
                self.publish_frame(*parts[:5])
            except Exception:
                # rclpy's SIGINT handler invalidates the context before shutdown() runs.
                if not rclpy.ok():
                    return
                raise
            self.report_rate()
            time.sleep(max(0.0, self.period - (time.monotonic() - start)))

    def publish_frame(self, color_jpeg, infra_jpeg, depth_raw, intrinsics_json, color_calib_json=None):
        stamp = self.get_clock().now().to_msg()
        intrinsics = json.loads(intrinsics_json)
        width, height = intrinsics["width"], intrinsics["height"]

        color = np.asarray(PILImage.open(io.BytesIO(color_jpeg)).convert("RGB"))
        self.pub_color.publish(image_msg(color, "rgb8", stamp, COLOR_FRAME))
        self.pub_color_jpeg.publish(compressed_msg(color_jpeg, stamp, COLOR_FRAME))
        if color_calib_json is not None:
            color_calib = json.loads(color_calib_json)
            self.pub_color_info.publish(camera_info_msg(color_calib, stamp, COLOR_FRAME))
            if color_calib != self.color_calib:
                self.tf_broadcaster.sendTransform(depth_to_color_tf(color_calib, stamp))
                self.color_calib = color_calib

        depth = np.frombuffer(depth_raw, np.uint16).reshape(height, width)
        info = camera_info_msg(intrinsics, stamp, DEPTH_FRAME)
        self.pub_depth.publish(image_msg(depth, "16UC1", stamp, DEPTH_FRAME))
        self.pub_depth_info.publish(info)

        infra = np.asarray(PILImage.open(io.BytesIO(infra_jpeg)).convert("L"))
        infra1, infra2 = infra[:, :width], infra[:, width:]
        self.pub_infra1.publish(image_msg(infra1, "mono8", stamp, DEPTH_FRAME))
        self.pub_infra1_info.publish(info)
        self.pub_infra2.publish(image_msg(infra2, "mono8", stamp, INFRA2_FRAME))

        if self.publish_cloud:
            self.pub_cloud.publish(self.cloud_msg(depth, infra1, intrinsics, stamp))

    def cloud_msg(self, depth, intensity, intrinsics, stamp):
        step = self.cloud_step
        if self.rays is None:
            v, u = np.mgrid[0 : intrinsics["height"] : step, 0 : intrinsics["width"] : step].astype(np.float32)
            self.rays = ((u - intrinsics["cx"]) / intrinsics["fx"], (v - intrinsics["cy"]) / intrinsics["fy"])

        z = depth[::step, ::step].astype(np.float32) * intrinsics["depth_scale"]
        valid = (z > 0.1) & (z < self.z_max)
        z = z[valid]
        points = np.column_stack(
            [self.rays[0][valid] * z, self.rays[1][valid] * z, z, intensity[::step, ::step][valid]]
        ).astype(np.float32)

        msg = PointCloud2(
            height=1,
            width=len(points),
            fields=CLOUD_FIELDS,
            is_bigendian=False,
            point_step=16,
            row_step=16 * len(points),
            is_dense=True,
            data=points.tobytes(),
        )
        msg.header.stamp, msg.header.frame_id = stamp, DEPTH_FRAME
        return msg

    def report_rate(self):
        self.frames_since_report += 1
        elapsed = time.monotonic() - self.last_report
        if elapsed >= 10.0:
            self.get_logger().info(f"{self.frames_since_report / elapsed:.1f} Hz")
            self.frames_since_report, self.last_report = 0, time.monotonic()


def image_msg(array, encoding, stamp, frame_id):
    array = np.ascontiguousarray(array)
    msg = Image(
        height=array.shape[0],
        width=array.shape[1],
        encoding=encoding,
        is_bigendian=0,
        step=array.strides[0],
        data=array.tobytes(),
    )
    msg.header.stamp, msg.header.frame_id = stamp, frame_id
    return msg


def compressed_msg(jpeg, stamp, frame_id):
    msg = CompressedImage(format="jpeg", data=jpeg)
    msg.header.stamp, msg.header.frame_id = stamp, frame_id
    return msg


def camera_info_msg(intrinsics, stamp, frame_id):
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    msg = CameraInfo(
        width=intrinsics["width"],
        height=intrinsics["height"],
        distortion_model="plumb_bob",
        d=[0.0] * 5,
        k=[fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        p=[fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
    )
    msg.header.stamp, msg.header.frame_id = stamp, frame_id
    return msg


def depth_to_color_tf(color_calib, stamp):
    """Pose of the colour optical frame in the depth optical frame.

    The server sends R, t with p_color = R @ p_depth + t; the colour frame's pose in
    the depth frame is the inverse: rotation R^T, origin -R^T t.
    """
    rotation = np.array(color_calib["depth_to_color_rotation"]).reshape(3, 3)
    origin = -rotation.T @ np.array(color_calib["depth_to_color_translation"])
    x, y, z, w = Rotation.from_matrix(rotation.T).as_quat()
    msg = TransformStamped()
    msg.header.stamp, msg.header.frame_id = stamp, DEPTH_FRAME
    msg.child_frame_id = COLOR_FRAME
    t = msg.transform.translation
    t.x, t.y, t.z = origin.tolist()
    r = msg.transform.rotation
    r.x, r.y, r.z, r.w = float(x), float(y), float(z), float(w)
    return msg


def main():
    rclpy.init()
    node = RealSenseBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
