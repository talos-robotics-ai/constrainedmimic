"""Snapshot the RealSense topics on request and segment the colour image with SAM 3.

The node buffers the last few colour images, depth images, depth CameraInfos and
point clouds published by realsense_bridge. A call to /g1_perception/capture_scene
takes the newest frame present on all four topics (the bridge stamps all topics of
a frame identically, but the large cloud arrives later than the images) and runs
SAM 3 on its colour image with the requested text prompts.

The service only returns a summary (labels, scores, frame stamp). The frame and the
packed masks are published latched on /g1_perception/snapshot/*, so a late
subscriber still gets the last snapshot, and a mask overlay on
/g1_perception/overlay for RViz.

SAM 3 is loaded once at startup (~4 GB of GPU memory). The masks are in colour
pixels; mapping them onto depth points needs the colour intrinsics and the
depth->colour extrinsics, which the server does not send yet.
"""

from collections import deque

import numpy as np
import rclpy
import torch
from PIL import Image as PILImage
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from sensor_msgs.msg import CameraInfo, Image, PointCloud2

from g1_control_msgs.srv import CaptureScene

TOPICS = {
    "color": (Image, "/camera/color/image_raw"),
    "depth": (Image, "/camera/depth/image_rect_raw"),
    "depth_info": (CameraInfo, "/camera/depth/camera_info"),
    "cloud": (PointCloud2, "/camera/depth/points"),
}
SNAPSHOT_TOPICS = {key: msg_type for key, (msg_type, _) in TOPICS.items()} | {"label_image": Image}
BUFFER_SIZE = 10  # frames kept per topic, ~1 s at the bridge's rate
OVERLAY_ALPHA = 0.5


class SceneCapture(Node):
    def __init__(self):
        super().__init__("scene_capture")
        self.default_prompts = list(self.declare_parameter("prompts", ["table"]).value)
        self.confidence = self.declare_parameter("confidence", 0.5).value
        self.max_age = self.declare_parameter("max_age", 1.0).value

        self.buffers = {key: deque(maxlen=BUFFER_SIZE) for key in TOPICS}
        for key, (msg_type, topic) in TOPICS.items():
            self.create_subscription(msg_type, topic, lambda msg, buffer=self.buffers[key]: buffer.append(msg), 2)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_snapshot = {
            key: self.create_publisher(msg_type, f"/g1_perception/snapshot/{key}", latched)
            for key, msg_type in SNAPSHOT_TOPICS.items()
        }
        self.pub_overlay = self.create_publisher(Image, "/g1_perception/overlay", latched)

        self.get_logger().info("Loading SAM 3...")
        self.processor = Sam3Processor(build_sam3_image_model(), confidence_threshold=self.confidence)
        self.create_service(CaptureScene, "/g1_perception/capture_scene", self.capture_scene)
        self.get_logger().info(f"Ready, default prompts {self.default_prompts}")

    def capture_scene(self, request, response):
        frames, error = self.newest_frame()
        if error:
            response.success, response.message = False, error
            self.get_logger().warn(f"capture_scene failed: {error}")
            return response

        prompts = list(request.prompts) or self.default_prompts
        color = image_to_array(frames["color"])
        labels, scores, masks = self.segment(color, prompts)

        header = frames["color"].header
        frames["label_image"] = label_image_msg(masks, color.shape[:2], header)
        for key, msg in frames.items():
            self.pub_snapshot[key].publish(msg)
        self.pub_overlay.publish(overlay_msg(color, masks, header))

        response.success = True
        response.message = f"{len(labels)} detections for {prompts}"
        response.stamp = header.stamp
        response.labels, response.scores = labels, scores
        self.get_logger().info(response.message)
        return response

    def newest_frame(self):
        """Newest stamp buffered on every topic -> ({key: msg}, None), or (None, error)."""
        missing = [TOPICS[key][1] for key, buffer in self.buffers.items() if not buffer]
        if missing:
            return None, f"no message yet on {', '.join(missing)}"
        by_stamp = {key: {stamp_ns(msg): msg for msg in buffer} for key, buffer in self.buffers.items()}
        common = set.intersection(*(set(msgs) for msgs in by_stamp.values()))
        if not common:
            return None, "no frame received on all four topics among the buffered ones"
        stamp = max(common)
        age = (self.get_clock().now().nanoseconds - stamp) * 1e-9
        if age > self.max_age:
            return None, f"newest frame is {age:.1f} s old, is realsense_bridge running?"
        return {key: msgs[stamp] for key, msgs in by_stamp.items()}, None

    @torch.inference_mode()
    def segment(self, color, prompts):
        """Run every prompt on one image -> labels, scores and masks sorted by score (descending)."""
        detections = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.processor.set_image(PILImage.fromarray(color))
            for prompt in prompts:
                self.processor.reset_all_prompts(state)
                out = self.processor.set_text_prompt(state=state, prompt=prompt)
                masks = out["masks"].squeeze(1).cpu().numpy()
                scores = out["scores"].float().cpu().numpy()
                detections += [(prompt, float(score), mask) for score, mask in zip(scores, masks)]
        detections.sort(key=lambda d: -d[1])
        return [d[0] for d in detections], [d[1] for d in detections], [d[2] for d in detections]


def stamp_ns(msg):
    return Time.from_msg(msg.header.stamp).nanoseconds


def image_to_array(msg):
    """rgb8 Image -> (height, width, 3) uint8 array."""
    rows = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    return rows[:, : 3 * msg.width].reshape(msg.height, msg.width, 3)


def label_image_msg(masks, shape, header):
    """Pack masks (sorted by score, descending) into one image: 0 = none, i + 1 = mask i."""
    labels = np.zeros(shape, np.uint16)
    for i in reversed(range(len(masks))):  # paint the best mask last so it wins overlaps
        labels[masks[i]] = i + 1
    return array_msg(labels, "16UC1", header)


def overlay_msg(color, masks, header):
    out = color.astype(np.float32)
    rng = np.random.default_rng(0)
    for mask in reversed(masks):
        out[mask] = (1 - OVERLAY_ALPHA) * out[mask] + OVERLAY_ALPHA * rng.uniform(60, 255, 3)
    return array_msg(out.astype(np.uint8), "rgb8", header)


def array_msg(array, encoding, header):
    array = np.ascontiguousarray(array)
    return Image(
        header=header,
        height=array.shape[0],
        width=array.shape[1],
        encoding=encoding,
        is_bigendian=0,
        step=array.strides[0],
        data=array.tobytes(),
    )


def main():
    rclpy.init()
    node = SceneCapture()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
