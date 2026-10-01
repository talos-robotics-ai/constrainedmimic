"""Snapshot the RealSense topics on request, segment them with SAM 3 and fit boxes.

The node buffers the last few colour images, depth images, CameraInfos and point
clouds published by realsense_bridge. A call to /g1_perception/capture_scene:

1. takes the newest frame present on every topic (the bridge stamps all topics of a
   frame identically, but the large cloud arrives later than the images),
2. runs SAM 3 on its colour image with the requested text prompts,
3. back-projects the depth image (per-pixel median over the buffered frames, to cut
   stereo noise; the robot stands still while capturing) and labels every depth
   point with the mask it lands in, via the depth->colour transform from TF,
4. fits the support plane to the points of the best `support_prompt` detection
   (the table) and a box per detection: the table as a slab of `table_thickness`
   below its top surface, objects as boxes standing on the table and rotated only
   about its normal (an oriented bounding box when there is no table),
5. inflates the boxes by `margin` and returns them in the depth optical frame.

The service response holds only the summary and the boxes. Everything is also
published latched, so a bag records the whole capture: the frame and the packed
masks on /g1_perception/snapshot/*, the boxes on /g1_perception/boxes and as
markers on /g1_perception/obstacles, and a mask overlay on /g1_perception/overlay.

SAM 3 is loaded once at startup (~4 GB of GPU memory). The colour calibration
comes from realsense_server_calib.py; with the original server the masks cannot
be mapped onto depth and the service fails.
"""

from collections import deque

import numpy as np
import rclpy
import torch
from PIL import Image as PILImage
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from scipy import ndimage
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from g1_control_msgs.msg import Box, BoxArray
from g1_control_msgs.srv import CaptureScene
from g1_perception import geometry

TOPICS = {
    "color": (Image, "/camera/color/image_raw"),
    "color_info": (CameraInfo, "/camera/color/camera_info"),
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
        self.support_prompt = self.declare_parameter("support_prompt", "table").value
        self.depth_scale = self.declare_parameter("depth_scale", 0.001).value  # z16 units -> m
        self.depth_range = (
            self.declare_parameter("min_depth", 0.15).value,
            self.declare_parameter("max_depth", 3.0).value,
        )
        self.mask_erosion = self.declare_parameter("mask_erosion", 3).value  # px, drops mask edges
        self.min_points = self.declare_parameter("min_points", 50).value
        self.cluster_voxel = self.declare_parameter("cluster_voxel", 0.02).value
        self.plane_threshold = self.declare_parameter("plane_threshold", 0.015).value
        self.min_height = self.declare_parameter("min_height", 0.01).value  # above the table
        self.table_thickness = self.declare_parameter("table_thickness", 0.05).value
        self.margin = self.declare_parameter("margin", 0.02).value

        self.buffers = {key: deque(maxlen=BUFFER_SIZE) for key in TOPICS}
        for key, (msg_type, topic) in TOPICS.items():
            self.create_subscription(msg_type, topic, lambda msg, buffer=self.buffers[key]: buffer.append(msg), 2)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_snapshot = {
            key: self.create_publisher(msg_type, f"/g1_perception/snapshot/{key}", latched)
            for key, msg_type in SNAPSHOT_TOPICS.items()
        }
        self.pub_overlay = self.create_publisher(Image, "/g1_perception/overlay", latched)
        self.pub_markers = self.create_publisher(MarkerArray, "/g1_perception/obstacles", latched)
        self.pub_boxes = self.create_publisher(BoxArray, "/g1_perception/boxes", latched)

        self.get_logger().info("Loading SAM 3...")
        self.processor = Sam3Processor(build_sam3_image_model(), confidence_threshold=self.confidence)
        self.create_service(CaptureScene, "/g1_perception/capture_scene", self.capture_scene)
        self.get_logger().info(f"Ready, default prompts {self.default_prompts}")

    def capture_scene(self, request, response):
        frames, error = self.newest_frame()
        if error is None:
            depth_to_color, error = self.lookup_depth_to_color(frames)
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

        boxes = self.fit_boxes(frames, depth_to_color, labels, scores, masks)
        self.pub_boxes.publish(BoxArray(header=frames["depth"].header, boxes=boxes))
        self.pub_markers.publish(markers_msg(boxes))

        response.success = True
        response.message = f"{len(labels)} detections, {len(boxes)} boxes for {prompts}"
        response.stamp = header.stamp
        response.labels, response.scores, response.boxes = labels, scores, boxes
        self.get_logger().info(response.message)
        return response

    def newest_frame(self):
        """Newest stamp buffered on every topic -> ({key: msg}, None), or (None, error)."""
        missing = [TOPICS[key][1] for key, buffer in self.buffers.items() if not buffer]
        if missing:
            return None, f"no message yet on {', '.join(missing)} (colour calibration needs realsense_server_calib.py)"
        by_stamp = {key: {stamp_ns(msg): msg for msg in buffer} for key, buffer in self.buffers.items()}
        common = set.intersection(*(set(msgs) for msgs in by_stamp.values()))
        if not common:
            return None, "no frame received on all topics among the buffered ones"
        stamp = max(common)
        age = (self.get_clock().now().nanoseconds - stamp) * 1e-9
        if age > self.max_age:
            return None, f"newest frame is {age:.1f} s old, is realsense_bridge running?"
        return {key: msgs[stamp] for key, msgs in by_stamp.items()}, None

    def lookup_depth_to_color(self, frames):
        """(R, t) with p_color = R @ p_depth + t, or an error."""
        try:
            tf = self.tf_buffer.lookup_transform(
                frames["color"].header.frame_id,
                frames["depth"].header.frame_id,
                Time(),
                timeout=Duration(seconds=0.5),
            ).transform
        except TransformException as e:
            return None, f"no depth->colour transform: {e}"
        r, t = tf.rotation, tf.translation
        rotation = Rotation.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
        return (rotation, np.array([t.x, t.y, t.z])), None

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

    def fit_boxes(self, frames, depth_to_color, labels, scores, masks):
        """Depth points of each detection -> inflated Box messages (depth optical frame)."""
        depth_m = self.median_depth(frames["depth"])
        points = geometry.depth_to_points(depth_m, camera_matrix(frames["depth_info"])).reshape(-1, 3)
        points = points[(points[:, 2] > self.depth_range[0]) & (points[:, 2] < self.depth_range[1])]

        # Eroded masks: depth samples on a silhouette often belong to the background
        eroded = [ndimage.binary_erosion(m, iterations=self.mask_erosion) for m in masks]
        label_image = pack_labels(eroded, (frames["color"].height, frames["color"].width))
        point_labels = geometry.label_points(points, *depth_to_color, camera_matrix(frames["color_info"]), label_image)
        detection_points = [points[point_labels == i + 1] for i in range(len(labels))]

        support = next(
            (i for i, label in enumerate(labels)
             if label == self.support_prompt and len(detection_points[i]) >= self.min_points),
            None,
        )
        plane = None
        fitted = []
        if support is not None:
            normal, offset, inliers = geometry.fit_plane(detection_points[support], self.plane_threshold)
            plane = (geometry.plane_rotation(normal), offset)
            top = geometry.largest_cluster(detection_points[support][inliers], self.cluster_voxel)
            fitted.append((support, geometry.box_on_plane(top, *plane, (-self.table_thickness, 0.0))))

        for i, pts in enumerate(detection_points):
            if i == support:
                continue
            pts = geometry.largest_cluster(pts, self.cluster_voxel)
            if plane is not None:
                pts = pts[geometry.heights_above(pts, *plane) > self.min_height]
            if len(pts) < self.min_points:
                self.get_logger().info(f"{labels[i]} ({scores[i]:.2f}): {len(pts)} depth points, no box")
                continue
            if plane is not None:
                top = geometry.heights_above(pts, *plane).max()
                fitted.append((i, geometry.box_on_plane(pts, *plane, (0.0, top))))
            else:
                fitted.append((i, geometry.box_pca(pts)))

        header = frames["depth"].header
        return [
            box_msg(header, labels[i], scores[i], centre, rotation, half + self.margin)
            for i, (centre, rotation, half) in sorted(fitted, key=lambda f: f[0])
        ]

    def median_depth(self, selected):
        """Per-pixel median of the valid samples over the buffered depth frames, in metres."""
        stack = np.stack([depth_to_array(msg) for msg in self.buffers["depth"]
                          if (msg.height, msg.width) == (selected.height, selected.width)])
        stack = np.where(stack > 0, stack.astype(np.float32), np.nan)
        with np.errstate(all="ignore"):
            depth = np.nanmedian(stack, axis=0)
        return np.nan_to_num(depth, nan=0.0) * self.depth_scale


def stamp_ns(msg):
    return Time.from_msg(msg.header.stamp).nanoseconds


def camera_matrix(info):
    return np.array(info.k).reshape(3, 3)


def image_to_array(msg):
    """rgb8 Image -> (height, width, 3) uint8 array."""
    rows = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    return rows[:, : 3 * msg.width].reshape(msg.height, msg.width, 3)


def depth_to_array(msg):
    """16UC1 Image -> (height, width) uint16 array."""
    return np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.step // 2)[:, : msg.width]


def pack_labels(masks, shape):
    """Masks (sorted by score, descending) -> one image: 0 = none, i + 1 = mask i."""
    labels = np.zeros(shape, np.uint16)
    for i in reversed(range(len(masks))):  # paint the best mask last so it wins overlaps
        labels[masks[i]] = i + 1
    return labels


def label_image_msg(masks, shape, header):
    return array_msg(pack_labels(masks, shape), "16UC1", header)


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


def box_msg(header, label, score, centre, rotation, half_extents):
    msg = Box(header=header, label=label, score=float(score))
    p, q, h = msg.pose.position, msg.pose.orientation, msg.half_extents
    p.x, p.y, p.z = (float(v) for v in centre)
    q.x, q.y, q.z, q.w = (float(v) for v in Rotation.from_matrix(rotation).as_quat())
    h.x, h.y, h.z = (float(v) for v in half_extents)
    return msg


def markers_msg(boxes):
    markers = MarkerArray(markers=[Marker(action=Marker.DELETEALL)])
    rng = np.random.default_rng(0)
    for i, box in enumerate(boxes):
        r, g, b = rng.uniform(0.3, 1.0, 3)
        cube = Marker(header=box.header, ns="boxes", id=i, type=Marker.CUBE, pose=box.pose)
        h = box.half_extents
        cube.scale.x, cube.scale.y, cube.scale.z = 2 * h.x, 2 * h.y, 2 * h.z
        cube.color.r, cube.color.g, cube.color.b, cube.color.a = r, g, b, 0.5
        text = Marker(header=box.header, ns="labels", id=i, type=Marker.TEXT_VIEW_FACING, pose=box.pose)
        text.text = f"{box.label} {box.score:.2f}"
        text.scale.z = 0.05
        text.color.r = text.color.g = text.color.b = text.color.a = 1.0
        markers.markers += [cube, text]
    return markers


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
