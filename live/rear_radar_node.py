#!/usr/bin/env python3
"""ROS 2 node: rear-radar car detection, live or from a bag. CPU is enough.

Subscribes to the rear FRGen21 PointCloud2 and the ego odometry, and for every scan runs
``RearRadarDetector`` (training feature code, real 4-scan stacks, the exported
RadarPillar). Publishes, stamped with the scan's header time, in ``rear_axle_middle``:

  rear_radar/detections   vision_msgs/Detection3DArray, class "car" (if vision_msgs exists)
  rear_radar/markers      visualization_msgs/MarkerArray: box outlines + scores, for RViz
  rear_radar/points       PointCloud2 of the current scan as the model sees it
                          (x y z v_abs rcs moving), after the coverage cut
  rear_radar/status       std_msgs/String, JSON: counts, ego speed, latency, stack depth

Boxes are x, y, z centre, l, w, h, yaw about z. Only boxes scoring >= score_threshold go
to detections; markers also draw weaker ones faintly.

    python3 live/rear_radar_node.py --ros-args -p use_sim_time:=true

Parameters: model, radar_topic, odom_topic, score_threshold, device, twist_wait_s,
gate_min_speed, frame_id. See declare_parameter below for defaults.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
import rclpy.executors
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

try:
    from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
except ImportError:  # optional: sudo apt install ros-$ROS_DISTRO-vision-msgs
    Detection3DArray = None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rear_radar_model import RearRadarDetector, builder  # noqa: E402

DEFAULT_MODEL = HERE / "models" / "rear_radarpillar_v5.pt"
STRONG = ColorRGBA(r=0.165, g=0.471, b=0.839, a=1.0)
FAINT = ColorRGBA(r=0.165, g=0.471, b=0.839, a=0.35)
WHITE = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)


def stamp_s(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def yaw_quaternion(yaw: float):
    from geometry_msgs.msg import Quaternion
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))


def cloud(header, points: np.ndarray, names: list[str]) -> PointCloud2:
    points = np.ascontiguousarray(points, dtype=np.float32)
    msg = PointCloud2(header=header, height=1, width=len(points), is_bigendian=False, is_dense=True,
                      point_step=4 * len(names), row_step=4 * len(names) * len(points))
    msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate(names)]
    msg.data = points.tobytes()
    return msg


def box_edges(b: np.ndarray) -> list[Point]:
    """LINE_LIST points for a 3D box outline plus a nose line showing heading."""
    x, y, z, l, w, h, yaw = map(float, b[:7])
    c, s = math.cos(yaw), math.sin(yaw)
    corner = lambda u, v, k: Point(x=x + u * c - v * s, y=y + u * s + v * c, z=z + k)
    bottom = [corner(u, v, -h / 2) for u, v in ((l / 2, w / 2), (-l / 2, w / 2), (-l / 2, -w / 2), (l / 2, -w / 2))]
    top = [corner(u, v, h / 2) for u, v in ((l / 2, w / 2), (-l / 2, w / 2), (-l / 2, -w / 2), (l / 2, -w / 2))]
    pts = []
    for ring in (bottom, top):
        for i in range(4):
            pts += [ring[i], ring[(i + 1) % 4]]
    for i in range(4):
        pts += [bottom[i], top[i]]
    pts += [corner(0, 0, h / 2), corner(l / 2, 0, h / 2)]
    return pts


class RearRadarNode(Node):
    def __init__(self):
        super().__init__("rear_radar_detector")
        p = lambda name, default: self.declare_parameter(name, default).value
        model = p("model", str(DEFAULT_MODEL))
        device = p("device", "")
        self.radar_topic = p("radar_topic", "/radar_rear/frgen21/pointcloud")
        self.odom_topic = p("odom_topic", "/vehicle/uva_odometry")
        self.frame_id = p("frame_id", "rear_axle_middle")
        # Wait this long for odometry past a scan's time, so its twist is interpolated
        # between samples as in the dataset; after that, use the newest sample.
        self.twist_wait_s = float(p("twist_wait_s", 0.1))
        # Training never saw scans below min_ego_speed_mps (static clutter has no Doppler
        # contrast there); publish no detections for them unless this is false.
        self.gate_min_speed = bool(p("gate_min_speed", True))

        self.get_logger().info(f"loading {model}")
        self.det = RearRadarDetector.load(model, device=device or None)
        self.threshold = float(p("score_threshold", self.det.hp["eval_score_threshold"]))
        self.min_speed = float(self.det.features_cfg["min_ego_speed_mps"])
        self.child_frame = self.det.features_cfg["ego_odom_child_frame"]
        info = self.det.info
        self.get_logger().info(f"{info.get('cfg')} epoch {info.get('epoch')} on {self.det.device}, "
                               f"{self.det.num_sweeps} stacked scans, threshold {self.threshold}")

        best_effort = QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(PointCloud2, self.radar_topic, self.on_radar, best_effort)
        self.create_subscription(Odometry, self.odom_topic, self.on_odom,
                                 QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.pub_det = self.create_publisher(Detection3DArray, "rear_radar/detections", 10) if Detection3DArray else None
        if self.pub_det is None:
            self.get_logger().warn("vision_msgs not found: rear_radar/detections is off (markers and status still publish)")
        self.pub_markers = self.create_publisher(MarkerArray, "rear_radar/markers", 10)
        self.pub_points = self.create_publisher(PointCloud2, "rear_radar/points", 10)
        self.pub_status = self.create_publisher(String, "rear_radar/status", 10)

        self.twist: deque[tuple[float, np.ndarray, np.ndarray]] = deque(maxlen=400)
        self.pending: deque[tuple[float, float, PointCloud2]] = deque(maxlen=8)
        self.counts = {"scans": 0, "with_car": 0, "slow": 0, "no_odom": 0, "dropped": 0}
        self.latency = deque(maxlen=160)
        self.last_t = None
        self.create_timer(0.005, self.drain)

    # ------------------------------------------------------------------ inputs
    def on_odom(self, msg: Odometry):
        if self.child_frame and builder.strip_frame(msg.child_frame_id) != self.child_frame:
            self.get_logger().warn(f"odometry child_frame_id {msg.child_frame_id!r}, model expects "
                                   f"{self.child_frame!r}: Doppler compensation will be off",
                                   throttle_duration_sec=10.0)
        lin, ang = msg.twist.twist.linear, msg.twist.twist.angular
        self.twist.append((stamp_s(msg.header.stamp), np.array([lin.x, lin.y, lin.z]), np.array([ang.x, ang.y, ang.z])))

    def on_radar(self, msg: PointCloud2):
        if len(self.pending) == self.pending.maxlen:
            self.counts["dropped"] += 1  # falling behind: drop the oldest, stay live
        self.pending.append((stamp_s(msg.header.stamp), time.monotonic(), msg))

    def twist_at(self, t: float):
        ts = np.array([s[0] for s in self.twist])
        order = np.argsort(ts)
        ts = ts[order]
        lin = np.array([s[1] for s in self.twist])[order]
        ang = np.array([s[2] for s in self.twist])[order]
        return (np.array([np.interp(t, ts, lin[:, k]) for k in range(3)]),
                np.array([np.interp(t, ts, ang[:, k]) for k in range(3)]))

    def drain(self):
        while self.pending:
            t_scan, arrived, msg = self.pending[0]
            newest = self.twist[-1][0] if self.twist else -math.inf
            if newest < t_scan and time.monotonic() - arrived < self.twist_wait_s:
                return
            self.pending.popleft()
            if not self.twist:
                self.counts["no_odom"] += 1
                self.get_logger().warn(f"no odometry on {self.odom_topic} yet; skipping scans", throttle_duration_sec=5.0)
                continue
            self.process(t_scan, msg)

    # ------------------------------------------------------------------ core
    def process(self, t_scan: float, msg: PointCloud2):
        start = time.monotonic()
        if self.last_t is not None and t_scan < self.last_t - 1.0:
            self.get_logger().info("time jumped back (bag looped?): clearing the scan history")
            self.det.reset()
        self.last_t = t_scan
        raw = builder.pointcloud_array(msg, builder.SOURCE_FIELDS)
        v, w = self.twist_at(t_scan)
        d = self.det.step(t_scan, raw, v, w)
        slow = d.ego_speed < self.min_speed
        boxes, scores = d.boxes.copy(), d.scores
        boxes[:, 6] = (boxes[:, 6] + math.pi) % (2 * math.pi) - math.pi  # OpenPCDet leaves yaw unwrapped
        if slow and self.gate_min_speed:
            boxes, scores = boxes[:0], scores[:0]
        strong = scores >= self.threshold

        header = msg.header
        header.frame_id = self.frame_id
        self.publish_markers(header, boxes, scores, strong)
        if self.pub_det is not None:
            self.publish_detections(header, boxes[strong], scores[strong])
        f = d.features
        moving = (np.abs(f[:, 4]) >= 2.0).astype(np.float32)
        self.pub_points.publish(cloud(header, np.column_stack([f[:, :3], f[:, 4], f[:, 7], moving]),
                                      ["x", "y", "z", "v_abs", "rcs", "moving"]))

        latency_ms = (time.monotonic() - start) * 1e3
        self.latency.append(latency_ms)
        c = self.counts
        c["scans"] += 1
        c["with_car"] += bool(strong.any())
        c["slow"] += slow
        cars = [{"x": round(float(b[0]), 2), "y": round(float(b[1]), 2), "yaw": round(float(b[6]), 3),
                 "range_m": round(float(math.hypot(b[0], b[1])), 1), "score": round(float(s), 3)}
                for b, s in zip(boxes[strong], scores[strong])]
        status = {"cars": cars, "ego_speed_mps": round(d.ego_speed, 2), "below_min_speed": bool(slow),
                  "points": int(len(f)), "previous_scans": d.previous, "latency_ms": round(latency_ms, 1),
                  "model_ms": round(d.timings_ms["model"], 1), "counts": c}
        self.pub_status.publish(String(data=json.dumps(status)))
        if c["scans"] % 160 == 0:  # ~10 s at 16 Hz
            self.get_logger().info(
                f"{c['scans']} scans | car in {c['with_car'] / c['scans']:.0%} | median latency "
                f"{np.median(self.latency):.0f} ms | dropped {c['dropped']} | below {self.min_speed:.0f} m/s {c['slow']}")

    def publish_detections(self, header, boxes, scores):
        out = Detection3DArray(header=header)
        for b, s in zip(boxes, scores):
            d = Detection3D(header=header)
            d.bbox.center.position = Point(x=float(b[0]), y=float(b[1]), z=float(b[2]))
            d.bbox.center.orientation = yaw_quaternion(float(b[6]))
            d.bbox.size.x, d.bbox.size.y, d.bbox.size.z = float(b[3]), float(b[4]), float(b[5])
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = "car"
            hyp.hypothesis.score = float(s)
            hyp.pose.pose = d.bbox.center
            d.results = [hyp]
            out.detections.append(d)
        self.pub_det.publish(out)

    def publish_markers(self, header, boxes, scores, strong):
        markers = [Marker(header=header, action=Marker.DELETEALL)]
        life = Duration(sec=0, nanosec=300_000_000)
        for i, (b, s, st) in enumerate(zip(boxes, scores, strong)):
            m = Marker(header=header, ns="boxes", id=i, type=Marker.LINE_LIST, action=Marker.ADD, lifetime=life)
            m.pose.orientation.w = 1.0
            m.scale.x = 0.12 if st else 0.05
            m.color = STRONG if st else FAINT
            m.points = box_edges(b)
            markers.append(m)
            if st:
                t = Marker(header=header, ns="scores", id=i, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD,
                           lifetime=life, text=f"car {s:.2f}  {math.hypot(b[0], b[1]):.0f} m")
                t.pose.position = Point(x=float(b[0]), y=float(b[1]), z=float(b[2]) + float(b[5]) / 2 + 1.0)
                t.pose.orientation.w = 1.0
                t.scale.z = 1.0
                t.color = WHITE
                markers.append(t)
        self.pub_markers.publish(MarkerArray(markers=markers))


def main():
    rclpy.init()
    node = RearRadarNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    except Exception:
        # Jazzy and later shut the context down on SIGINT even while a callback is
        # publishing, which surfaces as RCLError from spin(). Only a live context's error is real.
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
