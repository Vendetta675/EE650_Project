#!/usr/bin/env python3
"""
Mine waypoint node - fixed-mount camera ground projection + TF2.

/camera/debug_annotated is published for EVERY incoming frame, whatever the
state of camera_info, TF or the detector. A status bar burned into the frame
says exactly what is failing.

Camera model: the gimbal is locked straight down, so camera->body is a constant
rotation (param optical_in_body) plus an offset (camera_offset_body). Only
map -> base_link is needed from TF.

Duplicate / ghost handling:
  1. detector NMS with a strict IoU threshold (param iou_thresh)
  2. TF looked up at the IMAGE timestamp (removes latency error while manoeuvring)
  3. detections rejected when the drone is tilted more than max_tilt_deg or the
     box touches the image border
  4. a mine is only published after confirm_obs separate frames agree
     (within assoc_radius_m); its position is the running average
  5. ground-plane dedup against already published mines (min_duplicate_dist_m)

RViz: /mine_markers uses TRANSIENT_LOCAL and every marker is also re-published
at 1 Hz, so RViz shows them even if it is started (or its display re-added)
after the mines were found. Fixed Frame in RViz must be 'map'.

cv_bridge is intentionally NOT used.
"""

import math
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros
from image_geometry import PinholeCameraModel

try:
    from mine_waypoint_pkg.detect import YOLO26LiteRT
except ImportError as e:
    print(f"[FATAL] Could not import YOLO26LiteRT: {e}")
    sys.exit(1)

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 230, 118)
RED = (0, 0, 255)
AMBER = (0, 165, 255)
GREY = (170, 170, 170)

_CHANNELS = {'bgr8': 3, 'rgb8': 3, 'bgra8': 4, 'rgba8': 4, 'mono8': 1}


def quat_to_rot(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]])


def _slerp(q0, q1, a):
    """Interpolate unit quaternions (x, y, z, w) numpy arrays."""
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
    else:
        th = math.acos(d)
        q = (math.sin((1.0 - a) * th) * q0 + math.sin(a * th) * q1) / math.sin(th)
    return q / np.linalg.norm(q)


def image_msg_to_bgr(msg):
    """sensor_msgs/Image -> writable BGR uint8 array (no cv_bridge)."""
    ch = _CHANNELS.get(msg.encoding)
    if ch is None:
        raise ValueError(f"unsupported image encoding '{msg.encoding}'")
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    arr = np.array(buf.reshape(msg.height, msg.step)[:, :msg.width * ch])
    arr = arr.reshape(msg.height, msg.width, ch)
    if msg.encoding == 'bgr8':
        return arr
    if msg.encoding == 'rgb8':
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if msg.encoding == 'rgba8':
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    if msg.encoding == 'bgra8':
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)  # mono8


def bgr_to_image_msg(bgr, header):
    """BGR uint8 array -> sensor_msgs/Image (no cv_bridge)."""
    out = Image()
    out.header = header
    out.height, out.width = int(bgr.shape[0]), int(bgr.shape[1])
    out.encoding = 'bgr8'
    out.is_bigendian = 0
    out.step = out.width * 3
    out.data = np.ascontiguousarray(bgr).tobytes()
    return out


class MineWaypointNode(Node):
    def __init__(self):
        super().__init__('mine_waypoint_node')

        # --- ghost suppression / timing
        # arrival = node clock when the image is received (works when image stamps are sim time
        # but TF is wall time); image = image header stamp; latest = newest TF
        self.declare_parameter('stamp_source', 'arrival')
        self.declare_parameter('tf_timeout_s', 0.5)
        # if TF has not caught up with the image, accept the newest pose if it is at most this old
        self.declare_parameter('max_tf_age_s', 0.6)
        self.declare_parameter('max_tilt_deg', 12.0)
        self.declare_parameter('edge_margin_px', 8)
        self.declare_parameter('confirm_obs', 3)
        self.declare_parameter('assoc_radius_m', 1.0)
        self.declare_parameter('min_duplicate_dist_m', 1.5)

        # --- fixed camera mount
        self.declare_parameter('use_fixed_mount', True)
        self.declare_parameter('body_frame', 'base_link')
        self.declare_parameter('camera_offset_body', [0.0, 0.0, -0.15])  # camera pos in base_link (FLU)
        # optical axes expressed in body FLU, row-major (columns = x_opt, y_opt, z_opt)
        self.declare_parameter('optical_in_body',
                               [0.0, -1.0, 0.0,
                                -1.0, 0.0, 0.0,
                                0.0, 0.0, -1.0])
        self.declare_parameter('ground_z', 0.0)
        # Use /mavros pose (recorded with THIS node's arrival clock) instead of TF stamps.
        # Only applies when use_fixed_mount is true.
        self.declare_parameter('use_pose_topic', True)
        self.declare_parameter('pose_topic', '/mavros/local_position/pose')

        self.declare_parameter('map_frame', 'map')
        # Only used when use_fixed_mount:=false. That frame must be a true optical frame
        # (Z forward, X right, Y down), e.g. 'pitch_link_optical'.
        self.declare_parameter('camera_frame_override', '')
        # Used only while no /camera/camera_info has been received. <= 0 disables it.
        self.declare_parameter('fallback_hfov_deg', 80.0)
        # Only enable if nothing else publishes map -> base_link (double broadcast = jitter).
        self.declare_parameter('publish_tf_from_mavros', False)
        self.declare_parameter('conf_thresh', 0.40)
        self.declare_parameter('iou_thresh', 0.20)
        self.declare_parameter('max_range_m', 50.0)
        self.declare_parameter('marker_republish_hz', 1.0)

        self.conf_thresh = self.get_parameter('conf_thresh').value
        self.detector = YOLO26LiteRT(model_path="640x384", threads=2,
                                     conf_thresh=self.conf_thresh,
                                     iou_thresh=self.get_parameter('iou_thresh').value)

        self.camera_model = PinholeCameraModel()
        self.cam_info_state = 'MISSING'  # MISSING | FALLBACK | OK
        self.known_mines = []
        self.candidates = []
        self.frame_idx = 0
        self.marker_id_counter = 0
        self.markers = []  # every marker ever published, re-sent periodically
        self._first_frame_logged = False
        self.pose_buf = deque(maxlen=256)   # (t_sec, pos[3], quat_xyzw[4]) by arrival time
        self._pose_lock = threading.Lock()
        self.cb_image = MutuallyExclusiveCallbackGroup()
        self.cb_pose = MutuallyExclusiveCallbackGroup()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)

        sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)
        # depth 1 for images: drop stale frames instead of queueing them behind slow inference.
        image_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                               history=HistoryPolicy.KEEP_LAST)
        # latched-style QoS so late RViz subscribers still get the markers
        marker_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                history=HistoryPolicy.KEEP_LAST)

        self.create_subscription(CameraInfo, '/camera_info', self.camera_info_cb, sensor_qos)
        self.create_subscription(Image, '/camera_image', self.image_cb, image_qos,
                                 callback_group=self.cb_image)
        if self.get_parameter('use_pose_topic').value:
            self.create_subscription(PoseStamped, self.get_parameter('pose_topic').value,
                                     self._pose_buf_cb, sensor_qos,
                                     callback_group=self.cb_pose)

        if self.get_parameter('publish_tf_from_mavros').value:
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
            self.create_subscription(PoseStamped, '/mavros/local_position/pose',
                                     self.pose_cb, sensor_qos)

        self.debug_img_pub = self.create_publisher(Image, '/camera/debug_annotated', 10)
        self.mine_loc_pub = self.create_publisher(Point, '/mine_waypoints', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/mine_markers', marker_qos)

        hz = self.get_parameter('marker_republish_hz').value
        if hz > 0:
            self.create_timer(1.0 / hz, self._republish_markers)

        self.get_logger().info("Mine waypoint node initialized. Debug stream: /camera/debug_annotated")

    # ------------------------------------------------------------------ inputs
    def _load_model(self, info: CameraInfo):
        if hasattr(self.camera_model, 'from_camera_info'):
            self.camera_model.from_camera_info(info)
        else:
            self.camera_model.fromCameraInfo(info)

    def camera_info_cb(self, msg: CameraInfo):
        self._load_model(msg)
        self.cam_info_state = 'OK'

    def _init_fallback_model(self, msg: Image):
        hfov = math.radians(self.get_parameter('fallback_hfov_deg').value)
        w, h = msg.width, msg.height
        fx = (w / 2.0) / math.tan(hfov / 2.0)
        cx, cy = w / 2.0, h / 2.0

        info = CameraInfo()
        info.header = msg.header
        info.width, info.height = w, h
        info.distortion_model = 'plumb_bob'
        info.d = [0.0] * 5
        info.k = [fx, 0.0, cx, 0.0, fx, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fx, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self._load_model(info)
        self.cam_info_state = 'FALLBACK'
        self.get_logger().warn(
            f"No /camera/camera_info yet - using fallback intrinsics "
            f"(hfov={self.get_parameter('fallback_hfov_deg').value:.0f} deg, {w}x{h}).")

    def pose_cb(self, msg: PoseStamped):
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = self.get_parameter('map_frame').value
        t.child_frame_id = 'base_link'
        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z
        t.transform.rotation = msg.pose.orientation
        self.tf_broadcaster.sendTransform(t)

    # ------------------------------------------------------------------- main
    def image_cb(self, msg: Image):
        t_arrival = self.get_clock().now()
        if not self._first_frame_logged:
            self._first_frame_logged = True
            self.get_logger().info(
                f"First frame: {msg.width}x{msg.height}, encoding={msg.encoding}, "
                f"frame_id='{msg.header.frame_id}'")

        try:
            frame = image_msg_to_bgr(msg)
        except Exception as e:
            self.get_logger().error(f"Image decode failed: {e}", throttle_duration_sec=2.0)
            return
        h, w = frame.shape[:2]

        # 1. Detector. Errors are shown on the stream, never swallowed.
        dets, det_err = [], None
        try:
            dets, _ = self.detector.predict(frame)
            annotated = self.detector.draw_detections(frame, dets)
        except Exception as e:
            det_err = f"DETECTOR ERROR: {e}"
            annotated = frame.copy()
            self.get_logger().error(det_err, throttle_duration_sec=2.0)

        # 2. Camera model (real camera_info, else fallback intrinsics)
        if self.cam_info_state == 'MISSING' and self.get_parameter('fallback_hfov_deg').value > 0:
            self._init_fallback_model(msg)

        # 3. TF (at the image timestamp; never fall back to "latest")
        if self.get_parameter('use_fixed_mount').value:
            cam_frame = self.get_parameter('body_frame').value
        else:
            override = self.get_parameter('camera_frame_override').value
            cam_frame = override if override else msg.header.frame_id
        map_frame = self.get_parameter('map_frame').value
        trans, tf_err = None, None
        try:
            src = self.get_parameter('stamp_source').value
            if src == 'image':
                when = Time.from_msg(msg.header.stamp)
            elif src == 'arrival':
                when = t_arrival
            else:
                when = None
            if (self.get_parameter('use_pose_topic').value
                    and self.get_parameter('use_fixed_mount').value):
                trans = self._pose_at(when if when is not None else t_arrival, map_frame)
            else:
                trans = self._lookup_tf(map_frame, cam_frame, when)
        except tf2_ros.TransformException as ex:
            tf_err = str(ex)
            self.get_logger().warn(f"TF {map_frame}<-{cam_frame} failed: {ex}",
                                   throttle_duration_sec=2.0)

        # 4. Project every detection, publish only confirmed new ones
        ready = trans is not None and self.cam_info_state != 'MISSING'
        new_count = 0
        margin = self.get_parameter('edge_margin_px').value
        need = self.get_parameter('confirm_obs').value
        self.frame_idx += 1
        for det in dets:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            if x1 <= margin or y1 <= margin or x2 >= w - margin or y2 >= h - margin:
                tag, color = "edge - skipped", GREY
            elif not ready:
                tag, color = "not projected", RED
            else:
                res, why = self._project_to_ground(det["box"], trans)
                if res is None:
                    tag, color = f"no hit: {why}", AMBER
                else:
                    status, info = self._observe(res[0], res[1], det)
                    if status == 'confirmed':
                        new_count += 1
                        tag, color = f"NEW ({info[0]:.1f},{info[1]:.1f})", GREEN
                    elif status == 'pending':
                        tag, color = f"seen {info}/{need} ({res[0]:.1f},{res[1]:.1f})", AMBER
                    else:
                        tag, color = f"dup ({res[0]:.1f},{res[1]:.1f})", GREY
            cv2.putText(annotated, tag, (x1, min(h - 6, y2 + 16)), FONT, 0.5, color, 2)

        # 5. Status bar + debug publish: ALWAYS
        self._draw_status(annotated, dets, new_count, trans is not None,
                          cam_frame, map_frame, tf_err, det_err)
        self.debug_img_pub.publish(bgr_to_image_msg(annotated, msg.header))

    # ---------------------------------------------------------------- helpers
    def _pose_buf_cb(self, msg: PoseStamped):
        """Record each pose with THIS node's clock at receipt (ignores MAVROS header stamps)."""
        p, q = msg.pose.position, msg.pose.orientation
        t = self.get_clock().now().nanoseconds * 1e-9
        with self._pose_lock:
            self.pose_buf.append((t,
                                  np.array([p.x, p.y, p.z], dtype=float),
                                  np.array([q.x, q.y, q.z, q.w], dtype=float)))

    def _pose_at(self, when, map_frame):
        """Pose (as TransformStamped map<-base_link) at node-clock time `when`, interpolated.
        Waits up to tf_timeout_s for a pose newer than `when`; otherwise holds the newest pose
        if it is at most max_tf_age_s old."""
        t = when.nanoseconds * 1e-9
        deadline = time.monotonic() + self.get_parameter('tf_timeout_s').value
        while True:
            with self._pose_lock:
                buf = list(self.pose_buf)
            if (buf and buf[-1][0] >= t) or time.monotonic() > deadline:
                break
            time.sleep(0.005)

        topic = self.get_parameter('pose_topic').value
        if not buf:
            raise tf2_ros.LookupException(f"no pose received yet on {topic}")

        if t >= buf[-1][0]:
            age = t - buf[-1][0]
            limit = self.get_parameter('max_tf_age_s').value
            if age > limit:
                raise tf2_ros.ExtrapolationException(
                    f"extrapolation - pose is {age:.2f}s old (limit {limit:.2f}s)")
            pos, quat = buf[-1][1], buf[-1][2]
        elif t < buf[0][0]:
            raise tf2_ros.ExtrapolationException(
                f"extrapolation - image is {buf[0][0] - t:.2f}s before oldest pose")
        else:
            pos = quat = None
            for i in range(len(buf) - 1):
                t0, p0, q0 = buf[i]
                t1, p1, q1 = buf[i + 1]
                if t0 <= t <= t1:
                    a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                    pos = p0 + a * (p1 - p0)
                    quat = _slerp(q0, q1, a)
                    break
            if pos is None:
                raise tf2_ros.ExtrapolationException("extrapolation - no bracketing pose")

        out = TransformStamped()
        out.header.frame_id = map_frame
        out.child_frame_id = 'base_link'
        out.transform.translation.x = float(pos[0])
        out.transform.translation.y = float(pos[1])
        out.transform.translation.z = float(pos[2])
        out.transform.rotation.x = float(quat[0])
        out.transform.rotation.y = float(quat[1])
        out.transform.rotation.z = float(quat[2])
        out.transform.rotation.w = float(quat[3])
        return out

    def _lookup_tf(self, map_frame, cam_frame, when):
        """TF at time `when`. If TF lags behind `when`, wait up to tf_timeout_s, then accept the
        newest pose when it is at most max_tf_age_s older than `when`. Otherwise raise."""
        if when is None:
            return self.tf_buffer.lookup_transform(map_frame, cam_frame, Time())
        try:
            return self.tf_buffer.lookup_transform(
                map_frame, cam_frame, when,
                timeout=Duration(seconds=self.get_parameter('tf_timeout_s').value))
        except tf2_ros.ExtrapolationException:
            latest = self.tf_buffer.lookup_transform(map_frame, cam_frame, Time())
            age = (when - Time.from_msg(latest.header.stamp)).nanoseconds * 1e-9
            limit = self.get_parameter('max_tf_age_s').value
            if 0.0 <= age <= limit:
                return latest
            raise tf2_ros.ExtrapolationException(
                f"extrapolation - pose is {age:.2f}s old (limit {limit:.2f}s)")

    def _project_to_ground(self, box, trans):
        """Pixel -> ray -> intersect Z=ground_z in map. Returns ((x, y), None) or (None, reason)."""
        x1, y1, x2, y2 = box
        px, py = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ray = np.array(self.camera_model.projectPixelTo3dRay((px, py)), dtype=float)
        ray /= np.linalg.norm(ray)

        R = quat_to_rot(trans.transform.rotation)
        t = np.array([trans.transform.translation.x,
                      trans.transform.translation.y,
                      trans.transform.translation.z])

        use_fixed = self.get_parameter('use_fixed_mount').value
        if use_fixed:
            R_bo = np.array(self.get_parameter('optical_in_body').value).reshape(3, 3)
            axis = R @ R_bo[:, 2]
        else:
            axis = R[:, 2]

        # Tilt gate: angle between the optical axis and straight down
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -axis[2]))))
        if tilt > self.get_parameter('max_tilt_deg').value:
            return None, f"tilt {tilt:.0f} deg"

        if use_fixed:
            off = np.array(self.get_parameter('camera_offset_body').value, dtype=float)
            d = R @ (R_bo @ ray)          # ray direction in map
            o = t + R @ off               # camera origin in map
        else:                             # frame must be a true optical frame
            d = R @ ray
            o = t

        gz = self.get_parameter('ground_z').value
        hgt = o[2] - gz
        if hgt <= 0.0:
            return None, f"camera height {hgt:.2f} (<=0)"
        if d[2] >= -1e-3:
            return None, f"ray not down (z={d[2]:+.2f})"
        g = o + (-hgt / d[2]) * d
        rng = math.hypot(g[0] - o[0], g[1] - o[1])
        if rng > self.get_parameter('max_range_m').value:
            return None, f"too far ({rng:.0f} m)"
        return (float(g[0]), float(g[1])), None

    def _observe(self, gx, gy, det):
        """Multi-frame confirmation. Returns (status, info): dup | pending(n) | confirmed((x, y))."""
        dup_r = self.get_parameter('min_duplicate_dist_m').value
        if any(math.hypot(gx - ex, gy - ey) < dup_r for ex, ey in self.known_mines):
            return 'dup', None

        self.candidates = [c for c in self.candidates if self.frame_idx - c['frame'] < 300]
        assoc = self.get_parameter('assoc_radius_m').value
        best, bd = None, assoc
        for c in self.candidates:
            d = math.hypot(gx - c['x'], gy - c['y'])
            if d < bd:
                best, bd = c, d

        if best is None:
            self.candidates.append({'x': gx, 'y': gy, 'n': 1, 'frame': self.frame_idx,
                                    'cls': det['class_name'], 'conf': det['confidence']})
            return 'pending', 1

        if best['frame'] != self.frame_idx:      # count each frame once
            best['n'] += 1
            best['x'] += (gx - best['x']) / best['n']
            best['y'] += (gy - best['y']) / best['n']
            best['frame'] = self.frame_idx
            best['conf'] = max(best['conf'], det['confidence'])

        if best['n'] >= self.get_parameter('confirm_obs').value:
            self.candidates.remove(best)
            x, y = best['x'], best['y']
            if any(math.hypot(x - ex, y - ey) < dup_r for ex, ey in self.known_mines):
                return 'dup', None
            self._register_mine(x, y, {'class_name': best['cls'], 'confidence': best['conf']})
            return 'confirmed', (x, y)
        return 'pending', best['n']

    def _register_mine(self, gx, gy, det):
        self.known_mines.append((gx, gy))
        self.publish_marker_rviz(gx, gy, det["class_name"], det["confidence"])
        self.mine_loc_pub.publish(Point(x=float(gx), y=float(gy), z=0.0))
        self.get_logger().info(
            f"NEW {det['class_name']} ({det['confidence']:.2f}) at [{gx:.2f}, {gy:.2f}] "
            f"| total {len(self.known_mines)}")

    def _draw_status(self, img, dets, new_count, tf_ok, cam_frame, map_frame, tf_err, det_err):
        state = self.cam_info_state
        if state == 'FALLBACK':
            cam_txt = f"FALLBACK hfov={self.get_parameter('fallback_hfov_deg').value:.0f}"
        else:
            cam_txt = state
        short = cam_frame if len(cam_frame) <= 32 else "..." + cam_frame[-29:]

        if state == 'OK' and tf_ok:
            c1 = GREEN
        elif state != 'MISSING' and tf_ok:
            c1 = AMBER
        else:
            c1 = RED

        if tf_ok:
            tf_txt = 'OK'
        elif tf_err and 'extrapolation' in tf_err.lower():
            tf_txt = 'NO DATA AT IMAGE TIME (lag)'
        else:
            tf_txt = 'NO TRANSFORM'

        lines = [
            (f"CAM_INFO: {cam_txt} | TF {map_frame}<-{short}: {tf_txt}", c1),
            (f"dets: {len(dets)} | new: {new_count} | mines: {len(self.known_mines)} "
             f"| conf>={self.conf_thresh:.2f}", GREEN),
        ]
        if tf_err:
            lines.append((f"TF: {tf_err[:90]}", RED))
        if det_err:
            lines.append((det_err[:90], RED))

        bar_h = 22 * len(lines) + 6
        cv2.rectangle(img, (0, 0), (img.shape[1], bar_h), (20, 20, 20), -1)
        for i, (txt, col) in enumerate(lines):
            cv2.putText(img, txt, (8, 20 + 22 * i), FONT, 0.5, col, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ RViz
    def publish_marker_rviz(self, x, y, cls_name, conf):
        """Create the sphere + label for a new mine, store them, publish everything."""
        map_frame = self.get_parameter('map_frame').value
        idx = self.marker_id_counter
        self.marker_id_counter += 1

        m = Marker()
        m.header.frame_id = map_frame
        m.header.stamp = Time().to_msg()  # stamp 0 = "use latest TF"; immune to sim/wall clock mismatch
        m.ns = 'mines_physical'
        m.id = idx
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), 0.1
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.5
        if cls_name == "mine_disc":
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 0.0, 1.0
        else:
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.65, 0.0, 1.0
        m.lifetime = Duration(seconds=0).to_msg()  # forever

        text = Marker()
        text.header.frame_id = map_frame
        text.header.stamp = Time().to_msg()
        text.ns = 'mines_labels'
        text.id = idx
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x, text.pose.position.y, text.pose.position.z = float(x), float(y), 0.8
        text.pose.orientation.w = 1.0
        text.scale.z = 0.4
        text.color.r = text.color.g = text.color.b = text.color.a = 1.0
        text.text = f"{cls_name}\n({conf * 100:.0f}%)"
        text.lifetime = Duration(seconds=0).to_msg()

        self.markers.extend([m, text])
        self._republish_markers()

    def _republish_markers(self):
        if not self.markers:
            return
        arr = MarkerArray()
        arr.markers = list(self.markers)
        self.marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = MineWaypointNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
