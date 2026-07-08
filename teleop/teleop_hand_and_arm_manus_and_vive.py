import time
import argparse
import importlib
import json
import math
from dataclasses import dataclass
from multiprocessing import Value, Array, Lock
import threading
import numpy as np
import logging_mp
logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize # dds 
from televuer import TeleVuerWrapper
from teleop.robot_control.robot_arm import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController, H2_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK, H2_ArmIK
from teleimager.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.ipc import IPC_Server
from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper
from sshkeyboard import listen_keyboard, stop_listening

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
def publish_reset_category(category: int, publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")


def _fast_mat_inv(mat):
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret


T_MANUS_TO_UNITREE_HAND_LEFT = np.array([[0, 1, 0, 0],
                                         [0, 0, -1, 0],
                                         [1, 0, 0, 0],
                                         [0, 0, 0, 1]], dtype=float)
T_MANUS_TO_UNITREE_HAND_RIGHT = np.array([[0, 1, 0, 0],
                                          [0, 0, -1, 0],
                                          [-1, 0, 0, 0],
                                          [0, 0, 0, 1]], dtype=float)

TRACKER_WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=float)
TRACKER_TRANSLATION_SCALE = 0.9
CALIBRATION_ORIENTATION_MAX_ERROR_DEG = 45.0

# Tracker-device -> arm EE axis correction. The tracker rotation is first expressed
# in the calibrated tracker frame, matching the translation coordinate conversion.
# Columns are EE axes expressed in tracker coordinates.
R_TRACKER_TO_EE_LEFT = np.array([[0, 0, 1],
                                 [-1, 0, 0],
                                 [0, -1, 0]], dtype=float)
R_TRACKER_TO_EE_RIGHT = np.array([[0, 0, -1],
                                  [-1, 0, 0],
                                  [0, 1, 0]], dtype=float)
R_TRACKER_TO_EE_ABS = {
    "left": R_TRACKER_TO_EE_LEFT,
    "right": R_TRACKER_TO_EE_RIGHT,
}

HAND_POSITION_EES = ("dex3", "inspire_ftp", "inspire_dfx", "brainco")
SIX_DOF_HAND_EES = ("inspire_dfx", "inspire_ftp", "brainco")
BODY_ACTION_SCALE = 0.3


@dataclass
class EndEffectorRuntime:
    controller: object = None
    left_hand_pos_array: object = None
    right_hand_pos_array: object = None
    dual_hand_data_lock: object = None
    dual_hand_state_array: object = None
    dual_hand_action_array: object = None
    dual_hand_raw_action_array: object = None
    left_gripper_value: object = None
    right_gripper_value: object = None
    dual_gripper_data_lock: object = None
    dual_gripper_state_array: object = None
    dual_gripper_action_array: object = None
    left_gripper_trigger_in: object = None
    left_gripper_squeeze_in: object = None
    right_gripper_trigger_in: object = None
    right_gripper_squeeze_in: object = None


def _normalize_ros_msg_type(msg_type):
    if "/msg/" in msg_type:
        return msg_type
    if "." in msg_type:
        parts = msg_type.split(".")
        if len(parts) >= 3 and parts[-2] == "msg":
            return f"{parts[0]}/msg/{parts[-1]}"
    if "/" in msg_type:
        pkg, name = msg_type.split("/", 1)
        return f"{pkg}/msg/{name}"
    raise ValueError(f"ROS2 message type must look like 'pkg/msg/Type': {msg_type}")


def _load_ros_msg_type(msg_type):
    normalized = _normalize_ros_msg_type(msg_type)
    try:
        from rosidl_runtime_py.utilities import get_message
        return get_message(normalized)
    except Exception:
        pkg, _, name = normalized.split("/")
        module = importlib.import_module(f"{pkg}.msg")
        return getattr(module, name)


def _quat_to_rot(qx, qy, qz, qw):
    quat = np.array([qx, qy, qz, qw], dtype=float)
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm < 1e-8:
        return None
    x, y, z, w = quat / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)


def _pose_to_mat(position, orientation):
    rot = _quat_to_rot(orientation.x, orientation.y, orientation.z, orientation.w)
    if rot is None:
        return None
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = [position.x, position.y, position.z]
    return mat


def _point_like_to_xyz(point):
    if hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z"):
        return [point.x, point.y, point.z]
    if hasattr(point, "pose"):
        return _point_like_to_xyz(point.pose)
    if hasattr(point, "position"):
        return _point_like_to_xyz(point.position)
    if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 3:
        return [point[0], point[1], point[2]]
    return None


def _extract_manus_glove_side(msg, topic=None):
    side = getattr(msg, "side", None)
    if isinstance(side, str) and side.lower() in ("left", "right"):
        return side.lower()
    if topic:
        topic_lower = topic.lower()
        if "left" in topic_lower or topic_lower.endswith("_l"):
            return "left"
        if "right" in topic_lower or topic_lower.endswith("_r"):
            return "right"
    return None


def _extract_manus_glove_positions(msg):
    raw_nodes = getattr(msg, "raw_nodes", None)
    if not raw_nodes:
        return None, None

    positions = np.zeros((25, 3), dtype=float)
    wrist_mat = None
    filled = np.zeros(25, dtype=bool)
    for node in raw_nodes:
        node_id = getattr(node, "node_id", None)
        if node_id is None or not (0 <= node_id < 25):
            continue
        pose = getattr(node, "pose", None)
        if pose is None:
            continue
        position = getattr(pose, "position", None)
        if position is None:
            continue
        xyz = _point_like_to_xyz(position)
        if xyz is None:
            continue
        positions[node_id] = xyz
        filled[node_id] = True
        if node_id == 0 and hasattr(pose, "orientation"):
            wrist_mat = _pose_to_mat(pose.position, pose.orientation)

    if not filled[0]:
        return None, None
    return positions, wrist_mat


def _normalize_vec(vec, eps=1e-6):
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if not np.isfinite(norm) or norm < eps:
        return None
    return vec / norm


def _project_rotation(rot):
    if rot is None or not np.all(np.isfinite(rot)):
        return None
    u, _, vt = np.linalg.svd(rot)
    projected = u @ vt
    if np.linalg.det(projected) < 0:
        u[:, -1] *= -1.0
        projected = u @ vt
    return projected


def _rot_to_euler_xyz(rot):
    rot = _project_rotation(rot)
    if rot is None:
        return None
    sy = np.sqrt(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(rot[2, 1], rot[2, 2])
        y = np.arctan2(-rot[2, 0], sy)
        z = np.arctan2(rot[1, 0], rot[0, 0])
    else:
        x = np.arctan2(-rot[1, 2], rot[1, 1])
        y = np.arctan2(-rot[2, 0], sy)
        z = 0.0
    return np.array([x, y, z], dtype=float)


def _rot_to_euler_xyz_deg(rot):
    euler_xyz = _rot_to_euler_xyz(rot)
    if euler_xyz is None:
        return None
    return np.degrees(euler_xyz)


def _rpy_to_rot(rpy):
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=float)


def _rot_to_rpy(rot):
    rot = _project_rotation(rot)
    if rot is None:
        return np.zeros(3, dtype=float)
    pitch = math.asin(max(-1.0, min(1.0, -float(rot[2, 0]))))
    cp = math.cos(pitch)
    if abs(cp) > 1.0e-6:
        roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
        yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(-float(rot[0, 1]), float(rot[1, 1]))
    return np.array([roll, pitch, yaw], dtype=float)


def _rot_to_quat_wxyz(rot):
    rot = _project_rotation(rot)
    if rot is None:
        return [1.0, 0.0, 0.0, 0.0]

    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = [
            0.25 * s,
            (rot[2, 1] - rot[1, 2]) / s,
            (rot[0, 2] - rot[2, 0]) / s,
            (rot[1, 0] - rot[0, 1]) / s,
        ]
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        quat = [
            (rot[2, 1] - rot[1, 2]) / s,
            0.25 * s,
            (rot[0, 1] + rot[1, 0]) / s,
            (rot[0, 2] + rot[2, 0]) / s,
        ]
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        quat = [
            (rot[0, 2] - rot[2, 0]) / s,
            (rot[0, 1] + rot[1, 0]) / s,
            0.25 * s,
            (rot[1, 2] + rot[2, 1]) / s,
        ]
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        quat = [
            (rot[1, 0] - rot[0, 1]) / s,
            (rot[0, 2] + rot[2, 0]) / s,
            (rot[1, 2] + rot[2, 1]) / s,
            0.25 * s,
        ]

    quat = np.asarray(quat, dtype=float)
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm < 1.0e-8:
        return [1.0, 0.0, 0.0, 0.0]
    return (quat / norm).astype(float).tolist()


def _rotation_error_deg(rot_a, rot_b):
    rot_a = _project_rotation(rot_a)
    rot_b = _project_rotation(rot_b)
    if rot_a is None or rot_b is None:
        return None
    rel = _project_rotation(rot_a.T @ rot_b)
    if rel is None:
        return None
    cos_angle = (np.trace(rel) - 1.0) * 0.5
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(angle))


def _fmt_vec(vec):
    return f"[{vec[0]: .3f}, {vec[1]: .3f}, {vec[2]: .3f}]"


def _fmt_mat(mat):
    mat = np.asarray(mat, dtype=float)
    return np.array2string(mat, precision=3, suppress_small=True)


def _fmt_array(arr):
    return np.array2string(np.asarray(arr, dtype=float), precision=3, suppress_small=True)


def _rot_summary_text(rot):
    rot = _project_rotation(rot)
    if rot is None:
        return "invalid"
    euler = _rot_to_euler_xyz_deg(rot)
    angle = _rotation_error_deg(np.eye(3), rot)
    euler_part = "xyz_deg=invalid" if euler is None else f"xyz_deg=[{euler[0]: .1f}, {euler[1]: .1f}, {euler[2]: .1f}]"
    angle_part = "angle_deg=invalid" if angle is None else f"angle_deg={angle:.1f}"
    return f"{euler_part} {angle_part}"


def _frame_axes_text(prefix, rot):
    rot = _project_rotation(rot)
    if rot is None:
        return f"{prefix}_axes=invalid"
    return (
        f"{prefix}_x={_fmt_vec(rot[:, 0])} "
        f"{prefix}_y={_fmt_vec(rot[:, 1])} "
        f"{prefix}_z={_fmt_vec(rot[:, 2])}"
    )


def _pose_frame_text(label, pose):
    if pose is None:
        return f"{label}=invalid"
    rot = _project_rotation(pose[:3, :3])
    euler = _rot_to_euler_xyz_deg(rot)
    euler_part = ""
    if euler is not None:
        euler_part = f" {label}_xyz_deg=[{euler[0]: .1f}, {euler[1]: .1f}, {euler[2]: .1f}]"
    return (
        f"{label}_pos={_fmt_vec(pose[:3, 3])} "
        f"{_frame_axes_text(label, rot)}"
        f"{euler_part}"
    )


def _tracker_rot_to_calib_frame(tracker_rot, tracker_basis):
    tracker_rot = _project_rotation(tracker_rot)
    if tracker_rot is None or tracker_basis is None:
        return None
    return _project_rotation(tracker_basis.T @ tracker_rot)


def _tracker_abs_rot_to_ee_rot(tracker_rot, side, tracker_basis):
    tracker_rot = _tracker_rot_to_calib_frame(tracker_rot, tracker_basis)
    if tracker_rot is None:
        return None
    return _project_rotation(tracker_rot @ R_TRACKER_TO_EE_ABS[side])


class LibsurviveTFReader:
    def __init__(
        self,
        node,
        left_name,
        right_name,
        head_name=None,
        tracking_frame="libsurvive_world",
        stale_timeout=0.5,
    ):
        import tf2_ros

        self._buffer = tf2_ros.Buffer()
        self._listener = tf2_ros.TransformListener(self._buffer, node)
        self._names = {"left": left_name, "right": right_name}
        if head_name:
            self._names["head"] = head_name
        self._tracking_frame = tracking_frame
        self._stale_timeout = stale_timeout
        self._last_ok = {key: 0.0 for key in self._names}
        self._last_poses = {key: None for key in self._names}
        logger_mp.info(
            f"LibsurviveTFReader: left='{left_name}' right='{right_name}' "
            f"head='{head_name}' frame='{tracking_frame}'"
        )

    def _lookup_one(self, side):
        name = self._names.get(side)
        if not name:
            return None, False
        try:
            from rclpy.time import Time as RclpyTime

            transform = self._buffer.lookup_transform(self._tracking_frame, name, RclpyTime())
            pose = _pose_to_mat(transform.transform.translation, transform.transform.rotation)
            if pose is not None:
                self._last_ok[side] = time.monotonic()
                self._last_poses[side] = pose
        except Exception as e:
            now = time.monotonic()
            if now - getattr(self, f"_last_err_{side}", 0.0) > 3.0:
                logger_mp.warning(f"[Vive/TF] {side} lookup failed: {e}")
                setattr(self, f"_last_err_{side}", now)

        now = time.monotonic()
        ok = (now - self._last_ok[side]) <= self._stale_timeout
        return self._last_poses[side], ok

    def read(self):
        left_pose, left_ok = self._lookup_one("left")
        right_pose, right_ok = self._lookup_one("right")
        return left_pose, right_pose, left_ok, right_ok

    def read_head(self):
        if "head" not in self._names:
            return None, False
        return self._lookup_one("head")


class ManusHandReader:
    def __init__(self, node, topics, msg_type, stale_timeout):
        self._lock = threading.Lock()
        self._positions = {"left": None, "right": None}
        self._wrist_mats = {"left": None, "right": None}
        self._stamp = {"left": 0.0, "right": 0.0}
        self._last_warn = {"left": 0.0, "right": 0.0}
        self._stale_timeout = stale_timeout
        ros_msg_type = _load_ros_msg_type(msg_type)
        self._subs = []
        for topic in topics:
            self._subs.append(
                node.create_subscription(
                    ros_msg_type,
                    topic,
                    lambda msg, topic=topic: self._callback(msg, topic),
                    5,
                )
            )
        logger_mp.info(f"ManusHandReader subscribed: topics={topics}, type={msg_type}")

    def _callback(self, msg, topic):
        side = _extract_manus_glove_side(msg, topic)
        if side is None:
            now = time.monotonic()
            if now - getattr(self, "_last_side_warn", 0.0) > 3.0:
                logger_mp.warning("Cannot determine Manus glove side. Set msg.side or use left/right in topic names.")
                self._last_side_warn = now
            return

        positions, wrist_mat = _extract_manus_glove_positions(msg)
        now = time.monotonic()
        if positions is None or positions.shape != (25, 3) or not np.all(np.isfinite(positions)):
            if now - self._last_warn[side] > 2.0:
                logger_mp.warning(f"Cannot parse {side} Manus hand positions as 25 xyz joints.")
                self._last_warn[side] = now
            return

        with self._lock:
            self._positions[side] = positions
            self._wrist_mats[side] = wrist_mat
            self._stamp[side] = now

    def read(self):
        now = time.monotonic()
        with self._lock:
            left_pos = None if self._positions["left"] is None else self._positions["left"].copy()
            right_pos = None if self._positions["right"] is None else self._positions["right"].copy()
            left_wrist = None if self._wrist_mats["left"] is None else self._wrist_mats["left"].copy()
            right_wrist = None if self._wrist_mats["right"] is None else self._wrist_mats["right"].copy()
            left_ok = left_pos is not None and (now - self._stamp["left"]) <= self._stale_timeout
            right_ok = right_pos is not None and (now - self._stamp["right"]) <= self._stale_timeout
        return left_pos, right_pos, left_wrist, right_wrist, left_ok, right_ok


class ViveManusInfoReader:
    def __init__(
        self,
        left_tracker_name,
        right_tracker_name,
        head_tracker_name,
        manus_topics,
        manus_msg_type,
        libsurvive_tracking_frame="libsurvive_world",
        stale_timeout=0.5,
    ):
        import rclpy
        from rclpy.executors import MultiThreadedExecutor

        self._rclpy = rclpy
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init(args=None)

        self.node = rclpy.create_node("vive_manus_info_reader")
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()

        self.vive_reader = LibsurviveTFReader(
            self.node,
            left_name=left_tracker_name,
            right_name=right_tracker_name,
            head_name=head_tracker_name,
            tracking_frame=libsurvive_tracking_frame,
            stale_timeout=stale_timeout,
        )
        self.manus_reader = ManusHandReader(self.node, manus_topics, manus_msg_type, stale_timeout)
        self._tracker_ref = {"left": None, "right": None}
        self._tracker_rot_ref = {"left": None, "right": None}
        self._wrist_ref = {"left": None, "right": None}
        self._last_wrist_pose = {"left": None, "right": None}
        self._tracker_origin = None
        self._tracker_basis = None
        self._head_calib_rot = None
        self._head_calibrated = False
        self._head_last_warn = 0.0

    def close(self):
        try:
            self.executor.shutdown()
        except Exception:
            pass
        try:
            self.node.destroy_node()
        except Exception:
            pass
        if self._owns_rclpy:
            try:
                self._rclpy.shutdown()
            except Exception:
                pass

    def read(self):
        left_tracker, right_tracker, left_tracker_ok, right_tracker_ok = self.vive_reader.read()
        head_tracker, head_tracker_ok = self.vive_reader.read_head()
        left_hand, right_hand, left_wrist, right_wrist, left_hand_ok, right_hand_ok = self.manus_reader.read()
        return {
            "left_tracker": left_tracker,
            "right_tracker": right_tracker,
            "head_tracker": head_tracker,
            "left_tracker_ok": left_tracker_ok,
            "right_tracker_ok": right_tracker_ok,
            "head_tracker_ok": head_tracker_ok,
            "left_hand": left_hand,
            "right_hand": right_hand,
            "left_wrist": left_wrist,
            "right_wrist": right_wrist,
            "left_hand_ok": left_hand_ok,
            "right_hand_ok": right_hand_ok,
        }

    def read_head_tracker_rot(self):
        head_tracker, head_ok = self.vive_reader.read_head()
        if not head_ok or head_tracker is None:
            return None, False

        if self._tracker_basis is not None:
            head_rot = _tracker_rot_to_calib_frame(head_tracker[:3, :3], self._tracker_basis)
        else:
            head_rot = _project_rotation(head_tracker[:3, :3])
        return head_rot, head_rot is not None

    def calibrate_head_camera(self):
        head_rot, head_ok = self.read_head_tracker_rot()
        if not head_ok or head_rot is None:
            now = time.monotonic()
            if now - self._head_last_warn >= 2.0:
                logger_mp.warning("[HeadCam] head tracker not valid, calibration skipped")
                self._head_last_warn = now
            return False

        self._head_calib_rot = head_rot.copy()
        self._head_calibrated = True
        logger_mp.info("[HeadCam] calibrated head-camera neutral rotation")
        return True

    def head_camera_delta_rot(self):
        if not self._head_calibrated or self._head_calib_rot is None:
            if not self.calibrate_head_camera():
                return None, False

        head_rot, head_ok = self.read_head_tracker_rot()
        if not head_ok or head_rot is None:
            return None, False

        return _project_rotation(self._head_calib_rot.T @ head_rot), True

    def calibrate_reference_frame(self, tele_data=None):
        info = self.read()
        if not (
            info["left_tracker_ok"]
            and info["right_tracker_ok"]
            and info["left_tracker"] is not None
            and info["right_tracker"] is not None
        ):
            logger_mp.warning("[Vive/Frame] Cannot calibrate. Need valid left/right trackers.")
            return False

        if tele_data is None:
            logger_mp.warning("[Vive/Frame] Cannot calibrate orientation. Robot EE pose is not available yet.")
            return False

        left_xyz = info["left_tracker"][:3, 3].copy()
        right_xyz = info["right_tracker"][:3, 3].copy()
        origin = 0.5 * (left_xyz + right_xyz)

        z_axis = TRACKER_WORLD_UP.copy()
        side_axis = _normalize_vec(right_xyz - left_xyz)
        if z_axis is None or side_axis is None:
            logger_mp.warning("[Vive/Frame] Cannot calibrate. Tracker geometry is degenerate.")
            return False

        side_axis = _normalize_vec(side_axis - z_axis * float(side_axis @ z_axis))
        if side_axis is None:
            logger_mp.warning("[Vive/Frame] Cannot calibrate. side and z axes are nearly parallel.")
            return False

        x_axis = _normalize_vec(np.cross(z_axis, side_axis))
        if x_axis is None:
            logger_mp.warning("[Vive/Frame] Cannot calibrate. Failed to derive forward x-axis.")
            return False
        y_axis = _normalize_vec(np.cross(z_axis, x_axis))
        if y_axis is None:
            logger_mp.warning("[Vive/Frame] Cannot calibrate. Failed to orthogonalize y-axis.")
            return False

        tracker_basis = np.column_stack([x_axis, y_axis, z_axis])
        orientation_errors = {}
        orientation_debug = {}
        for side in ("left", "right"):
            tracker_raw_rot = _project_rotation(info[f"{side}_tracker"][:3, :3])
            tracker_calib_rot = _tracker_rot_to_calib_frame(info[f"{side}_tracker"][:3, :3], tracker_basis)
            tracker_rot = _tracker_abs_rot_to_ee_rot(info[f"{side}_tracker"][:3, :3], side, tracker_basis)
            wrist_pose = getattr(tele_data, f"{side}_wrist_pose", None)
            wrist_rot = None if wrist_pose is None else wrist_pose[:3, :3]
            error_deg = _rotation_error_deg(tracker_rot, wrist_rot)
            if error_deg is None:
                logger_mp.warning(f"[Vive/Frame] Cannot calibrate. {side} orientation is invalid.")
                return False
            orientation_errors[side] = error_deg
            orientation_debug[side] = (
                tracker_raw_rot,
                tracker_calib_rot,
                tracker_rot,
                _project_rotation(wrist_rot),
            )

        print("[Vive/Frame orientation check]", flush=True)
        for side in ("left", "right"):
            tracker_raw_rot, tracker_calib_rot, tracker_rot, wrist_rot = orientation_debug[side]
            print(
                f"{side} tracker_raw_rot=\n{_fmt_mat(tracker_raw_rot)}\n"
                f"{side} tracker_calib_rot=\n{_fmt_mat(tracker_calib_rot)}\n"
                f"{side} tracker_rot_ee=\n{_fmt_mat(tracker_rot)}\n"
                f"{side} ee_rot=\n{_fmt_mat(wrist_rot)}\n"
                f"{side} rot_error_deg={orientation_errors[side]:.1f}",
                flush=True,
            )

        max_error = max(orientation_errors.values())
        if max_error > CALIBRATION_ORIENTATION_MAX_ERROR_DEG:
            logger_mp.warning(
                "[Vive/Frame] Cannot calibrate. Tracker/EE orientation mismatch is too large: "
                f"left={orientation_errors['left']:.1f}deg "
                f"right={orientation_errors['right']:.1f}deg "
                f"limit={CALIBRATION_ORIENTATION_MAX_ERROR_DEG:.1f}deg"
            )
            return False

        self._tracker_origin = origin
        self._tracker_basis = tracker_basis
        self._tracker_ref["left"] = self._tracker_basis.T @ (left_xyz - origin)
        self._tracker_ref["right"] = self._tracker_basis.T @ (right_xyz - origin)
        self._tracker_rot_ref = {
            "left": orientation_debug["left"][2].copy(),
            "right": orientation_debug["right"][2].copy(),
        }
        self._last_wrist_pose = {"left": None, "right": None}
        self._head_calibrated = False
        self._head_calib_rot = None

        if tele_data is None:
            self._wrist_ref = {"left": None, "right": None}
        else:
            self._wrist_ref = {
                "left": tele_data.left_wrist_pose.copy(),
                "right": tele_data.right_wrist_pose.copy(),
            }

        logger_mp.info(
            "[Vive/Frame] Calibrated from attention pose. "
            f"origin(wrist_mid)={origin.round(3)} "
            f"x(forward)={x_axis.round(3)} "
            f"y(right_to_left)={y_axis.round(3)} "
            f"z(world_up)={z_axis.round(3)} "
            f"orientation_error(left/right)={orientation_errors['left']:.1f}/{orientation_errors['right']:.1f}deg"
        )
        return True

    def _apply_side_relative_motion(self, tele_data, info, side):
        if (
            self._tracker_basis is None
            or self._tracker_origin is None
        ):
            return False

        tracker_pose = info[f"{side}_tracker"]
        tracker_ok = info[f"{side}_tracker_ok"]
        attr = f"{side}_wrist_pose"

        if not tracker_ok or tracker_pose is None:
            if self._last_wrist_pose[side] is not None:
                setattr(tele_data, attr, self._last_wrist_pose[side].copy())
            return False

        current_tracker_xyz = self._tracker_basis.T @ (tracker_pose[:3, 3] - self._tracker_origin)
        current_wrist_pose = getattr(tele_data, attr)
        if self._wrist_ref[side] is None:
            self._wrist_ref[side] = current_wrist_pose.copy()
            logger_mp.info(
                f"[Vive/Relative] {side} wrist_ref={self._wrist_ref[side][:3, 3].round(3)} "
                f"tracker_ref(frame)={self._tracker_ref[side].round(3)}"
            )

        target_wrist_pose = self._wrist_ref[side].copy()
        target_wrist_pose[:3, 3] = (
            self._wrist_ref[side][:3, 3]
            + (current_tracker_xyz - self._tracker_ref[side]) * TRACKER_TRANSLATION_SCALE
        )
        current_tracker_rot = _tracker_abs_rot_to_ee_rot(tracker_pose[:3, :3], side, self._tracker_basis)
        if current_tracker_rot is not None:
            if self._tracker_rot_ref[side] is None:
                self._tracker_rot_ref[side] = current_tracker_rot.copy()
                logger_mp.info(f"[Vive/Relative] {side} tracker_rot_ref set.")
            relative_tracker_rot = self._tracker_rot_ref[side].T @ current_tracker_rot
            target_wrist_rot = _project_rotation(self._wrist_ref[side][:3, :3] @ relative_tracker_rot)
            if target_wrist_rot is not None:
                target_wrist_pose[:3, :3] = target_wrist_rot
        setattr(tele_data, attr, target_wrist_pose)
        self._last_wrist_pose[side] = target_wrist_pose.copy()
        return True

    def apply_relative_wrist_motion(self, tele_data):
        info = self.read()
        left_ready = self._apply_side_relative_motion(tele_data, info, "left")
        right_ready = self._apply_side_relative_motion(tele_data, info, "right")
        return left_ready and right_ready

    def _hand_pos_for_retargeting(self, hand_positions, wrist_mat, valid, side):
        if not valid or hand_positions is None:
            return np.zeros((25, 3))
        if wrist_mat is not None:
            arm = _fast_mat_inv(wrist_mat)
        else:
            wrist = hand_positions[0].copy()
            arm = np.eye(4)
            arm[:3, 3] = -wrist

        t_manus_to_unitree = (
            T_MANUS_TO_UNITREE_HAND_LEFT
            if side == "left"
            else T_MANUS_TO_UNITREE_HAND_RIGHT
        )
        hom = np.concatenate([hand_positions.T, np.ones((1, hand_positions.shape[0]))])
        local = arm @ hom
        return (t_manus_to_unitree @ local)[0:3, :].T

    def apply_manus_hand_data(self, tele_data):
        (
            left_hand_raw,
            right_hand_raw,
            left_wrist_mat,
            right_wrist_mat,
            left_hand_ok,
            right_hand_ok,
        ) = self.manus_reader.read()

        left_hand_pos = self._hand_pos_for_retargeting(
            left_hand_raw, left_wrist_mat, left_hand_ok, "left"
        )
        right_hand_pos = self._hand_pos_for_retargeting(
            right_hand_raw, right_wrist_mat, right_hand_ok, "right"
        )

        hands_ready = left_hand_ok and right_hand_ok
        tele_data.left_hand_pos = left_hand_pos
        tele_data.right_hand_pos = right_hand_pos
        tele_data.left_hand_pinchValue = (
            float(np.linalg.norm(left_hand_pos[4] - left_hand_pos[9]) * 100.0)
            if left_hand_ok
            else 0.0
        )
        tele_data.right_hand_pinchValue = (
            float(np.linalg.norm(right_hand_pos[4] - right_hand_pos[9]) * 100.0)
            if right_hand_ok
            else 0.0
        )
        tele_data.hand_motion_data_ready = hands_ready
        return hands_ready

    def print_tracker_positions(self, tele_data=None):
        info = self.read()
        samples = [
            ("left", info["left_tracker"], info["left_tracker_ok"]),
            ("right", info["right_tracker"], info["right_tracker_ok"]),
            ("head", info["head_tracker"], info["head_tracker_ok"]),
        ]

        frame_xyz_by_side = {}
        for side, pose, valid in samples:
            if pose is None:
                continue
            raw_xyz = pose[:3, 3]
            if self._tracker_basis is not None and self._tracker_origin is not None:
                frame_xyz = self._tracker_basis.T @ (raw_xyz - self._tracker_origin)
                frame_xyz_by_side[side] = frame_xyz

        if self._tracker_basis is None or self._tracker_origin is None:
            print("[tracker calib pos] not calibrated. Press [c] in attention pose.", flush=True)
        else:
            tracker_parts = []
            for side in ("left", "right", "head"):
                valid = info[f"{side}_tracker_ok"]
                frame_xyz = frame_xyz_by_side.get(side)
                if frame_xyz is None:
                    tracker_parts.append(f"{side}: invalid")
                else:
                    status = "" if valid else " stale"
                    tracker_parts.append(f"{side}: {_fmt_vec(frame_xyz)}{status}")
            print("[tracker calib pos] " + " | ".join(tracker_parts), flush=True)

            if "left" in frame_xyz_by_side and "right" in frame_xyz_by_side:
                center = 0.5 * (frame_xyz_by_side["left"] + frame_xyz_by_side["right"])
                span = frame_xyz_by_side["right"] - frame_xyz_by_side["left"]
                print(f"[tracker calib delta] center: {_fmt_vec(center)} | right-left: {_fmt_vec(span)}", flush=True)

        ee_parts = []
        for side in ("left", "right"):
            attr = f"{side}_wrist_pose"
            pose = None
            if tele_data is not None and hasattr(tele_data, attr):
                pose = getattr(tele_data, attr)
            elif self._last_wrist_pose[side] is not None:
                pose = self._last_wrist_pose[side]
            if pose is None:
                ee_parts.append(f"{side}: invalid")
            else:
                ee_parts.append(f"{side}: {_fmt_vec(pose[:3, 3])}")
        print("[robot ee pos] " + " | ".join(ee_parts), flush=True)

        if self._tracker_basis is None:
            print("[tracker rot debug] not calibrated. Press [c] in attention pose.", flush=True)
            return

        for side in ("left", "right"):
            tracker_pose = info[f"{side}_tracker"]
            tracker_ok = info[f"{side}_tracker_ok"]
            if not tracker_ok or tracker_pose is None:
                print(f"[tracker rot debug] {side}: invalid tracker", flush=True)
                continue

            tracker_rot_ee = _tracker_abs_rot_to_ee_rot(tracker_pose[:3, :3], side, self._tracker_basis)
            if tracker_rot_ee is None:
                print(f"[tracker rot debug] {side}: invalid tracker_rot_ee", flush=True)
                continue

            relative_tracker_rot = None
            if self._tracker_rot_ref.get(side) is not None:
                relative_tracker_rot = _project_rotation(self._tracker_rot_ref[side].T @ tracker_rot_ee)

            target_wrist_rot = None
            if relative_tracker_rot is not None and self._wrist_ref.get(side) is not None:
                target_wrist_rot = _project_rotation(self._wrist_ref[side][:3, :3] @ relative_tracker_rot)

            attr = f"{side}_wrist_pose"
            current_wrist_pose = None
            if tele_data is not None and hasattr(tele_data, attr):
                current_wrist_pose = getattr(tele_data, attr)
            elif self._last_wrist_pose[side] is not None:
                current_wrist_pose = self._last_wrist_pose[side]
            current_wrist_rot = None if current_wrist_pose is None else _project_rotation(current_wrist_pose[:3, :3])

            print(
                f"[tracker rot debug] {side}\n"
                f"tracker_rot_ee ({_rot_summary_text(tracker_rot_ee)})=\n{_fmt_mat(tracker_rot_ee)}\n"
                f"relative_tracker_rot ({_rot_summary_text(relative_tracker_rot)})=\n{_fmt_mat(relative_tracker_rot) if relative_tracker_rot is not None else 'invalid'}\n"
                f"target_wrist_rot_calc ({_rot_summary_text(target_wrist_rot)})=\n{_fmt_mat(target_wrist_rot) if target_wrist_rot is not None else 'invalid'}\n"
                f"current_wrist_rot ({_rot_summary_text(current_wrist_rot)})=\n{_fmt_mat(current_wrist_rot) if current_wrist_rot is not None else 'invalid'}",
                flush=True,
            )


class HeadCameraDDSPublisher:
    def __init__(
        self,
        publisher,
        topic,
        camera_name,
        frame,
        mode,
        rate_hz,
        smoothing,
        correction_rpy_deg,
        rpy_scale,
        max_rpy_deg,
        debug=False,
    ):
        self.publisher = publisher
        self.topic = topic
        self.camera_name = camera_name
        self.frame = frame
        self.mode = mode
        self.period = 0.0 if rate_hz <= 0.0 else 1.0 / float(rate_hz)
        self.smoothing = max(0.0, min(0.99, float(smoothing)))
        self.correction_rpy_deg = np.asarray(correction_rpy_deg, dtype=float)
        self.rpy_scale = np.asarray(rpy_scale, dtype=float)
        self.max_rpy = np.deg2rad(np.asarray(max_rpy_deg, dtype=float))
        self.debug = debug
        self.seq = 0
        self.last_publish = 0.0
        self.last_warn = 0.0
        self.last_debug = 0.0

    def recalibrate(self, vive_manus_reader):
        self.last_publish = 0.0
        if vive_manus_reader is None:
            return False
        return vive_manus_reader.calibrate_head_camera()

    def _camera_rot_from_head_delta(self, head_delta_rot):
        tracker_rpy = _rot_to_rpy(head_delta_rot)
        rpy = np.array(
            [
                0.0,
                tracker_rpy[0],
                -tracker_rpy[1],
            ],
            dtype=float,
        ) * self.rpy_scale
        for idx, max_abs in enumerate(self.max_rpy):
            if max_abs > 0.0:
                rpy[idx] = max(-max_abs, min(max_abs, rpy[idx]))
        return _rpy_to_rot(rpy), rpy

    def maybe_publish(self, vive_manus_reader, now=None):
        now = time.monotonic() if now is None else now
        if self.period > 0.0 and (now - self.last_publish) < self.period:
            return False

        if vive_manus_reader is None:
            return False

        head_delta_rot, ok = vive_manus_reader.head_camera_delta_rot()
        if not ok or head_delta_rot is None:
            if now - self.last_warn >= 2.0:
                logger_mp.warning("[HeadCam] skipping camera command; head tracker is not ready")
                self.last_warn = now
            return False

        try:
            camera_rot, rpy = self._camera_rot_from_head_delta(head_delta_rot)
            payload = {
                "camera": self.camera_name,
                "mode": self.mode,
                "frame": self.frame,
                "quat_wxyz": _rot_to_quat_wxyz(camera_rot),
                "smoothing": self.smoothing,
                "seq": self.seq,
                "timestamp": time.time(),
            }
            self.publisher.Write(String_(data=json.dumps(payload, separators=(",", ":"))))
            self.seq += 1
            self.last_publish = now
            if self.debug and now - self.last_debug >= 1.0:
                logger_mp.info(
                    "[HeadCam] published "
                    f"rpy_deg={np.rad2deg(rpy).round(2).tolist()} "
                    f"quat_wxyz={payload['quat_wxyz']}"
                )
                self.last_debug = now
            return True
        except Exception as e:
            if now - self.last_warn >= 2.0:
                logger_mp.warning(f"[HeadCam] failed to publish camera command: {e}")
                self.last_warn = now
            return False


VIVE_MANUS_READER = None
PRINT_TRACKER_POSITION = False
PRINT_IK_ROTATION_DEBUG = False
PRINT_HAND_DEBUG = False
CALIBRATE_TRACKER_FRAME = False
CALIBRATE_TRACKER_FRAME_AT = None
CALIBRATE_TRACKER_FRAME_DELAY = 1.0
CALIBRATE_WAIT_WARN_AT = 0.0

HAND_TIP_IDS = np.array([4, 9, 14, 19, 24], dtype=int)
HAND_TIP_NAMES = ("thumb", "index", "middle", "ring", "pinky")


def maybe_print_tracker_position(tele_data=None):
    global PRINT_TRACKER_POSITION
    if not PRINT_TRACKER_POSITION:
        return
    PRINT_TRACKER_POSITION = False
    if VIVE_MANUS_READER is None:
        logger_mp.warning("[Vive tracker position] reader is not ready yet.")
        return
    try:
        VIVE_MANUS_READER.print_tracker_positions(tele_data=tele_data)
    except Exception as e:
        logger_mp.warning(f"[Vive tracker position] print failed: {e}")


def _current_ee_poses_from_q(arm_ik, q):
    try:
        import pinocchio as pin

        model = arm_ik.reduced_robot.model
        data = model.createData()
        q = np.asarray(q, dtype=float)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        return {
            "left": data.oMf[arm_ik.L_hand_id].homogeneous.copy(),
            "right": data.oMf[arm_ik.R_hand_id].homogeneous.copy(),
        }
    except Exception as e:
        logger_mp.warning(f"[IK rot debug] FK failed: {e}")
        return None


def maybe_print_ik_rotation_debug(tele_data, arm_ik, current_lr_arm_q, sol_q):
    global PRINT_IK_ROTATION_DEBUG
    if not PRINT_IK_ROTATION_DEBUG:
        return
    PRINT_IK_ROTATION_DEBUG = False
    if tele_data is None or arm_ik is None or current_lr_arm_q is None:
        logger_mp.warning("[IK rot debug] missing tele_data, arm_ik, or current arm q")
        return

    sol_poses = _current_ee_poses_from_q(arm_ik, sol_q) if sol_q is not None else None

    print(f"[IK rot debug] sol_q={_fmt_array(sol_q)}", flush=True)
    if sol_q is not None and current_lr_arm_q is not None:
        print(f"[IK rot debug] sol_q-current_q={_fmt_array(np.asarray(sol_q) - np.asarray(current_lr_arm_q))}", flush=True)

    for side in ("left", "right"):
        target_pose = getattr(tele_data, f"{side}_wrist_pose", None)
        target_rot = None if target_pose is None else _project_rotation(target_pose[:3, :3])
        sol_rot = None if sol_poses is None else _project_rotation(sol_poses[side][:3, :3])
        sol_error_deg = _rotation_error_deg(target_rot, sol_rot)
        sol_error_text = "invalid" if sol_error_deg is None else f"{sol_error_deg:.1f}deg"
        print(
            f"[IK rot debug] {side}\n"
            f"target_wrist_rot ({_rot_summary_text(target_rot)})=\n{_fmt_mat(target_rot) if target_rot is not None else 'invalid'}\n"
            f"sol_q_FK_rot ({_rot_summary_text(sol_rot)})=\n{_fmt_mat(sol_rot) if sol_rot is not None else 'invalid'}\n"
            f"target_vs_sol_q_FK_error_deg={sol_error_text}",
            flush=True,
        )


def _hand_joint_labels(args, count):
    if args.ee in ("inspire_dfx", "inspire_ftp"):
        return ["pinky", "ring", "middle", "index", "thumb_bend", "thumb_rot"]
    if args.ee == "dex3":
        return ["thumb0", "thumb1", "thumb2", "middle0", "middle1", "index0", "index1"]
    if args.ee == "brainco":
        return ["thumb_meta", "thumb_prox", "index", "middle", "ring", "pinky"]
    return [f"q{i}" for i in range(count)]


def _fmt_labeled_array(labels, values):
    values = np.asarray(values, dtype=float)
    labels = labels[:len(values)]
    return " ".join(f"{label}={value:.3f}" for label, value in zip(labels, values))


def _hand_tip_debug_text(hand_pos):
    hand_pos = np.asarray(hand_pos, dtype=float)
    if hand_pos.shape[0] <= int(HAND_TIP_IDS[-1]) or hand_pos.shape[1] != 3:
        return "invalid hand_pos shape"
    if not np.all(np.isfinite(hand_pos)):
        return "invalid hand_pos values"

    wrist = hand_pos[0]
    tips = hand_pos[HAND_TIP_IDS]
    tip_norms = np.linalg.norm(tips - wrist, axis=1)
    tip_text = " ".join(
        f"{name}={_fmt_vec(tip)}"
        for name, tip in zip(HAND_TIP_NAMES, tips)
    )
    norm_text = " ".join(
        f"{name}={norm:.3f}"
        for name, norm in zip(HAND_TIP_NAMES, tip_norms)
    )
    return f"wrist={_fmt_vec(wrist)} tip_norms({norm_text}) tips({tip_text})"


def _retarget_ref_norm_text(controller, side, hand_pos):
    hand_retargeting = getattr(controller, "hand_retargeting", None)
    if hand_retargeting is None:
        return "retarget unavailable"
    indices = getattr(hand_retargeting, f"{side}_indices", None)
    if indices is None:
        return "indices unavailable"
    hand_pos = np.asarray(hand_pos, dtype=float)
    ref_value = hand_pos[indices[1, :]] - hand_pos[indices[0, :]]
    ref_norms = np.linalg.norm(ref_value, axis=1)
    return _fmt_array(ref_norms)


def _hand_state_action_snapshot(ee_runtime):
    if ee_runtime is None or ee_runtime.dual_hand_state_array is None or ee_runtime.dual_hand_action_array is None:
        return None, None, None
    lock = ee_runtime.dual_hand_data_lock
    raw_action_array = getattr(ee_runtime, "dual_hand_raw_action_array", None)
    if lock is None:
        state_data = np.asarray(ee_runtime.dual_hand_state_array[:], dtype=float)
        action_data = np.asarray(ee_runtime.dual_hand_action_array[:], dtype=float)
        raw_action_data = None if raw_action_array is None else np.asarray(raw_action_array[:], dtype=float)
        return state_data, action_data, raw_action_data
    with lock:
        state_data = np.asarray(ee_runtime.dual_hand_state_array[:], dtype=float)
        action_data = np.asarray(ee_runtime.dual_hand_action_array[:], dtype=float)
        raw_action_data = None if raw_action_array is None else np.asarray(raw_action_array[:], dtype=float)
        return state_data, action_data, raw_action_data


def maybe_print_hand_debug(args, tele_data, ee_runtime):
    global PRINT_HAND_DEBUG
    if not PRINT_HAND_DEBUG:
        return
    PRINT_HAND_DEBUG = False

    if args.input_mode != "hand" or args.ee not in HAND_POSITION_EES:
        print(f"[hand dbg] unavailable for input_mode={args.input_mode} ee={args.ee}", flush=True)
        return
    if tele_data is None:
        logger_mp.warning("[hand dbg] missing tele_data")
        return

    left_hand_pos = np.asarray(getattr(tele_data, "left_hand_pos", np.zeros((25, 3))), dtype=float)
    right_hand_pos = np.asarray(getattr(tele_data, "right_hand_pos", np.zeros((25, 3))), dtype=float)
    hands_ready = getattr(tele_data, "hand_motion_data_ready", False)
    print(
        "[hand dbg] "
        f"ready={hands_ready} ee={args.ee} "
        f"left_pinch={getattr(tele_data, 'left_hand_pinchValue', 0.0):.3f} "
        f"right_pinch={getattr(tele_data, 'right_hand_pinchValue', 0.0):.3f}",
        flush=True,
    )

    for side, hand_pos in (("left", left_hand_pos), ("right", right_hand_pos)):
        print(f"[hand tips] {side} {_hand_tip_debug_text(hand_pos)}", flush=True)
        print(
            f"[hand ref norms] {side} {_retarget_ref_norm_text(ee_runtime.controller, side, hand_pos)}",
            flush=True,
        )

    state_data, action_data, raw_action_data = _hand_state_action_snapshot(ee_runtime)
    if state_data is None or action_data is None or len(action_data) == 0:
        print("[hand q] state/action unavailable", flush=True)
        return

    half = len(action_data) // 2
    labels = _hand_joint_labels(args, half)
    left_action, right_action = action_data[:half], action_data[half:]
    left_state, right_state = state_data[:half], state_data[half:]
    if raw_action_data is not None and len(raw_action_data) == len(action_data):
        left_raw, right_raw = raw_action_data[:half], raw_action_data[half:]
        print(f"[hand raw q] left {_fmt_labeled_array(labels, left_raw)}", flush=True)
        print(f"[hand raw q] right {_fmt_labeled_array(labels, right_raw)}", flush=True)
    print(f"[hand action] left {_fmt_labeled_array(labels, left_action)}", flush=True)
    print(f"[hand action] right {_fmt_labeled_array(labels, right_action)}", flush=True)
    print(f"[hand state] left {_fmt_labeled_array(labels, left_state)}", flush=True)
    print(f"[hand state] right {_fmt_labeled_array(labels, right_state)}", flush=True)
    print(f"[hand err] left {_fmt_labeled_array(labels, left_action - left_state)}", flush=True)
    print(f"[hand err] right {_fmt_labeled_array(labels, right_action - right_state)}", flush=True)


def maybe_calibrate_tracker_frame(tele_data=None):
    global CALIBRATE_TRACKER_FRAME, CALIBRATE_TRACKER_FRAME_AT, CALIBRATE_WAIT_WARN_AT
    if not CALIBRATE_TRACKER_FRAME:
        return False
    if CALIBRATE_TRACKER_FRAME_AT is not None and time.monotonic() < CALIBRATE_TRACKER_FRAME_AT:
        return False
    if tele_data is None:
        now = time.monotonic()
        if now - CALIBRATE_WAIT_WARN_AT > 1.0:
            logger_mp.warning("[Vive/Frame] Waiting for robot EE pose before calibration.")
            CALIBRATE_WAIT_WARN_AT = now
        return False
    CALIBRATE_TRACKER_FRAME = False
    CALIBRATE_TRACKER_FRAME_AT = None
    if VIVE_MANUS_READER is None:
        logger_mp.warning("[Vive/Frame] reader is not ready yet.")
        return False
    try:
        return VIVE_MANUS_READER.calibrate_reference_frame(tele_data=tele_data)
    except Exception as e:
        logger_mp.warning(f"[Vive/Frame] calibration failed: {e}")
        return False


def maybe_calibrate_tracker_and_head_camera(tele_data=None, head_camera_dds=None):
    global START
    calibrated = maybe_calibrate_tracker_frame(tele_data=tele_data)
    if calibrated:
        if head_camera_dds is not None:
            head_camera_dds.recalibrate(VIVE_MANUS_READER)
        START = False
        logger_mp.info("[Vive/Frame] Calibration complete. Press [r] to sync motion.")
    return calibrated

# state transition
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
#  -------        ---------                -----------                -----------            ---------
#   state          [Ready]      ==>        [Recording]     ==>         [AutoSave]     -->     [Ready]
#  -------        ---------      |         -----------      |         -----------      |     ---------
#   START           True         |manual      True          |manual      True          |        True
#   READY           True         |set         False         |set         False         |auto    True
#   RECORD_RUNNING  False        |to          True          |to          False         |        False
#                                ∨                          ∨                          ∨
#   RECORD_TOGGLE   False       True          False        True          False                  False
#  -------        ---------                -----------                 -----------            ---------
#  ==> manual: when READY is True, set RECORD_TOGGLE=True to transition.
#  --> auto  : Auto-transition after saving data.

def on_press(key):
    global STOP, START, RECORD_TOGGLE, PRINT_TRACKER_POSITION, PRINT_IK_ROTATION_DEBUG, PRINT_HAND_DEBUG, CALIBRATE_TRACKER_FRAME, CALIBRATE_TRACKER_FRAME_AT
    if key == 'r':
        if CALIBRATE_TRACKER_FRAME:
            logger_mp.warning("[Vive/Frame] Calibration is pending. Press [r] again after calibration completes.")
            return
        START = True
        PRINT_TRACKER_POSITION = True
        PRINT_IK_ROTATION_DEBUG = True
        PRINT_HAND_DEBUG = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 's' and START == True:
        RECORD_TOGGLE = True
    elif key == 'c':
        START = False
        CALIBRATE_TRACKER_FRAME = True
        CALIBRATE_TRACKER_FRAME_AT = time.monotonic() + CALIBRATE_TRACKER_FRAME_DELAY
        logger_mp.info(
            f"[Vive/Frame] Calibration scheduled in {CALIBRATE_TRACKER_FRAME_DELAY:.1f}s. "
            "Motion sync is paused until [r]."
        )
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, READY
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
    }


def _init_arm_stack(args):
    if args.arm == "G1_29":
        return G1_29_ArmIK(), G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
    if args.arm == "G1_23":
        return G1_23_ArmIK(), G1_23_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
    if args.arm == "H1_2":
        return H1_2_ArmIK(), H1_2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
    if args.arm == "H1":
        return H1_ArmIK(), H1_ArmController(simulation_mode=args.sim)
    if args.arm == "H2":
        return H2_ArmIK(), H2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
    raise ValueError(f"Unsupported arm type: {args.arm}")


def _new_dual_hand_buffers(array_len):
    return (
        Array('d', 75, lock=True),       # [input] left hand position
        Array('d', 75, lock=True),       # [input] right hand position
        Lock(),
        Array('d', array_len, lock=False),
        Array('d', array_len, lock=False),
    )


def _init_end_effector(args, xr_motion_data_ready):
    if args.ee in ("dex3", "inspire_ftp", "inspire_dfx") and args.input_mode == "controller":
        raise ValueError(f"{args.ee} does not support controller input mode.")

    runtime = EndEffectorRuntime()

    if args.ee == "dex3":
        from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller

        (
            runtime.left_hand_pos_array,
            runtime.right_hand_pos_array,
            runtime.dual_hand_data_lock,
            runtime.dual_hand_state_array,
            runtime.dual_hand_action_array,
        ) = _new_dual_hand_buffers(14)
        runtime.controller = Dex3_1_Controller(
            runtime.left_hand_pos_array,
            runtime.right_hand_pos_array,
            runtime.dual_hand_data_lock,
            runtime.dual_hand_state_array,
            runtime.dual_hand_action_array,
            simulation_mode=args.sim,
            xr_motion_data_ready_in=xr_motion_data_ready,
        )
    elif args.ee == "dex1":
        from teleop.robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller

        runtime.left_gripper_value = Value('d', 0.0, lock=True)
        runtime.right_gripper_value = Value('d', 0.0, lock=True)
        runtime.dual_gripper_data_lock = Lock()
        runtime.dual_gripper_state_array = Array('d', 2, lock=False)
        runtime.dual_gripper_action_array = Array('d', 2, lock=False)
        runtime.controller = Dex1_1_Gripper_Controller(
            runtime.left_gripper_value,
            runtime.right_gripper_value,
            runtime.dual_gripper_data_lock,
            runtime.dual_gripper_state_array,
            runtime.dual_gripper_action_array,
            simulation_mode=args.sim,
            xr_motion_data_ready_in=xr_motion_data_ready,
        )
    elif args.ee == "inspire_dfx":
        from teleop.robot_control.robot_hand_inspire import Inspire_Controller_DFX

        (
            runtime.left_hand_pos_array,
            runtime.right_hand_pos_array,
            runtime.dual_hand_data_lock,
            runtime.dual_hand_state_array,
            runtime.dual_hand_action_array,
        ) = _new_dual_hand_buffers(12)
        runtime.dual_hand_raw_action_array = Array('d', 12, lock=False)
        runtime.controller = Inspire_Controller_DFX(
            runtime.left_hand_pos_array,
            runtime.right_hand_pos_array,
            runtime.dual_hand_data_lock,
            runtime.dual_hand_state_array,
            runtime.dual_hand_action_array,
            runtime.dual_hand_raw_action_array,
            simulation_mode=args.sim,
            xr_motion_data_ready_in=xr_motion_data_ready,
        )
    elif args.ee == "inspire_ftp":
        from teleop.robot_control.robot_hand_inspire import Inspire_Controller_FTP

        (
            runtime.left_hand_pos_array,
            runtime.right_hand_pos_array,
            runtime.dual_hand_data_lock,
            runtime.dual_hand_state_array,
            runtime.dual_hand_action_array,
        ) = _new_dual_hand_buffers(12)
        runtime.dual_hand_raw_action_array = Array('d', 12, lock=False)
        runtime.controller = Inspire_Controller_FTP(
            runtime.left_hand_pos_array,
            runtime.right_hand_pos_array,
            runtime.dual_hand_data_lock,
            runtime.dual_hand_state_array,
            runtime.dual_hand_action_array,
            runtime.dual_hand_raw_action_array,
            simulation_mode=args.sim,
            xr_motion_data_ready_in=xr_motion_data_ready,
        )
    elif args.ee == "brainco" and args.input_mode == "hand":
        from teleop.robot_control.robot_hand_brainco import Brainco_Controller_hand

        (
            runtime.left_hand_pos_array,
            runtime.right_hand_pos_array,
            runtime.dual_hand_data_lock,
            runtime.dual_hand_state_array,
            runtime.dual_hand_action_array,
        ) = _new_dual_hand_buffers(12)
        runtime.controller = Brainco_Controller_hand(
            runtime.left_hand_pos_array,
            runtime.right_hand_pos_array,
            runtime.dual_hand_data_lock,
            runtime.dual_hand_state_array,
            runtime.dual_hand_action_array,
            simulation_mode=args.sim,
            xr_motion_data_ready_in=xr_motion_data_ready,
        )
    elif args.ee == "brainco" and args.input_mode == "controller":
        from teleop.robot_control.robot_hand_brainco import Brainco_Controller_ctrl

        runtime.left_gripper_trigger_in = Value('d', 10.0, lock=True)
        runtime.left_gripper_squeeze_in = Value('d', 0.0, lock=True)
        runtime.right_gripper_trigger_in = Value('d', 10.0, lock=True)
        runtime.right_gripper_squeeze_in = Value('d', 0.0, lock=True)
        runtime.dual_hand_data_lock = Lock()
        runtime.dual_hand_state_array = Array('d', 12, lock=False)
        runtime.dual_hand_action_array = Array('d', 12, lock=False)
        runtime.controller = Brainco_Controller_ctrl(
            runtime.left_gripper_trigger_in,
            runtime.left_gripper_squeeze_in,
            runtime.right_gripper_trigger_in,
            runtime.right_gripper_squeeze_in,
            runtime.dual_hand_data_lock,
            runtime.dual_hand_state_array,
            runtime.dual_hand_action_array,
            simulation_mode=args.sim,
            xr_motion_data_ready_in=xr_motion_data_ready,
        )

    return runtime


def _copy_shared_array(shared_array, values):
    with shared_array.get_lock():
        shared_array[:] = values


def _set_shared_value(shared_value, value):
    with shared_value.get_lock():
        shared_value.value = value


def _controller_body_action(tele_data):
    return [
        -tele_data.left_ctrl_thumbstickValue[1] * BODY_ACTION_SCALE,
        -tele_data.left_ctrl_thumbstickValue[0] * BODY_ACTION_SCALE,
        -tele_data.right_ctrl_thumbstickValue[0] * BODY_ACTION_SCALE,
    ]


def _apply_end_effector_inputs(args, tele_data, ee_runtime):
    if args.ee in HAND_POSITION_EES and args.input_mode == "hand":
        _copy_shared_array(ee_runtime.left_hand_pos_array, tele_data.left_hand_pos.flatten())
        _copy_shared_array(ee_runtime.right_hand_pos_array, tele_data.right_hand_pos.flatten())
    elif args.ee == "brainco" and args.input_mode == "controller":
        _set_shared_value(ee_runtime.left_gripper_trigger_in, tele_data.left_ctrl_triggerValue)
        _set_shared_value(ee_runtime.left_gripper_squeeze_in, tele_data.left_ctrl_squeezeValue)
        _set_shared_value(ee_runtime.right_gripper_trigger_in, tele_data.right_ctrl_triggerValue)
        _set_shared_value(ee_runtime.right_gripper_squeeze_in, tele_data.right_ctrl_squeezeValue)
    elif args.ee == "dex1" and args.input_mode == "controller":
        _set_shared_value(ee_runtime.left_gripper_value, tele_data.left_ctrl_triggerValue)
        _set_shared_value(ee_runtime.right_gripper_value, tele_data.right_ctrl_triggerValue)
    elif args.ee == "dex1" and args.input_mode == "hand":
        _set_shared_value(ee_runtime.left_gripper_value, tele_data.left_hand_pinchValue)
        _set_shared_value(ee_runtime.right_gripper_value, tele_data.right_hand_pinchValue)


def _read_end_effector_record_data(args, tele_data, arm_ctrl, ee_runtime):
    if args.ee == "dex3" and args.input_mode == "hand":
        with ee_runtime.dual_hand_data_lock:
            return (
                ee_runtime.dual_hand_state_array[:7],
                ee_runtime.dual_hand_state_array[-7:],
                ee_runtime.dual_hand_action_array[:7],
                ee_runtime.dual_hand_action_array[-7:],
                [],
                [],
            )
    if args.ee == "dex1" and args.input_mode == "hand":
        with ee_runtime.dual_gripper_data_lock:
            return (
                [ee_runtime.dual_gripper_state_array[0]],
                [ee_runtime.dual_gripper_state_array[1]],
                [ee_runtime.dual_gripper_action_array[0]],
                [ee_runtime.dual_gripper_action_array[1]],
                [],
                [],
            )
    if args.ee == "dex1" and args.input_mode == "controller":
        with ee_runtime.dual_gripper_data_lock:
            return (
                [ee_runtime.dual_gripper_state_array[0]],
                [ee_runtime.dual_gripper_state_array[1]],
                [ee_runtime.dual_gripper_action_array[0]],
                [ee_runtime.dual_gripper_action_array[1]],
                arm_ctrl.get_current_motor_q().tolist(),
                _controller_body_action(tele_data),
            )
    if args.ee in SIX_DOF_HAND_EES and args.input_mode == "hand":
        with ee_runtime.dual_hand_data_lock:
            return (
                ee_runtime.dual_hand_state_array[:6],
                ee_runtime.dual_hand_state_array[-6:],
                ee_runtime.dual_hand_action_array[:6],
                ee_runtime.dual_hand_action_array[-6:],
                [],
                [],
            )
    if args.ee == "brainco" and args.input_mode == "controller":
        with ee_runtime.dual_hand_data_lock:
            return (
                ee_runtime.dual_hand_state_array[:6],
                ee_runtime.dual_hand_state_array[-6:],
                ee_runtime.dual_hand_action_array[:6],
                ee_runtime.dual_hand_action_array[-6:],
                arm_ctrl.get_current_motor_q().tolist(),
                _controller_body_action(tele_data),
            )
    return [], [], [], [], [], []


def _collect_record_colors(camera_config, head_img, left_wrist_img, right_wrist_img):
    colors = {}
    if camera_config['head_camera']['binocular']:
        if head_img is not None:
            half_width = camera_config['head_camera']['image_shape'][1] // 2
            colors["color_0"] = head_img.bgr[:, :half_width]
            colors["color_1"] = head_img.bgr[:, half_width:]
        else:
            logger_mp.warning("Head image is None!")
        if camera_config['left_wrist_camera']['enable_zmq']:
            if left_wrist_img is not None:
                colors["color_2"] = left_wrist_img.bgr
            else:
                logger_mp.warning("Left wrist image is None!")
        if camera_config['right_wrist_camera']['enable_zmq']:
            if right_wrist_img is not None:
                colors["color_3"] = right_wrist_img.bgr
            else:
                logger_mp.warning("Right wrist image is None!")
    else:
        if head_img is not None:
            colors["color_0"] = head_img.bgr
        else:
            logger_mp.warning("Head image is None!")
        if camera_config['left_wrist_camera']['enable_zmq']:
            if left_wrist_img is not None:
                colors["color_1"] = left_wrist_img.bgr
            else:
                logger_mp.warning("Left wrist image is None!")
        if camera_config['right_wrist_camera']['enable_zmq']:
            if right_wrist_img is not None:
                colors["color_2"] = right_wrist_img.bgr
            else:
                logger_mp.warning("Right wrist image is None!")
    return colors


def _build_record_state_action(
    current_lr_arm_q,
    sol_q,
    left_ee_state,
    right_ee_state,
    left_hand_action,
    right_hand_action,
    current_body_state,
    current_body_action,
):
    left_arm_state = current_lr_arm_q[:7]
    right_arm_state = current_lr_arm_q[-7:]
    left_arm_action = sol_q[:7]
    right_arm_action = sol_q[-7:]
    states = {
        "left_arm": {
            "qpos": left_arm_state.tolist(),
            "qvel": [],
            "torque": [],
        },
        "right_arm": {
            "qpos": right_arm_state.tolist(),
            "qvel": [],
            "torque": [],
        },
        "left_ee": {
            "qpos": left_ee_state,
            "qvel": [],
            "torque": [],
        },
        "right_ee": {
            "qpos": right_ee_state,
            "qvel": [],
            "torque": [],
        },
        "body": {
            "qpos": current_body_state,
        },
    }
    actions = {
        "left_arm": {
            "qpos": left_arm_action.tolist(),
            "qvel": [],
            "torque": [],
        },
        "right_arm": {
            "qpos": right_arm_action.tolist(),
            "qvel": [],
            "torque": [],
        },
        "left_ee": {
            "qpos": left_hand_action,
            "qvel": [],
            "torque": [],
        },
        "right_ee": {
            "qpos": right_hand_action,
            "qvel": [],
            "torque": [],
        },
        "body": {
            "qpos": current_body_action,
        },
    }
    return states, actions


def _apply_vive_manus_input(tele_data, head_camera_dds=None):
    if VIVE_MANUS_READER is None:
        return

    maybe_calibrate_tracker_and_head_camera(tele_data=tele_data, head_camera_dds=head_camera_dds)
    if not START or CALIBRATE_TRACKER_FRAME:
        tele_data.arm_motion_data_ready = False
        tele_data.motion_data_ready = False
        maybe_print_tracker_position(tele_data=tele_data)
        return

    trackers_ready = VIVE_MANUS_READER.apply_relative_wrist_motion(tele_data)
    hands_ready = VIVE_MANUS_READER.apply_manus_hand_data(tele_data)
    tele_data.arm_motion_data_ready = trackers_ready
    tele_data.motion_data_ready = trackers_ready or hands_ready
    maybe_print_tracker_position(tele_data=tele_data)

    if not trackers_ready:
        now = time.monotonic()
        if now - getattr(VIVE_MANUS_READER, "_last_tracker_wait_log", 0.0) > 2.0:
            logger_mp.warning("[Vive/Relative] Press [c] in attention pose and keep left/right/head trackers valid.")
            VIVE_MANUS_READER._last_tracker_wait_log = now
    if not hands_ready:
        now = time.monotonic()
        if now - getattr(VIVE_MANUS_READER, "_last_manus_wait_log", 0.0) > 2.0:
            logger_mp.warning("[Manus] Waiting for valid left/right glove data.")
            VIVE_MANUS_READER._last_manus_wait_log = now


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1', 'H2'], default='H1_2', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire_ftp', 'inspire_dfx', 'brainco'], help='Select end effector controller')
    # network parameters
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    # Vive Tracker / Manus ROS2 input parameters
    parser.add_argument('--left-tracker-name', type=str, default='LHR-36B992CB', help='libsurvive TF child frame for the left wrist tracker')
    parser.add_argument('--right-tracker-name', type=str, default='LHR-0FAD1369', help='libsurvive TF child frame for the right wrist tracker')
    parser.add_argument('--head-tracker-name', type=str, default='LHR-1F621773', help='libsurvive TF child frame for the head tracker; empty string disables head tracking')
    parser.add_argument('--libsurvive-tracking-frame', type=str, default='libsurvive_world', help='libsurvive TF parent frame')
    parser.add_argument('--ros-stale-timeout', type=float, default=0.5, help='Seconds before ROS2 tracker/glove data is considered stale')
    parser.add_argument('--manus-topics', type=str, nargs='+', default=['manus_glove_0', 'manus_glove_1'], help='ROS2 Manus glove topics; msg.side or topic name must identify left/right')
    parser.add_argument('--manus-msg-type', type=str, default='manus_ros2_msgs/msg/ManusGlove', help='ROS2 message type for Manus glove data')
    # Head tracker -> simulator camera DDS parameters
    parser.add_argument('--enable-head-camera-dds', action='store_true', help='Publish calibrated head tracker rotation to the sim robot camera DDS topic')
    parser.add_argument('--head-camera-topic', type=str, default='rt/robot_camera/cmd', help='DDS topic for robot camera orientation commands')
    parser.add_argument('--head-camera-name', type=str, default='front_camera', help='Camera name field sent to the simulator')
    parser.add_argument('--head-camera-frame', type=str, default='d435_link', help='Camera frame field sent to the simulator')
    parser.add_argument('--head-camera-mode', type=str, choices=['absolute', 'relative', 'raw'], default='absolute', help='Camera target mode understood by sim_main.py')
    parser.add_argument('--head-camera-rate', type=float, default=30.0, help='Head camera DDS publish rate in Hz')
    parser.add_argument('--head-camera-smoothing', type=float, default=0.7, help='Simulator-side camera smoothing value [0, 0.99]')
    parser.add_argument('--head-camera-correction-rpy-deg', type=float, nargs=3, default=[0.0, 0.0, 0.0], help='Legacy option; head camera maps tracker X rotation to camera pitch and inverted tracker Y rotation to camera yaw')
    parser.add_argument('--head-camera-rpy-scale', type=float, nargs=3, default=[0.0, 1.0, 1.0], help='Per-axis camera RPY scale after tracker-axis mapping; default disables roll and keeps pitch/yaw')
    parser.add_argument('--head-camera-max-rpy-deg', type=float, nargs=3, default=[30.0, 80.0, 120.0], help='Per-axis clamp in degrees; values <= 0 disable that axis clamp')
    parser.add_argument('--head-camera-debug', action='store_true', help='Print periodic head camera DDS command diagnostics')
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--affinity', action = 'store_true', help = 'Enable high priority and set CPU affinity mode')
    # record mode and task info
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording mode')
    parser.add_argument('--task-dir', type = str, default = './utils/data/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'pick cube', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'pick up cube.', help = 'task goal for recording at json file')
    parser.add_argument('--task-desc', type = str, default = 'task description', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: do this; step2: do that;', help = 'task steps for recording at json file')

    args = parser.parse_args()
    logger_mp.debug(f"args: {args}")

    arm_ctrl = None
    ipc_server = None
    listen_keyboard_thread = None
    img_client = None
    tv_wrapper = None
    sim_state_subscriber = None
    recorder = None
    ee_runtime = None
    head_camera_dds = None

    try:
        # setup dds communication domains id
        if args.sim:
            ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        else:
            ChannelFactoryInitialize(0, networkInterface=args.network_interface)

        # ipc communication mode. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press,get_state=get_state)
            ipc_server.start()
        # sshkeyboard communication mode
        else:
            listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                                      kwargs={"on_press": on_press, "until": None, "sequential": False,}, 
                                                      daemon=True)
            listen_keyboard_thread.start()

        # image client
        img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
        camera_config = img_client.get_cam_config()
        logger_mp.debug(f"Camera config: {camera_config}")
        xr_need_local_img = not (args.display_mode == 'pass-through' or camera_config['head_camera']['enable_webrtc'])

        # televuer_wrapper: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
        tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand", 
                                     binocular=camera_config['head_camera']['binocular'],
                                     img_shape=camera_config['head_camera']['image_shape'],
                                     # maybe should decrease fps for better performance?
                                     # https://github.com/unitreerobotics/xr_teleoperate/issues/172
                                     # display_fps=camera_config['head_camera']['fps'] ? args.frequency? 30.0?
                                     display_mode=args.display_mode,
                                     zmq=camera_config['head_camera']['enable_zmq'],
                                     webrtc=camera_config['head_camera']['enable_webrtc'],
                                     webrtc_url=f"https://{args.img_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
                                     arm_reference_mode="head_yaw"
                                     )

        VIVE_MANUS_READER = ViveManusInfoReader(
            left_tracker_name=args.left_tracker_name,
            right_tracker_name=args.right_tracker_name,
            head_tracker_name=args.head_tracker_name,
            libsurvive_tracking_frame=args.libsurvive_tracking_frame,
            manus_topics=args.manus_topics,
            manus_msg_type=args.manus_msg_type,
            stale_timeout=args.ros_stale_timeout,
        )
        
        # motion mode (G1: Regular mode R1+X, not Running mode R2+A)
        if args.motion:
            if args.input_mode == "controller":
                loco_wrapper = LocoClientWrapper()
        else:
            motion_switcher = MotionSwitcher()
            status, result = motion_switcher.Enter_Debug_Mode()
            logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")

        # arm
        arm_ik, arm_ctrl = _init_arm_stack(args)

        # end-effector
        xr_motion_data_ready = Value('b', False, lock=True)        # [input] whether XR hand/controller motion data has arrived
        ee_runtime = _init_end_effector(args, xr_motion_data_ready)
        
        # affinity mode (if you dont know what it is, then you probably don't need it)
        if args.affinity:
            import psutil
            p = psutil.Process(os.getpid())
            p.cpu_affinity([0,1,2,3]) # Set CPU affinity to cores 0-3
            try:
                p.nice(-20)           # Set highest priority
                logger_mp.info("Set high priority successfully.")
            except psutil.AccessDenied:
                logger_mp.warning("Failed to set high priority. Please run as root.")
                
            for child in p.children(recursive=True):
                try:
                    logger_mp.info(f"Child process {child.pid} name: {child.name()}")
                    child.cpu_affinity([5,6])
                    child.nice(-20)
                except psutil.AccessDenied:
                    pass

        # simulation mode
        if args.sim:
            reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
            reset_pose_publisher.Init()
            from teleop.utils.sim_state_topic import start_sim_state_subscribe
            sim_state_subscriber = start_sim_state_subscribe()

        if args.enable_head_camera_dds:
            try:
                head_camera_publisher = ChannelPublisher(args.head_camera_topic, String_)
                head_camera_publisher.Init()
                head_camera_dds = HeadCameraDDSPublisher(
                    publisher=head_camera_publisher,
                    topic=args.head_camera_topic,
                    camera_name=args.head_camera_name,
                    frame=args.head_camera_frame,
                    mode=args.head_camera_mode,
                    rate_hz=args.head_camera_rate,
                    smoothing=args.head_camera_smoothing,
                    correction_rpy_deg=args.head_camera_correction_rpy_deg,
                    rpy_scale=args.head_camera_rpy_scale,
                    max_rpy_deg=args.head_camera_max_rpy_deg,
                    debug=args.head_camera_debug,
                )
                logger_mp.info(f"[HeadCam] DDS publisher initialized on {args.head_camera_topic}")
            except Exception as e:
                logger_mp.error(f"[HeadCam] failed to initialize DDS publisher: {e}")
                head_camera_dds = None

        # record + headless / non-headless mode
        if args.record:
            recorder = EpisodeWriter(task_dir = os.path.join(args.task_dir, args.task_name),
                                     task_goal = args.task_goal,
                                     task_desc = args.task_desc,
                                     task_steps = args.task_steps,
                                     frequency = args.frequency, 
                                     rerun_log = not args.headless)

        logger_mp.info("----------------------------------------------------------------")
        logger_mp.info("🟢  Press [r] to start syncing the robot with your movements.")
        logger_mp.info("🟣  Press [r] while running to print current Vive tracker positions.")
        logger_mp.info("🟠  Press [c] in attention pose to calibrate; press [r] again to sync after calibration.")
        if args.record:
            logger_mp.info("🟡  Press [s] to START or SAVE recording (toggle cycle).")
        else:
            logger_mp.info("🔵  Recording is DISABLED (run with --record to enable).")
        logger_mp.info("🔴  Press [q] to stop and exit the program.")
        logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")
        READY = True                  # now ready to (1) enter START state
        while not START and not STOP: # wait for start or stop signal.
            time.sleep(0.033)
            tele_data = tv_wrapper.get_tele_data()
            maybe_print_tracker_position(tele_data=tele_data)
            maybe_calibrate_tracker_and_head_camera(tele_data=tele_data, head_camera_dds=head_camera_dds)
            if camera_config['head_camera']['enable_zmq'] and xr_need_local_img:
                head_img = img_client.get_head_frame()
                if head_img.bgr is not None:
                    tv_wrapper.render_to_xr(head_img.bgr)

        maybe_print_tracker_position(tele_data=tv_wrapper.get_tele_data())
        logger_mp.info("---------------------🚀start Tracking🚀-------------------------")
        arm_ctrl.speed_gradual_max()

        head_img = None
        left_wrist_img = None
        right_wrist_img = None

        # main loop. robot start to follow VR user's motion
        while not STOP:
            start_time = time.time()
            # get image
            if camera_config['head_camera']['enable_zmq']:
                if args.record or xr_need_local_img:
                    head_img = img_client.get_head_frame()
                if xr_need_local_img and head_img.bgr is not None:
                    tv_wrapper.render_to_xr(head_img.bgr)
            if camera_config['left_wrist_camera']['enable_zmq']:
                if args.record:
                    left_wrist_img = img_client.get_left_wrist_frame()
            if camera_config['right_wrist_camera']['enable_zmq']:
                if args.record:
                    right_wrist_img = img_client.get_right_wrist_frame()

            # record mode
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    if recorder.create_episode():
                        RECORD_RUNNING = True
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    RECORD_RUNNING = False
                    recorder.save_episode()
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)

            # get xr's tele data
            tele_data = tv_wrapper.get_tele_data()
            if not START or CALIBRATE_TRACKER_FRAME:
                maybe_print_tracker_position(tele_data=tele_data)
                maybe_calibrate_tracker_and_head_camera(tele_data=tele_data, head_camera_dds=head_camera_dds)
                tele_data.arm_motion_data_ready = False
                tele_data.motion_data_ready = False
                with xr_motion_data_ready.get_lock():
                    xr_motion_data_ready.value = False
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / args.frequency) - time_elapsed)
                time.sleep(sleep_time)
                logger_mp.debug(f"main process sleep: {sleep_time}")
                continue

            _apply_vive_manus_input(tele_data, head_camera_dds=head_camera_dds)
            if head_camera_dds is not None and START:
                head_camera_dds.maybe_publish(VIVE_MANUS_READER)
            if not START:
                with xr_motion_data_ready.get_lock():
                    xr_motion_data_ready.value = False
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / args.frequency) - time_elapsed)
                time.sleep(sleep_time)
                logger_mp.debug(f"main process sleep: {sleep_time}")
                continue
            _apply_end_effector_inputs(args, tele_data, ee_runtime)
            with xr_motion_data_ready.get_lock():
                xr_motion_data_ready.value = getattr(tele_data, "hand_motion_data_ready", tele_data.motion_data_ready)
            
            # high level control
            if args.input_mode == "controller" and args.motion:
                # quit teleoperate
                if tele_data.right_ctrl_aButton:
                    START = False
                    STOP = True
                # command robot to enter damping mode. soft emergency stop function
                if tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick:
                    loco_wrapper.Damp()
                # https://github.com/unitreerobotics/xr_teleoperate/issues/135, control, limit velocity to within 0.3
                loco_wrapper.Move(*_controller_body_action(tele_data))

            # get current robot state data.
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

            # solve ik using motor data and wrist pose, then use ik results to control arms.
            time_ik_start = time.time()
            sol_q, sol_tauff  = arm_ik.solve_ik(tele_data.left_wrist_pose, tele_data.right_wrist_pose, current_lr_arm_q, current_lr_arm_dq)
            time_ik_end = time.time()
            logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
            maybe_print_ik_rotation_debug(tele_data, arm_ik, current_lr_arm_q, sol_q)
            arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
            maybe_print_hand_debug(args, tele_data, ee_runtime)

            # record data
            if args.record:
                READY = recorder.is_ready() # now ready to (2) enter RECORD_RUNNING state
                (
                    left_ee_state,
                    right_ee_state,
                    left_hand_action,
                    right_hand_action,
                    current_body_state,
                    current_body_action,
                ) = _read_end_effector_record_data(args, tele_data, arm_ctrl, ee_runtime)

                if RECORD_RUNNING:
                    colors = _collect_record_colors(camera_config, head_img, left_wrist_img, right_wrist_img)
                    depths = {}
                    states, actions = _build_record_state_action(
                        current_lr_arm_q,
                        sol_q,
                        left_ee_state,
                        right_ee_state,
                        left_hand_action,
                        right_hand_action,
                        current_body_state,
                        current_body_action,
                    )
                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()            
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, sim_state=sim_state)
                    else:
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("⛔ KeyboardInterrupt, exiting program...")
    except Exception:
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
        try:
            if arm_ctrl is not None:
                arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")
        
        try:
            if args.ipc and ipc_server is not None:
                ipc_server.stop()
            elif listen_keyboard_thread is not None:
                stop_listening()
                listen_keyboard_thread.join()
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")
        
        try:
            if img_client is not None:
                img_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close image client: {e}")

        try:
            if tv_wrapper is not None:
                tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close televuer wrapper: {e}")

        try:
            if VIVE_MANUS_READER is not None:
                VIVE_MANUS_READER.close()
        except Exception as e:
            logger_mp.error(f"Failed to close Vive/Manus reader: {e}")

        try:
            if not args.motion:
                pass
                # status, result = motion_switcher.Exit_Debug_Mode()
                # logger_mp.info(f"Exit debug mode: {'Success' if status == 3104 else 'Failed'}")
        except Exception as e:
            logger_mp.error(f"Failed to exit debug mode: {e}")

        try:
            if args.sim and sim_state_subscriber is not None:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")
        
        try:
            if args.record and recorder is not None:
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")
        logger_mp.info("✅ Finally, exiting program.")
        exit(0)
