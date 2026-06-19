import numpy as np
import time
import argparse
import cv2
import json
import math
import pinocchio as pin
from multiprocessing import shared_memory, Value, Array, Lock
import threading
import importlib
from enum import Enum
import logging_mp
logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# XR wrapper & robot controllers
from televuer.tv_wrapper import (
    CONST_HEAD_POSE,
    T_OPENXR_ROBOT,
    T_ROBOT_OPENXR,
    T_TO_UNITREE_HUMANOID_LEFT_ARM,
    T_TO_UNITREE_HUMANOID_RIGHT_ARM,
    TeleData,
    TeleStateData,
    fast_mat_inv,
    safe_mat_update,
)
# libsurvive world frame: X=right, Y=forward, Z=up  →  robot: X=forward, Y=left, Z=up
_T_LIBSURVIVE_TO_ROBOT = np.array([[ 0, 1, 0, 0],
                                    [-1, 0, 0, 0],
                                    [ 0, 0, 1, 0],
                                    [ 0, 0, 0, 1]], dtype=float)
_T_ROBOT_TO_LIBSURVIVE = _T_LIBSURVIVE_TO_ROBOT[:3, :3].T  # orthogonal inverse
_T_ROBOT_TO_LIBSURVIVE = np.block([[_T_ROBOT_TO_LIBSURVIVE, np.zeros((3,1))],
                                    [np.zeros((1,3)),        np.ones((1,1))]])


# Wrist correction: tracker device frame  →  Unitree arm IK EE frame
# Tracker: X=pinky→thumb, Y=distal(finger ext), Z=palm outward
# Tune this if the rotation direction is wrong; T_TO_UNITREE_HUMANOID is OpenXR-based
# Current: same as T_TO_UNITREE (Rx +90° for left, Rx -90° for right)
# If axes are wrong, adjust here (see diagnostics print below)
# EE_X = tracker_Y (distal)      → T[:,0] = [0,1,0]
# EE_X = tracker_Y (distal/pitch axis), EE_Y = tracker_Z (palm/yaw axis), EE_Z = tracker_X (pinky→thumb/roll axis)
_T_WRIST_CORR_LEFT  = np.array([[0, 0, 1],
                                  [1, 0, 0],
                                  [0, 1, 0]], dtype=float)
_T_WRIST_CORR_RIGHT = np.array([[0, 0, 1],
                                  [1, 0, 0],
                                  [0, 1, 0]], dtype=float)
_LEFT_X_FORWARD_FIX = np.array([[-1, 0, 0],
                                [ 0,-1, 0],
                                [ 0, 0, 1]], dtype=float)

from teleop.robot_control.robot_arm import G1_29_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK 
from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller, Dex1_1_Gripper_Controller
from teleop.robot_control.robot_hand_inspire import Inspire_Controller
from teleop.robot_control.robot_hand_brainco import Brainco_Controller
from teleop.image_server.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter

# terminal-only key input
from sshkeyboard import listen_keyboard, stop_listening

# Inspire SDK (if used)
#from inspire_sdkpy import inspire_dds, inspire_hand_defaut  # noqa: F401

# for simulation reset signal
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

def publish_reset_category(category: int, publisher):
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")


def _normalize_ros_msg_type(msg_type: str) -> str:
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


def _load_ros_msg_type(msg_type: str):
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
    ])


def _pose_to_mat(position, orientation):
    rot = _quat_to_rot(orientation.x, orientation.y, orientation.z, orientation.w)
    if rot is None:
        return None
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = [position.x, position.y, position.z]
    return mat


def _orthonormalize_rot(rot):
    try:
        u, _, vh = np.linalg.svd(np.asarray(rot, dtype=float))
        result = u @ vh
        if np.linalg.det(result) < 0.0:
            u[:, -1] *= -1.0
            result = u @ vh
        return result
    except Exception:
        return np.eye(3)


def _rpy_to_rot(rpy):
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                    cp * cr],
    ], dtype=float)


def _rot_to_rpy(rot):
    rot = _orthonormalize_rot(rot)
    pitch = math.asin(max(-1.0, min(1.0, -float(rot[2, 0]))))
    cp = math.cos(pitch)
    if abs(cp) > 1.0e-6:
        roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
        yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(-float(rot[0, 1]), float(rot[1, 1]))
    return np.array([roll, pitch, yaw], dtype=float)


def _rotation_error_deg(rot_a, rot_b):
    rel = _orthonormalize_rot(rot_a).T @ _orthonormalize_rot(rot_b)
    cos_angle = (float(np.trace(rel)) - 1.0) * 0.5
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def _pose_debug_text(label, pose):
    if pose is None:
        return f"{label}: None"
    pos = np.asarray(pose[:3, 3], dtype=float)
    rot = _orthonormalize_rot(pose[:3, :3])
    rpy_deg = np.rad2deg(_rot_to_rpy(rot))
    return (
        f"{label}: pos={np.round(pos, 4).tolist()} "
        f"rpy_deg={np.round(rpy_deg, 2).tolist()} "
        f"rot=\n{np.array2string(rot, precision=4, suppress_small=True)}"
    )


def _se3_to_mat(se3):
    mat = np.eye(4)
    mat[:3, :3] = np.asarray(se3.rotation, dtype=float)
    mat[:3, 3] = np.asarray(se3.translation, dtype=float).reshape(3)
    return mat


def _current_sim_ee_poses(arm_ik, current_lr_arm_q):
    model = arm_ik.reduced_robot.model
    q = np.asarray(current_lr_arm_q, dtype=float).reshape(model.nq)
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return _se3_to_mat(data.oMf[arm_ik.L_hand_id]), _se3_to_mat(data.oMf[arm_ik.R_hand_id])


def _tracker_world_and_sync_pose(tv_wrapper, tracker_pose, side):
    robot_world_pose = tv_wrapper._pose_to_robot_world(tracker_pose)
    if robot_world_pose is None or not np.all(np.isfinite(robot_world_pose)):
        return None, None

    correction = np.eye(4)
    if tv_wrapper.apply_unitree_arm_orientation:
        correction[:3, :3] = _T_WRIST_CORR_LEFT if side == "left" else _T_WRIST_CORR_RIGHT
    sync_pose = robot_world_pose @ correction
    if side == "left":
        sync_pose = sync_pose.copy()
        sync_pose[:3, :3] = sync_pose[:3, :3] @ _LEFT_X_FORWARD_FIX
    return robot_world_pose, sync_pose


def _check_tracker_sim_sync(tv_wrapper, arm_ik, arm_ctrl, max_rotation_error_deg):
    try:
        current_lr_arm_q = arm_ctrl.get_current_dual_arm_q()
        left_sim_pose, right_sim_pose = _current_sim_ee_poses(arm_ik, current_lr_arm_q)
    except Exception as e:
        print(f"[SYNC ALARM] Cannot read current sim EE pose: {e}", flush=True)
        logger_mp.error(f"[SYNC ALARM] Cannot read current sim EE pose: {e}")
        return False, None, None, None, None

    left_tracker, right_tracker, left_ok, right_ok = tv_wrapper.vive_reader.read()
    tracker_world = {"left": None, "right": None}
    tracker_sync = {"left": None, "right": None}
    sim_pose = {"left": left_sim_pose, "right": right_sim_pose}
    valid = {"left": left_ok and left_tracker is not None, "right": right_ok and right_tracker is not None}

    for side, raw_pose in [("left", left_tracker), ("right", right_tracker)]:
        if not valid[side]:
            continue
        tracker_world[side], tracker_sync[side] = _tracker_world_and_sync_pose(tv_wrapper, raw_pose, side)
        valid[side] = tracker_world[side] is not None and tracker_sync[side] is not None

    lines = [
        "[SYNC] tracker/sim rotation check",
        f"threshold: rotation <= {max_rotation_error_deg:.1f} deg (position is printed only)",
    ]
    failed = []
    for side in ("left", "right"):
        if not valid[side]:
            failed.append(f"{side}: tracker invalid")
            lines.append(f"{side}: tracker invalid")
            continue

        pos_error = float(np.linalg.norm(tracker_sync[side][:3, 3] - sim_pose[side][:3, 3]))
        rot_error = _rotation_error_deg(tracker_sync[side][:3, :3], sim_pose[side][:3, :3])
        lines.append(f"{side}: pos_delta={pos_error:.4f} m (not checked), rot_error={rot_error:.2f} deg")
        lines.append(_pose_debug_text(f"{side} tracker", tracker_sync[side]))
        lines.append(_pose_debug_text(f"{side} sim", sim_pose[side]))

        if rot_error > max_rotation_error_deg:
            failed.append(f"{side}: rot_error={rot_error:.2f} deg")

    if failed:
        lines.insert(0, "[SYNC ALARM] Sync rejected: tracker and sim rotation are too different.")
        lines.append("failed checks: " + "; ".join(failed))
        report = "\n".join(lines)
        print(report, flush=True)
        logger_mp.error(report)
        return False, None, None, None, None

    report = "\n".join(lines)
    print(report, flush=True)
    logger_mp.info("[SYNC] Sync rotation check passed.")
    return True, left_sim_pose, right_sim_pose, tracker_world["left"], tracker_world["right"]


def _rot_to_quat_wxyz(rot):
    rot = _orthonormalize_rot(rot)
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = [0.25 * s, (rot[2, 1] - rot[1, 2]) / s, (rot[0, 2] - rot[2, 0]) / s, (rot[1, 0] - rot[0, 1]) / s]
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        quat = [(rot[2, 1] - rot[1, 2]) / s, 0.25 * s, (rot[0, 1] + rot[1, 0]) / s, (rot[0, 2] + rot[2, 0]) / s]
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        quat = [(rot[0, 2] - rot[2, 0]) / s, (rot[0, 1] + rot[1, 0]) / s, 0.25 * s, (rot[1, 2] + rot[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        quat = [(rot[1, 0] - rot[0, 1]) / s, (rot[0, 2] + rot[2, 0]) / s, (rot[1, 2] + rot[2, 1]) / s, 0.25 * s]

    quat = np.asarray(quat, dtype=float)
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm < 1.0e-8:
        return [1.0, 0.0, 0.0, 0.0]
    return (quat / norm).astype(float).tolist()


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


def _extract_manus_glove_side(msg):
    """ManusGlove 메시지에서 'left'/'right' 문자열 반환. 없으면 None."""
    side = getattr(msg, "side", None)
    if isinstance(side, str) and side.lower() in ("left", "right"):
        return side.lower()
    return None


def _extract_manus_glove_positions(msg):
    """ManusGlove.raw_nodes를 node_id 순서로 정렬해 positions (25,3)과 wrist 4x4 pose를 반환."""
    raw_nodes = getattr(msg, "raw_nodes", None)
    if not raw_nodes:
        return None, None

    arr = np.zeros((25, 3), dtype=float)
    wrist_mat = None
    for node in raw_nodes:
        node_id = getattr(node, "node_id", None)
        if node_id is None or not (0 <= node_id < 25):
            continue
        pose = getattr(node, "pose", None)
        if pose is None:
            continue
        pos = getattr(pose, "position", None)
        if pos is None:
            continue
        xyz = _point_like_to_xyz(pos)
        if xyz is not None:
            arr[node_id] = xyz
        if node_id == 0 and hasattr(pose, "orientation"):
            wrist_mat = _pose_to_mat(pose.position, pose.orientation)

    return arr, wrist_mat


class LibsurviveTFReader:
    """Vive Tracker pose via TF published by libsurvive_ros2.
    Looks up transforms: tracking_frame -> left/right/head tracker names.
    """
    def __init__(self, node, left_name, right_name, head_name=None, tracking_frame="libsurvive_frame", stale_timeout=0.5):
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
            # TF gives the latest tracker pose as translation + quaternion.
            # Convert it to a 4x4 matrix and cache it so brief TF drops do not
            # immediately zero the target pose.
            from rclpy.time import Time as RclpyTime
            t = self._buffer.lookup_transform(self._tracking_frame, name, RclpyTime())
            pose = _pose_to_mat(t.transform.translation, t.transform.rotation)
            if pose is not None:
                self._last_ok[side] = time.monotonic()
                self._last_poses[side] = pose
        except Exception as e:
            now2 = time.monotonic()
            if now2 - getattr(self, f"_last_err_{side}", 0.0) > 3.0:
                logger_mp.warning(f"[Vive/TF] {side} lookup failed: {e}")
                setattr(self, f"_last_err_{side}", now2)
        now = time.monotonic()
        ok = (now - self._last_ok[side]) <= self._stale_timeout
        return self._last_poses[side], ok

    def read(self):
        left_pose, left_ok   = self._lookup_one("left")
        right_pose, right_ok = self._lookup_one("right")
        return left_pose, right_pose, left_ok, right_ok

    def read_head(self):
        if "head" not in self._names:
            return None, False

        head_pose, head_ok = self._lookup_one("head")
        return head_pose, head_ok


class ManusHandReader:
    def __init__(self, node, topics, msg_type, stale_timeout):
        self._lock = threading.Lock()
        self._positions = {"left": None, "right": None}
        self._wrist_mats = {"left": None, "right": None}
        self._stamp = {"left": 0.0, "right": 0.0}
        self._last_warn = {"left": 0.0, "right": 0.0}
        self._stale_timeout = stale_timeout
        ros_msg_type = _load_ros_msg_type(msg_type)
        self._subs = [
            node.create_subscription(ros_msg_type, topic, self._callback, 5)
            for topic in topics
        ]
        logger_mp.info(f"ManusHandReader subscribed: topics={topics}, type={msg_type}")

    def _callback(self, msg):
        # ManusGlove publishes both hands on configurable topics. The side is
        # read from msg.side, and raw_nodes[0] is used as the wrist frame.
        side = _extract_manus_glove_side(msg)
        if side is None:
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


class ViveManusTeleopWrapper:
    def __init__(
        self,
        left_tracker_name,
        right_tracker_name,
        head_tracker_name,
        manus_topics,
        manus_msg_type,
        libsurvive_tracking_frame="libsurvive_frame",
        vive_input_frame="openxr",
        stale_timeout=0.5,
        position_scale=1.0,
        apply_unitree_arm_orientation=True,
    ):
        import rclpy
        from rclpy.executors import MultiThreadedExecutor

        self._rclpy = rclpy
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init(args=None)

        self.node = rclpy.create_node("vive_manus_teleop_wrapper")
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
        self.manus_reader = ManusHandReader(
            self.node, manus_topics, manus_msg_type, stale_timeout
        )

        self.vive_input_frame = vive_input_frame
        self.position_scale = float(position_scale)
        self.apply_unitree_arm_orientation = apply_unitree_arm_orientation
        self.head_pose = T_ROBOT_OPENXR @ CONST_HEAD_POSE @ T_OPENXR_ROBOT
        self._head_calib_tracker = None
        self._head_calibrated = False
        self._head_last_warn = 0.0
        self.prev_left_arm_pose = self._default_arm_pose("left")
        self.prev_right_arm_pose = self._default_arm_pose("right")

        # Startup calibration: maps tracker world pose to robot hand pose at calibration time.
        # Runtime targets are generated from the tracker's relative SE(3) change:
        #     T_delta = inv(T_tracker_start) @ T_tracker_now
        #     T_robot = T_robot_start @ T_delta
        self._calib_tracker = {"left": None, "right": None}
        self._calib_robot   = {"left": None, "right": None}
        self._calibrated    = {"left": False, "right": False}

    def reconnect(self):
        logger_mp.info("ViveManusTeleopWrapper uses ROS2 subscriptions; reconnect is a no-op.")

    def shutdown(self):
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

    def _pose_to_robot_world(self, pose):
        """Convert Vive/libsurvive tracker pose into the robot-world convention."""
        if self.vive_input_frame == "openxr":
            return T_ROBOT_OPENXR @ pose @ T_OPENXR_ROBOT
        if self.vive_input_frame == "libsurvive":
            # tracker: X=right, Y=forward, Z=up  →  robot: X=forward, Y=left, Z=up
            # Robot_X = Tracker_Y, Robot_Y = -Tracker_X, Robot_Z = Tracker_Z
            return _T_LIBSURVIVE_TO_ROBOT @ pose @ _T_ROBOT_TO_LIBSURVIVE
        return pose.copy()

    def _default_arm_pose(self, side):
        pose = np.eye(4)
        pose[:3, 3] = [0.25, 0.15 if side == "left" else -0.15, 0.08]
        if side == "left":
            pose[:3, :3] = T_TO_UNITREE_HUMANOID_LEFT_ARM[:3, :3]
        else:
            pose[:3, :3] = T_TO_UNITREE_HUMANOID_RIGHT_ARM[:3, :3]
        return pose

    def calibrate(self, left_robot_pose=None, right_robot_pose=None):
        """Capture current tracker world poses as calibration reference.
        After calibration, _arm_pose_for_ik outputs arm_pose via relative transforms:
            T_rel = inv(T_tracker_start) @ T_tracker_now
            T_arm = T_robot_hand_start @ T_rel
        """
        left_tracker, right_tracker, left_ok, right_ok = self.vive_reader.read()
        for side, tracker_pose, ok, robot_pose in [
            ("left",  left_tracker,  left_ok,  left_robot_pose),
            ("right", right_tracker, right_ok, right_robot_pose),
        ]:
            if not ok or tracker_pose is None:
                logger_mp.warning(f"[Calibrate] {side} tracker not valid, skipping")
                continue
            robot_world = self._pose_to_robot_world(tracker_pose)
            self._calib_tracker[side] = robot_world.copy()
            self._calib_robot[side]   = robot_pose.copy() if robot_pose is not None else self._default_arm_pose(side)
            self._calibrated[side]    = True
            logger_mp.info(
                f"[Calibrate] {side} done — tracker_world={robot_world[:3,3].round(3)}, "
                f"robot_ref={self._calib_robot[side][:3,3].round(3)}"
            )

    def calibrate_from_robot_world(self, left_tracker_world, right_tracker_world, left_robot_pose, right_robot_pose):
        """Calibrate from tracker poses already converted to robot-world coordinates."""
        all_ok = True
        for side, tracker_world, robot_pose in [
            ("left", left_tracker_world, left_robot_pose),
            ("right", right_tracker_world, right_robot_pose),
        ]:
            if tracker_world is None or robot_pose is None:
                logger_mp.warning(f"[SyncCalib] {side} pose missing, calibration skipped")
                self._calibrated[side] = False
                all_ok = False
                continue
            if not np.all(np.isfinite(tracker_world)) or not np.all(np.isfinite(robot_pose)):
                logger_mp.warning(f"[SyncCalib] {side} pose has non-finite values, calibration skipped")
                self._calibrated[side] = False
                all_ok = False
                continue

            self._calib_tracker[side] = tracker_world.copy()
            self._calib_robot[side] = robot_pose.copy()
            self._calibrated[side] = True
            logger_mp.info(
                f"[SyncCalib] {side} done - tracker_world={tracker_world[:3,3].round(3)}, "
                f"sim_ref={robot_pose[:3,3].round(3)}"
            )
        return all_ok

    def read_head_tracker(self):
        head_tracker, head_ok = self.vive_reader.read_head()
        if not head_ok or head_tracker is None:
            return None, False

        robot_world = self._pose_to_robot_world(head_tracker)
        if robot_world is None or not np.all(np.isfinite(robot_world)):
            return None, False

        self.head_pose = robot_world.copy()
        return robot_world, True

    def calibrate_head(self):
        head_pose, head_ok = self.read_head_tracker()
        if not head_ok or head_pose is None:
            now = time.monotonic()
            if now - self._head_last_warn >= 2.0:
                logger_mp.warning("[HeadCam] head tracker not valid, calibration skipped")
                self._head_last_warn = now
            return False

        self._head_calib_tracker = head_pose.copy()
        self._head_calibrated = True
        logger_mp.info(
            f"[HeadCam] calibrated head neutral pos={head_pose[:3,3].round(3)}"
        )
        return True

    def head_camera_delta_rot(self):
        if not self._head_calibrated or self._head_calib_tracker is None:
            if not self.calibrate_head():
                return None, False

        head_pose, head_ok = self.read_head_tracker()
        if not head_ok or head_pose is None:
            return None, False

        start_rot = _orthonormalize_rot(self._head_calib_tracker[:3, :3])
        now_rot = _orthonormalize_rot(head_pose[:3, :3])
        return start_rot.T @ now_rot, True

    def _arm_pose_for_ik(self, tracker_pose, side, valid):
        """Map current tracker pose to a robot EE target pose for IK."""
        fallback = self.prev_left_arm_pose if side == "left" else self.prev_right_arm_pose
        if not valid or tracker_pose is None:
            return fallback.copy()

        tracker_pose, valid = safe_mat_update(fallback, tracker_pose)
        if not valid:
            return fallback.copy()

        robot_world_pose = self._pose_to_robot_world(tracker_pose)

        if not self._calibrated[side]:
            # Fallback: should not normally happen since calibrate() is called at ACTIVE entry
            return self._default_arm_pose(side)

        calib_robot = self._calib_robot[side]
        correction = np.eye(4)
        if self.apply_unitree_arm_orientation:
            correction[:3, :3] = _T_WRIST_CORR_LEFT if side == "left" else _T_WRIST_CORR_RIGHT

        # Convert tracker-device axes into the robot hand/EE axes before taking
        # the relative transform, so the initial tracker orientation becomes zero.
        tracker_start = self._calib_tracker[side] @ correction
        tracker_now = robot_world_pose @ correction
        relative_tracker = fast_mat_inv(tracker_start) @ tracker_now

        arm_pose = calib_robot @ relative_tracker
        if side == "left":
            left_relative_rot = _LEFT_X_FORWARD_FIX @ relative_tracker[:3, :3] @ _LEFT_X_FORWARD_FIX
            arm_pose[:3, :3] = calib_robot[:3, :3] @ left_relative_rot
        # Position: world-frame delta applied directly (bypasses calib_robot_R axis mixing)
        delta_world = robot_world_pose[:3, 3] - self._calib_tracker[side][:3, 3]
        arm_pose[:3, 3] = calib_robot[:3, 3] + delta_world * self.position_scale

        if side == "left":
            self.prev_left_arm_pose = arm_pose.copy()
        else:
            self.prev_right_arm_pose = arm_pose.copy()
        return arm_pose

    def _hand_pos_for_retargeting(self, hand_positions, wrist_mat, valid):
        """Convert Manus world-space raw nodes into wrist-local URDF points."""
        if not valid or hand_positions is None:
            return np.zeros((25, 3))
        if wrist_mat is not None:
            arm = fast_mat_inv(wrist_mat)
        else:
            wrist = hand_positions[0].copy()
            arm = np.eye(4)
            arm[:3, 3] = -wrist
        # Manus wrist-local: +Z = 손가락 펼침(extension), +X = radial(thumb방향), +Y = dorsal
        # Inspire URDF L_hand_base_link: -Y = 손가락 펼침, +Z = radial(index방향), +X = dorsal
        # 매핑: Manus_Y→URDF_X(dorsal), Manus_Z→URDF_-Y(extension), Manus_X→URDF_Z(radial)
        T_MANUS_TO_URDF = np.array([[0,  1,  0, 0],
                                    [0,  0, -1, 0],
                                    [1,  0,  0, 0],
                                    [0,  0,  0, 1]])
        hom = np.concatenate([hand_positions.T, np.ones((1, hand_positions.shape[0]))])
        local = arm @ hom
        return (T_MANUS_TO_URDF @ local)[0:3, :].T

    def get_tele_data(self):
        """Read Vive + Manus inputs and expose them in the TeleData format."""
        left_tracker, right_tracker, left_tracker_ok, right_tracker_ok = self.vive_reader.read()
        left_hand_raw, right_hand_raw, left_wrist_mat, right_wrist_mat, left_hand_ok, right_hand_ok = self.manus_reader.read()

        # Tracker poses drive arm IK targets; Manus joint positions drive hand retargeting.
        left_arm_pose = self._arm_pose_for_ik(left_tracker, "left", left_tracker_ok)
        right_arm_pose = self._arm_pose_for_ik(right_tracker, "right", right_tracker_ok)
        left_hand_pos = self._hand_pos_for_retargeting(left_hand_raw, left_wrist_mat, left_hand_ok)
        right_hand_pos = self._hand_pos_for_retargeting(right_hand_raw, right_wrist_mat, right_hand_ok)

        tracker_active = left_tracker_ok and right_tracker_ok
        hands_active = left_hand_ok and right_hand_ok
        # Manus만 있어도 ACTIVE 진입 (핸드 리타게팅만 동작, arm IK는 tracker 있을 때만)
        tracking_active = hands_active or tracker_active
        session_alive = tracker_active or hands_active
        self._tracker_active = tracker_active

        left_pinch = float(np.linalg.norm(left_hand_pos[4] - left_hand_pos[9]) * 100.0) if left_hand_ok else 0.0
        right_pinch = float(np.linalg.norm(right_hand_pos[4] - right_hand_pos[9]) * 100.0) if right_hand_ok else 0.0
        tele_state = TeleStateData(
            left_pinch_state=left_pinch < 2.5,
            right_pinch_state=right_pinch < 2.5,
        )

        return TeleData(
            head_pose=self.head_pose.copy(),
            left_arm_pose=left_arm_pose,
            right_arm_pose=right_arm_pose,
            left_hand_pos=left_hand_pos,
            right_hand_pos=right_hand_pos,
            left_hand_rot=None,
            right_hand_rot=None,
            left_pinch_value=left_pinch,
            right_pinch_value=right_pinch,
            tele_state=tele_state,
            tracking_active=tracking_active,
            session_alive=session_alive,
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
        self.correction_rot = _rpy_to_rot(np.deg2rad(np.asarray(correction_rpy_deg, dtype=float)))
        self.rpy_scale = np.asarray(rpy_scale, dtype=float)
        self.max_rpy = np.deg2rad(np.asarray(max_rpy_deg, dtype=float))
        self.debug = debug
        self.seq = 0
        self.last_publish = 0.0
        self.last_warn = 0.0
        self.last_debug = 0.0

    def recalibrate(self, tv_wrapper):
        self.last_publish = 0.0
        return tv_wrapper.calibrate_head()

    def _camera_rot_from_head_delta(self, head_delta_rot):
        camera_delta = self.correction_rot @ head_delta_rot @ self.correction_rot.T
        rpy = _rot_to_rpy(camera_delta) * self.rpy_scale
        for idx, max_abs in enumerate(self.max_rpy):
            if max_abs > 0.0:
                rpy[idx] = max(-max_abs, min(max_abs, rpy[idx]))
        return _rpy_to_rot(rpy), rpy

    def maybe_publish(self, tv_wrapper, now=None):
        now = time.monotonic() if now is None else now
        if self.period > 0.0 and (now - self.last_publish) < self.period:
            return False

        head_delta_rot, ok = tv_wrapper.head_camera_delta_rot()
        if not ok or head_delta_rot is None:
            if now - self.last_warn >= 2.0:
                logger_mp.warning("[HeadCam] skipping camera command; head tracker is not ready")
                self.last_warn = now
            return False

        try:
            camera_rot, rpy = self._camera_rot_from_head_delta(head_delta_rot)
            quat_wxyz = _rot_to_quat_wxyz(camera_rot)
            payload = {
                "camera": self.camera_name,
                "mode": self.mode,
                "frame": self.frame,
                "quat_wxyz": quat_wxyz,
                "smoothing": self.smoothing,
                "seq": self.seq,
                "timestamp": time.time(),
            }
            self.publisher.Write(String_(data=json.dumps(payload, separators=(",", ":"))))
            self.seq += 1
            self.last_publish = now
            return True
        except Exception as e:
            if now - self.last_warn >= 2.0:
                logger_mp.warning(f"[HeadCam] failed to publish camera command: {e}")
                self.last_warn = now
            return False

# =========================
# Global state flags
# =========================
#start_signal = False

def on_press(key):
    """Terminal-only key handling.
    r: validate tracker/sim sync and start program
    q: quit program
    s: toggle recording (if --record)
    a: (optional) sim scene reset
    c: recalibrate wrist trackers and head-camera neutral
    """
    global running, should_toggle_recording, should_reset_scene, should_recalibrate, should_start_sync, sync_wait_until
    if key == 'r':
        should_start_sync = True
        sync_wait_until = time.time() + 3.0
        logger_mp.info("Tracker/sim sync requested (key 'r'). Waiting 3 seconds before checking.")
    elif key == 'q':
        stop_listening()
        running = False
    elif key == 's':
        should_toggle_recording = True
    elif key == 'a':
        should_reset_scene = True
    elif key == 'c':
        should_recalibrate = True
        logger_mp.info("Recalibration requested (key 'c').")
    else:
        logger_mp.info(f"{key} pressed (no action mapped).")


# spawn terminal keyboard thread (no OpenCV key handling)
listen_keyboard_thread = threading.Thread(
    target=listen_keyboard,
    kwargs={"on_press": on_press, "until": None, "sequential": False},
    daemon=True,
)
listen_keyboard_thread.start()


# =========================
# Simple FSM for robust re-entry
# =========================
class Mode(Enum):
    PREVIEW = 0
    ACTIVE = 1
    STANDBY = 2


fsm = Mode.STANDBY
lost_since = None
found_since = None
is_homed = False

# tuneable timeouts (seconds)
LOST_TIMEOUT = 0.5   # how long tracking must be missing before we commit to STANDBY
FOUND_CONFIRM = 0.5  # how long tracking must be present before we return to ACTIVE

last_good_left_pose = None
last_good_right_pose = None
pose_filter_enabled = True
is_recording = False
episode_started = False

running = True
should_toggle_recording = False
should_reset_scene = False    # terminal 'a'
should_recalibrate = False    # terminal 'c'
should_start_sync = False     # terminal 'r'
sync_wait_until = None
set_in_standby = False  # STANDBY에서 한 번만 리셋하기 위한 플래그

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_dir', type=str, default='./utils/data', help='path to save data')
    parser.add_argument('--frequency', type=float, default=60.0, help="main loop frequency (Hz)")

    # basic control parameters
    parser.add_argument('--xr-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device tracking source')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1'], default='G1_29', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['none', 'dex1', 'dex3', 'inspire1', 'brainco'], default='brainco', help='Select end effector controller')

    # Vive Tracker (libsurvive TF)
    parser.add_argument('--left-tracker-name',  type=str, default='LHR-2F1F34FC', help='libsurvive 왼손 트래커 시리얼 (예: LHR-ABCDEF12)')
    parser.add_argument('--right-tracker-name', type=str, default='LHR-6711118F', help='libsurvive 오른손 트래커 시리얼')
    parser.add_argument('--head-tracker-name', type=str, default='LHR-501D76A5', help='libsurvive 머리 트래커 시리얼')
    parser.add_argument('--enable-head-camera-dds', action='store_true',
                        help='Publish calibrated head tracker rotation to the sim robot camera DDS topic')
    parser.add_argument('--head-camera-topic', type=str, default='rt/robot_camera/cmd',
                        help='DDS topic for robot camera orientation commands')
    parser.add_argument('--head-camera-name', type=str, default='robot_camera',
                        help='Camera name field sent to the simulator')
    parser.add_argument('--head-camera-frame', type=str, default='d435_link',
                        help='Camera frame field sent to the simulator')
    parser.add_argument('--head-camera-mode', type=str, choices=['absolute', 'relative', 'raw'], default='absolute',
                        help='Camera target mode understood by sim_main.py')
    parser.add_argument('--head-camera-rate', type=float, default=30.0,
                        help='Head camera DDS publish rate in Hz')
    parser.add_argument('--head-camera-smoothing', type=float, default=0.7,
                        help='Simulator-side camera smoothing value [0, 0.99]')
    parser.add_argument('--head-camera-correction-rpy-deg', type=float, nargs=3, default=[0.0, -90.0, 0.0],
                        help='Roll pitch yaw correction in degrees for tracker-to-camera axis alignment')
    parser.add_argument('--head-camera-rpy-scale', type=float, nargs=3, default=[0.0, 1.0, 1.0],
                        help='Roll pitch yaw scale applied after neutral calibration; default disables roll')
    parser.add_argument('--head-camera-max-rpy-deg', type=float, nargs=3, default=[30.0, 80.0, 120.0],
                        help='Absolute roll pitch yaw clamp in degrees; use <=0 per axis to disable')
    parser.add_argument('--head-camera-debug', action='store_true',
                        help='Print periodic head camera DDS command diagnostics')

    parser.add_argument('--libsurvive-tracking-frame', type=str, default='libsurvive_world', help='libsurvive TF 기준 프레임')
    parser.add_argument('--manus-topics', type=str, nargs='+', default=['manus_glove_0', 'manus_glove_1'],
                        help='ROS2 topics for Manus gloves (side는 msg.side 필드로 자동 판별)')
    parser.add_argument('--manus-msg-type', type=str, default='manus_ros2_msgs/msg/ManusGlove', help='ROS2 message type for Manus glove')
    parser.add_argument('--vive-input-frame', type=str, choices=['openxr', 'libsurvive', 'robot', 'waist'], default='libsurvive',
                        help='Frame convention of Vive pose: libsurvive(X=right,Y=fwd,Z=up), openxr, robot world, or waist')
    parser.add_argument('--ros-stale-timeout', type=float, default=0.5, help='Seconds before a ROS2 input topic is considered stale')
    parser.add_argument('--vive-position-scale', type=float, default=1.0, help='Scale applied to Vive tracker translations')
    parser.add_argument('--no-unitree-arm-orientation-fix', action='store_true',
                        help='Do not apply OpenXR-to-Unitree wrist orientation correction after Vive pose conversion')

    # mode flags
    parser.add_argument('--record', action='store_true', help='Enable data recording')
    parser.add_argument('--motion', action='store_true', help='Enable motion control mode')
    parser.add_argument('--headless', action='store_true', help='Disable OpenCV preview')
    parser.add_argument('--sim', action='store_true', help='Enable Isaac simulation mode')
    parser.add_argument('--sync-max-position-error', type=float, default=0.25,
                        help='Deprecated: tracker-vs-sim position is printed during sync but no longer used to reject sync')
    parser.add_argument('--sync-max-rotation-error-deg', type=float, default=45.0,
                        help='Max tracker-vs-sim EE rotation error in degrees allowed when pressing r')

    args = parser.parse_args()
    logger_mp.info(f"args: {args}")

    # =========================
    # Camera config & Shared Memory for images
    # =========================
    if args.sim:
        img_config = {
            'fps': 30,
            'head_camera_type': 'opencv',
            'head_camera_image_shape': [480, 640],
            'head_camera_id_numbers': [0],
            'wrist_camera_type': 'opencv',
            'wrist_camera_image_shape': [480, 640],
            'wrist_camera_id_numbers': [2, 4],
        }
    else:
        # img_config = {
        #     'fps': 30,
        #     'head_camera_type': 'zed',
        #     'head_camera_image_shape': [376, 1344],
        #     # 'head_camera_image_shape': [480, 1280],
        #     'head_camera_id_numbers': [0],
        #     'wrist_camera_type': 'opencv',
        #     'wrist_camera_image_shape': [480, 640],
        #     'wrist_camera_id_numbers': [2, 4],
        # }

        img_config = {
            'fps': 30,
            'head_camera_type': 'opencv',
            'head_camera_image_shape': [480, 640],  # Head camera resolution
            'head_camera_id_numbers': [0, 1],
            'wrist_camera_type': 'opencv',
            'wrist_camera_image_shape': [480, 640],  # Wrist camera resolution
            'wrist_camera_id_numbers': [2, 4],
        }

        

    ASPECT_RATIO_THRESHOLD = 2.0
    BINOCULAR = len(img_config['head_camera_id_numbers']) > 1 or (
        img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD
    )
    WRIST = 'wrist_camera_type' in img_config

    if BINOCULAR and not (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1] * 2, 3)
    else:
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)

    tv_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize)
    tv_img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=tv_img_shm.buf)

    if WRIST:
        wrist_img_shape = (img_config['wrist_camera_image_shape'][0], img_config['wrist_camera_image_shape'][1] * 2, 3)
        wrist_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype=np.uint8, buffer=wrist_img_shm.buf)

    # ImageClient: server address selection
    if WRIST and args.sim:
        img_client = ImageClient(tv_img_shape=tv_img_shape, tv_img_shm_name=tv_img_shm.name,
                                 wrist_img_shape=wrist_img_shape, wrist_img_shm_name=wrist_img_shm.name, server_address="127.0.0.1")
    elif WRIST and not args.sim:
        img_client = ImageClient(tv_img_shape=tv_img_shape, tv_img_shm_name=tv_img_shm.name,
                                 wrist_img_shape=wrist_img_shape, wrist_img_shm_name=wrist_img_shm.name)
    else:
        img_client = ImageClient(tv_img_shape=tv_img_shape, tv_img_shm_name=tv_img_shm.name, server_address='192.168.123.164')

    image_receive_thread = threading.Thread(target=img_client.receive_process, daemon=True)
    image_receive_thread.start()

    # =========================
    # Vive Tracker / Manus ROS2 wrapper
    # =========================
    tv_wrapper = ViveManusTeleopWrapper(
        left_tracker_name=args.left_tracker_name,
        right_tracker_name=args.right_tracker_name,
        head_tracker_name=args.head_tracker_name,
        libsurvive_tracking_frame=args.libsurvive_tracking_frame,
        manus_topics=args.manus_topics,
        manus_msg_type=args.manus_msg_type,
        vive_input_frame=args.vive_input_frame,
        stale_timeout=args.ros_stale_timeout,
        position_scale=args.vive_position_scale,
        apply_unitree_arm_orientation=not args.no_unitree_arm_orientation_fix and args.vive_input_frame != "waist",
    )

    # =========================
    # Robot arm / IK selection
    # =========================
    if args.arm == "G1_29":
        arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        arm_ik = G1_29_ArmIK()
    else:
        raise ValueError("Unsupported arm type")

    # =========================
    # End-effector selection
    # =========================
    if args.ee == "dex3":
        left_hand_pos_array = Array('d', 75, lock=True)
        right_hand_pos_array = Array('d', 75, lock=True)
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 14, lock=False)
        dual_hand_action_array = Array('d', 14, lock=False)
        hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock,
                                      dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
    elif args.ee == "dex1":
        left_gripper_value = Value('d', 0.0, lock=True)
        right_gripper_value = Value('d', 0.0, lock=True)
        dual_gripper_data_lock = Lock()
        dual_gripper_state_array = Array('d', 2, lock=False)
        dual_gripper_action_array = Array('d', 2, lock=False)
        gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock,
                                                 dual_gripper_state_array, dual_gripper_action_array, simulation_mode=args.sim)
    elif args.ee == "inspire1":
        left_hand_pos_array = Array('d', 75, lock=True)
        right_hand_pos_array = Array('d', 75, lock=True)
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 12, lock=False)
        dual_hand_action_array = Array('d', 12, lock=False)
        # tactile sizes (must match controller)
        tactile_field_sizes = [9, 96, 80, 9, 96, 80, 9, 96, 80, 9, 96, 80, 9, 96, 9, 96, 112]
        tactile_total_size = sum(tactile_field_sizes)
        left_hand_tactile_array = Array('d', tactile_total_size, lock=True)
        right_hand_tactile_array = Array('d', tactile_total_size, lock=True)

        hand_ctrl = Inspire_Controller(
            left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock,
            dual_hand_state_array, dual_hand_action_array,
        )
    elif args.ee == "brainco":
        left_hand_pos_array = Array('d', 75, lock=True)
        right_hand_pos_array = Array('d', 75, lock=True)
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 12, lock=False)
        dual_hand_action_array = Array('d', 12, lock=False)
        hand_ctrl = Brainco_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock,
                                       dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
    else:
        logger_mp.info("End-effector control disabled (--ee none).")
        hand_ctrl = None

    # =========================
    # Simulation hooks
    # =========================
    if args.sim:
        reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
        reset_pose_publisher.Init()
        from teleop.utils.sim_state_topic import start_sim_state_subscribe
        sim_state_subscriber = start_sim_state_subscribe()

    head_camera_dds = None
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

    # Controller locomotion (optional)
    if args.xr_mode == "controller" and args.motion:
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        sport_client = LocoClient()
        sport_client.SetTimeout(0.0001)
        sport_client.Init()

    # Recorder
    if args.record:
        recorder = EpisodeWriter(task_dir=args.task_dir, frequency=args.frequency, rerun_log=not args.headless)

    
    try:
        # =========================
        # Pre-start preview (window always ON; keys only from terminal)
        # =========================
        #logger_mp.info("Press 'r' in the terminal to start. 'q' to quit.")
        #while not start_signal and running:
            #if not args.headless:
                # show head image preview
            #    tv_resized = cv2.resize(tv_img_array, (tv_img_shape[1] // 2, tv_img_shape[0] // 2))
            #    cv2.imshow("record image", tv_resized)
            #    cv2.waitKey(1)
            #time.sleep(0.01)

        if not running:
            raise KeyboardInterrupt

        # Arm safe speed enable once when starting ACTIVE
        arm_ctrl.speed_gradual_max()
        arm_ctrl.go_initial_pose(arm_ik)
        
        #fsm = Mode.ACTIVE
        fsm = Mode.STANDBY
        #logger_mp.info("Entering ACTIVE mode.")

        # =========================
        # Main loop
        # =========================
        grace_until = time.time() + 2.0     # 시작 후 2초 동안은 종료 버튼 무시
        exit_hold_start = None              # A 버튼 길게 누름 디텍션
        
        while running:
            loop_start = time.time()
            
            # A) Always show preview if not headless (no key handling here)
            #if not args.headless:
            #    tv_resized = cv2.resize(tv_img_array, (tv_img_shape[1] // 2, tv_img_shape[0] // 2))
            #    cv2.imshow("record image", tv_resized)
            #    cv2.waitKey(1)

            # B) optional sim reset
            if should_reset_scene:
                should_reset_scene = False
                if args.sim:
                    publish_reset_category(2, reset_pose_publisher)

            # B2) manual recalibration (key 'c')
            if should_recalibrate:
                should_recalibrate = False
                tv_wrapper.calibrate()
                if head_camera_dds is not None:
                    head_camera_dds.recalibrate(tv_wrapper)
                last_good_left_pose = last_good_right_pose = None

            # C) recorder toggle
            if args.record and should_toggle_recording:
                should_toggle_recording = False
                #if not 'is_recording' in globals():
                #    is_recording = False  # ensure defined
                if not is_recording:
                    if recorder.create_episode():
                        is_recording = True
                        episode_started = True
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    is_recording = False
                    recorder.save_episode()
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)
            
            # D) Read Vive Tracker + Manus data
            tele_data = tv_wrapper.get_tele_data()

            if head_camera_dds is not None:
                head_camera_dds.maybe_publish(tv_wrapper)

            # Hand controllers run in their own process/thread and read these
            # shared arrays. Each frame we publish the latest wrist-local Manus
            # joint positions for left/right hand retargeting.
            if args.ee in ("dex3", "inspire1", "brainco"):
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
            
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif args.ee == "dex1" and args.xr_mode == "controller":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_trigger_value
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_trigger_value
            elif args.ee == "dex1" and args.xr_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_pinch_value
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_pinch_value
            else:
                pass 
                 

            # E) High-level locomotion (controller + motion)
            if args.xr_mode == "controller" and args.motion :#and tele_data.tele_state is not None:
                # quit teleoperate via controller A button
                if getattr(tele_data.tele_state, 'right_aButton', False):
                    running = False
                    stop_listening()
                # damping when both thumbsticks pressed
                if getattr(tele_data.tele_state, 'left_thumbstick_state', False) and \
                   getattr(tele_data.tele_state, 'right_thumbstick_state', False):
                    sport_client.Damp()
                # velocity control (bounded)
                logger_mp.info("Entering ACTIVE 1 mode.") 
                ls = getattr(tele_data.tele_state, 'left_thumbstick_value', np.zeros(2))
                rs = getattr(tele_data.tele_state, 'right_thumbstick_value', np.zeros(2))
                sport_client.Move(-ls[1] * 0.3, -ls[0] * 0.3, -rs[0] * 0.3)

            # F) FSM for arm control
            tracking = getattr(tele_data, 'tracking_active', True)  # fallback True if older wrapper
            session_alive = getattr(tele_data, 'session_alive', True)
            now = time.time()
            last_reconnect_try = 0.0
            RECONNECT_TIMEOUT  = 5.0   # 세션/트래킹 둘 다 죽은 상태가 5초 넘으면 재연결 시도

            if fsm == Mode.ACTIVE:
                # periodic debug: tracker active + calibration state
                _dbg_now = time.time()
                if _dbg_now - getattr(on_press, '_last_dbg', 0.0) >= 2.0:
                    on_press._last_dbg = _dbg_now
                    lc = tv_wrapper._calibrated["left"]
                    rc = tv_wrapper._calibrated["right"]
                    ta = tv_wrapper._tracker_active
                    la = tele_data.left_arm_pose[:3, 3].round(3)
                    ra = tele_data.right_arm_pose[:3, 3].round(3)
                    print(f"[DBG] tracker_active={ta}  calib=L:{lc} R:{rc}  "
                          f"arm_pos L={la} R={ra}", flush=True)
                


                if not tele_data.tracking_active:  # tracking_active 사용
                    if lost_since is None:
                        lost_since = now
                    elif now - lost_since > LOST_TIMEOUT:
                        logger_mp.info("Tracking lost → STANDBY")
                        set_in_standby = False
                        fsm = Mode.STANDBY                   
                else:
                    lost_since = None

                    if tv_wrapper._tracker_active:
                        if not tv_wrapper._calibrated["left"] or not tv_wrapper._calibrated["right"]:
                            logger_mp.warning("[SYNC] Calibration missing in ACTIVE; returning to STANDBY. Press 'r' to sync again.")
                            last_good_left_pose = last_good_right_pose = None
                            set_in_standby = False
                            fsm = Mode.STANDBY
                            continue

                        if pose_filter_enabled and last_good_left_pose is not None:
                            left_jump = np.linalg.norm(
                                tele_data.left_arm_pose[:3, 3] - last_good_left_pose[:3, 3]
                            )
                            right_jump = np.linalg.norm(
                                tele_data.right_arm_pose[:3, 3] - last_good_right_pose[:3, 3]
                            )

                            MAX_FRAME_JUMP = 0.15  # 15cm로 늘림

                            if left_jump > MAX_FRAME_JUMP:
                                tele_data.left_arm_pose = last_good_left_pose.copy()
                            else:
                                last_good_left_pose = tele_data.left_arm_pose.copy()

                            if right_jump > MAX_FRAME_JUMP:
                                tele_data.right_arm_pose = last_good_right_pose.copy()
                            else:
                                last_good_right_pose = tele_data.right_arm_pose.copy()
                        else:
                            last_good_left_pose = tele_data.left_arm_pose.copy()
                            last_good_right_pose = tele_data.right_arm_pose.copy()

                        current_lr_arm_q = arm_ctrl.get_current_dual_arm_q()
                        current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()
                        try:
                            sol_q, sol_tauff = arm_ik.solve_ik(
                                tele_data.left_arm_pose, tele_data.right_arm_pose,
                                current_lr_arm_q, current_lr_arm_dq
                            )
                            arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
                        except Exception as _ik_e:
                            logger_mp.error(f"[IK/ctrl] exception: {_ik_e}", exc_info=True)
            elif fsm == Mode.STANDBY:
                session_alive = getattr(tele_data, 'session_alive', True)
                
                if not set_in_standby:
                    logger_mp.info("Now STANDBY mode. Press 'r' to sync tracker with sim and start.")
                    #hand_ctrl.enter_standby_closed()
                    arm_ctrl.go_ready_pose(arm_ik)
                    set_in_standby = True

                if should_start_sync:
                    if sync_wait_until is not None and now < sync_wait_until:
                        time.sleep(max(0.0, min(0.05, sync_wait_until - now)))
                        continue
                    should_start_sync = False
                    sync_wait_until = None
                    sync_ok, left_sim_pose, right_sim_pose, left_tracker_world, right_tracker_world = _check_tracker_sim_sync(
                        tv_wrapper,
                        arm_ik,
                        arm_ctrl,
                        args.sync_max_rotation_error_deg,
                    )
                    if sync_ok:
                        logger_mp.info("[SYNC] Accepted. Calibrating tracker against current sim pose and entering ACTIVE.")
                        arm_ctrl.speed_gradual_max()
                        calibrated = tv_wrapper.calibrate_from_robot_world(
                            left_tracker_world,
                            right_tracker_world,
                            left_sim_pose,
                            right_sim_pose,
                        )
                        if head_camera_dds is not None:
                            head_camera_dds.recalibrate(tv_wrapper)
                        if calibrated:
                            last_good_left_pose = last_good_right_pose = None
                            is_homed = False
                            found_since = None
                            lost_since = None
                            set_in_standby = False
                            fsm = Mode.ACTIVE
                        else:
                            logger_mp.error("[SYNC ALARM] Calibration failed after sync pose check; stay in STANDBY.")
                    else:
                        logger_mp.error("[SYNC ALARM] Sync request failed; stay in STANDBY.")
                elif tv_wrapper._tracker_active:
                    # if now - getattr(on_press, '_last_sync_wait', 0.0) >= 2.0:
                    #     on_press._last_sync_wait = now
                    #     print("[SYNC] tracking is available. Press 'r' to validate tracker/sim sync and start.", flush=True)
                    pass

 
                else:
                    found_since = None
                    # ② tracking도 session도 오래 죽어 있으면 XR 재연결 (한 번씩만)
                    if (not session_alive) and (now - last_reconnect_try > RECONNECT_TIMEOUT):
                    #    logger_mp.info("XR session idle for a while → restarting TeleVuer ...")
                        try:
                            tv_wrapper.reconnect()
                        except Exception as e:
                        #    logger_mp.warning(f"TeleVuer reconnect failed: {e}")
                            a = 1
                        last_reconnect_try = now


            # G) (optional) Recording payload — unchanged from original structure
            if args.record and is_recording :
                # Build and push episode items here, mirroring your original logic

                try:
                    # tv image copy
                    current_tv_image = tv_img_array.copy()
                    if WRIST:
                        current_wrist_image = wrist_img_array.copy()

                    # arm state/action
                    current_lr_arm_q = arm_ctrl.get_current_dual_arm_q()
                    left_arm_state = current_lr_arm_q[:7]
                    right_arm_state = current_lr_arm_q[-7:]
                    # last commanded q (sol_q) may not exist in STANDBY; handle safely
                    
                    try:
                        left_arm_action = sol_q[:7]
                        right_arm_action = sol_q[-7:]
                    except Exception:
                        left_arm_action, right_arm_action = left_arm_state, right_arm_state

                    # EE state/action depending on ee type (kept simple)
                    left_ee_state, right_ee_state = [], []
                    left_hand_action, right_hand_action = [], []
                    left_tactile_data, right_tactile_data = [], []

                    if args.ee == "dex3" and args.xr_mode == "hand":
                        with dual_hand_data_lock:
                            left_ee_state = dual_hand_state_array[:7]
                            right_ee_state = dual_hand_state_array[-7:]
                            left_hand_action = dual_hand_action_array[:7]
                            right_hand_action = dual_hand_action_array[-7:]
                    elif args.ee == "dex1" and args.xr_mode == "hand":
                        with dual_gripper_data_lock:
                            left_ee_state = [dual_gripper_state_array[0]]
                            right_ee_state = [dual_gripper_state_array[1]]
                            left_hand_action = [dual_gripper_action_array[0]]
                            right_hand_action = [dual_gripper_action_array[1]]
                    elif args.ee == "dex1" and args.xr_mode == "controller":
                        with dual_gripper_data_lock:
                            left_ee_state = [dual_gripper_state_array[0]]
                            right_ee_state = [dual_gripper_state_array[1]]
                            left_hand_action = [dual_gripper_action_array[0]]
                            right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-ls[1] * 0.3, -ls[0] * 0.3, -rs[0] * 0.3]
                    elif args.ee in ("inspire1", "brainco") and args.xr_mode == "hand":
                        with dual_hand_data_lock:
                            left_ee_state = dual_hand_state_array[:6]
                            right_ee_state = dual_hand_state_array[-6:]
                            left_hand_action = dual_hand_action_array[:6]
                            right_hand_action = dual_hand_action_array[-6:]
                        if args.ee == "inspire1":
                            try:
                                with left_hand_tactile_array.get_lock():
                                    left_tactile_data = np.array(left_hand_tactile_array[:])
                                with right_hand_tactile_array.get_lock():
                                    right_tactile_data = np.array(right_hand_tactile_array[:])
                            except Exception:
                                left_tactile_data = np.zeros(tactile_total_size)
                                right_tactile_data = np.zeros(tactile_total_size)

                    

                    # build colors dict
                    colors = {}
                    depths = {}
                    if BINOCULAR:
                        colors["color_0"] = current_tv_image[:, : tv_img_shape[1] // 2]
                        colors["color_1"] = current_tv_image[:, tv_img_shape[1] // 2 :]
                        if WRIST:
                            colors["color_2"] = current_wrist_image[:, : wrist_img_shape[1] // 2]
                            colors["color_3"] = current_wrist_image[:, wrist_img_shape[1] // 2 :]
                    else:
                        colors["color_0"] = current_tv_image
                        if WRIST:
                            colors["color_1"] = current_wrist_image[:, : wrist_img_shape[1] // 2]
                            colors["color_2"] = current_wrist_image[:, wrist_img_shape[1] // 2 :]

                    states = {
                        "left_arm": {"qpos": left_arm_state.tolist(), "qvel": [], "torque": []},
                        "right_arm": {"qpos": right_arm_state.tolist(), "qvel": [], "torque": []},
                        "left_ee": {"qpos": list(left_ee_state), "qvel": [], "torque": []},
                        "right_ee": {"qpos": list(right_ee_state), "qvel": [], "torque": []},
                        "body": {"qpos": []},
                    }
                    actions = {
                        "left_arm": {"qpos": list(left_arm_action), "qvel": [], "torque": []},
                        "right_arm": {"qpos": list(right_arm_action), "qvel": [], "torque": []},
                        "left_ee": {"qpos": list(left_hand_action), "qvel": [], "torque": []},
                        "right_ee": {"qpos": list(right_hand_action), "qvel": [], "torque": []},
                        "body": {"qpos": []},
                    }
                    tactile = {"left_ee": list(left_tactile_data), "right_ee": list(right_tactile_data)}


                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, sim_state=sim_state, tactiles=tactile)
                    else:
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, tactiles=tactile)


                except Exception as e:
                    logger_mp.debug(f"recording path skip: {e}")

            # H) pacing
            elapsed = time.time() - loop_start
            time.sleep(max(0, (1 / args.frequency) - elapsed))

    except KeyboardInterrupt:
        logger_mp.info("KeyboardInterrupt, exiting program…")
    finally:
        # always return to home safely
        try:
            #hand_ctrl.enter_standby_open() 
            #hand_ctrl.enter_standby_closed() 
            time.sleep(0.5)
            arm_ctrl.go_exit_pose(arm_ik)
            
            #time.sleep(1.5)
            
        except Exception:
            pass
        # stop simulation subscriber
        if args.sim:
            try:
                sim_state_subscriber.stop_subscribe()
            except Exception:
                pass
        try:
            tv_wrapper.shutdown()
        except Exception:
            pass
        # clean windows
        if not args.headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        # release shared memories
        try:
            tv_img_shm.close(); tv_img_shm.unlink()
        except Exception:
            pass
        if WRIST:
            try:
                wrist_img_shm.close(); wrist_img_shm.unlink()
            except Exception:
                pass
        # close recorder
        #if args.record:
        if args.record and episode_started:
            try:
                recorder.close()
            except Exception:
                pass
        # join keyboard thread
        try:
            listen_keyboard_thread.join(timeout=0.2)
        except Exception:
            pass
        logger_mp.info("Finally, exiting program…")
        sys.exit(0)
