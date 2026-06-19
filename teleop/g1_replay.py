# replay_episode.py
import json, time, argparse
from typing import List
import threading
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from sshkeyboard import listen_keyboard, stop_listening
# === 여러분 프로젝트의 컨트롤러/SDK 경로에 맞게 import ===
from teleop.robot_control.robot_arm import G1_29_ArmController 
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK 
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from inspire_sdkpy import inspire_dds, inspire_hand_defaut
import logging_mp
logger_mp = logging_mp.get_logger(__name__)
import pinocchio as pin
import numpy as np

# Inspire Hand DDS 토픽 (robot_hand_inspire.py와 동일)
kTopicInspireCtrlLeft  = "rt/inspire_hand/ctrl/l"
kTopicInspireCtrlRight = "rt/inspire_hand/ctrl/r"

def load_episode(path):
    with open(path, "r") as f:
        epi = json.load(f)
    info = epi.get("info", {})
    fps  = (info.get("image", {}) or {}).get("fps") or 60.0
    frames = epi["data"]
    return fps, frames

def get_qpos(frame, key, default_len=None):
    # actions 우선, 없으면 states 폴백
    q = (frame.get("actions", {}).get(key, {}) or {}).get("qpos")
    if not q:
        q = (frame.get("states", {}).get(key, {}) or {}).get("qpos")
    if q and default_len and len(q) != default_len:
        raise ValueError(f"{key}.qpos length {len(q)} != {default_len}")
    return q

running = True
# --- 변경: 첫 재생 자동 시작을 위한 Event ---
replay_event = threading.Event()
replay_event.set()  # 프로그램 시작 시 1회 자동 재생

def on_press(key):
    global running
    if key == 'q':
        stop_listening()
        running = False
        replay_event.set()  # 대기 중이면 언블록
    elif key == 'c':
        # 다음 사이클 재생 요청
        replay_event.set()
    else:
        logger_mp.info(f"{key} pressed (no action mapped).")

listen_keyboard_thread = threading.Thread(
    target=listen_keyboard,
    kwargs={"on_press": on_press, "until": None, "sequential": False},
    daemon=True,
)
listen_keyboard_thread.start()

def main():
    global running

    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True, )
    ap.add_argument("--iface", default="")            # DDS NIC (빈 문자열이면 기본)
    ap.add_argument("--hz", type=float, default=60)   # 재생 주파수 강제
    ap.add_argument("--arm", default="G1_29")
    args = ap.parse_args()

    fps_file, frames = load_episode(args.episode)
    play_hz = args.hz if (args.hz and args.hz > 0) else float(fps_file or 60.0)
    dt = 1.0 / play_hz

    # DDS/컨트롤러 초기화
#   ChannelFactoryInitialize(0, args.iface)
    arm_ctrl = G1_29_ArmController(motion_mode=False, simulation_mode=False)
    arm_ik = G1_29_ArmIK()
    
    robot = pin.RobotWrapper.BuildFromURDF('../assets/g1/g1_body29_hand14.urdf', '../assets/g1/')
    
    mixed_jointsToLockIDs = [
        "left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint",
        "left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint",
        "right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint",
        "right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint",
        "waist_yaw_joint","waist_roll_joint","waist_pitch_joint",
        "left_hand_thumb_0_joint","left_hand_thumb_1_joint","left_hand_thumb_2_joint",
        "left_hand_middle_0_joint","left_hand_middle_1_joint",
        "left_hand_index_0_joint","left_hand_index_1_joint",
        "right_hand_thumb_0_joint","right_hand_thumb_1_joint","right_hand_thumb_2_joint",
        "right_hand_index_0_joint","right_hand_index_1_joint",
        "right_hand_middle_0_joint","right_hand_middle_1_joint"
    ]

    reduced_robot = robot.buildReducedRobot(
        list_of_joints_to_lock=mixed_jointsToLockIDs,
        reference_configuration=np.array([0.0] * robot.model.nq),
    )
    data = reduced_robot.model.createData()
    
    pubL = ChannelPublisher(kTopicInspireCtrlLeft,  inspire_dds.inspire_hand_ctrl);  pubL.Init()
    pubR = ChannelPublisher(kTopicInspireCtrlRight, inspire_dds.inspire_hand_ctrl);  pubR.Init()

    print(f"[Replay] N={len(frames)} | {play_hz:.2f} Hz (dt={dt*1000:.1f} ms)")
    print("Press 'q' to quit. After each run, press 'c' to replay.")

    #arm.move_lower_body_to_standing(t_move=2.0)
    #arm.ctrl_dual_arm_go_home()
    arm_ctrl.go_initial_pose(arm_ik)
    time.sleep(1.0)

    try:
        while running:
            # --- 대기: 처음에는 set돼 있으므로 즉시 통과 → 1회 자동 재생 ---
            replay_event.wait()
            replay_event.clear()
            if not running:
                break

            t0 = time.monotonic()
            for i, fr in enumerate(frames):
                if not running:
                    break

                # 1) 팔: 7+7 = 14자유도 목표 위치
                la = get_qpos(fr, "left_arm",  default_len=7)  or [0.0]*7
                ra = get_qpos(fr, "right_arm", default_len=7)  or [0.0]*7
                q_cmd = np.array(list(la) + list(ra))

                v = np.zeros(reduced_robot.model.nv)
                a = np.zeros(reduced_robot.model.nv)
                # tau_ff = pin.rnea(reduced_robot.model, data, q_cmd, v, a)
                tau_ff = arm_ik.get_tauff(np.array(q_cmd))

                arm_ctrl.ctrl_dual_arm(q_cmd, tau_ff)

                # 2) 손: 0..1 정규화 -> 0..1000 스케일 후 퍼블리시
                lle = get_qpos(fr, "left_ee",  default_len=6)
                rle = get_qpos(fr, "right_ee", default_len=6)
                if lle:
                    msgL = inspire_hand_defaut.get_inspire_hand_ctrl()
                    msgL.angle_set = [int(max(0, min(1000, v*1000))) for v in lle]
                    msgL.mode = 0b0001  # 각도 모드
                    pubL.Write(msgL)
                if rle:
                    msgR = inspire_hand_defaut.get_inspire_hand_ctrl()
                    msgR.angle_set = [int(max(0, min(1000, v*1000))) for v in rle]
                    msgR.mode = 0b0001
                    pubR.Write(msgR)

                # 3) 타이밍
                t_target = t0 + (i+1)*dt
                now = time.monotonic()
                if t_target > now:
                    time.sleep(t_target - now)

            if not running:
                break

            print("Playback finished. Press 'c' to replay, or 'q' to quit.")

    except KeyboardInterrupt:
        logger_mp.info("KeyboardInterrupt, exiting program…")
    finally:
        try:
            arm_ctrl.go_exit_pose(arm_ik)
        except Exception:
            pass
        try:
            listen_keyboard_thread.join(timeout=0.2)
        except Exception:
            pass
        logger_mp.info("Finally, exiting program…")
        return
    

if __name__ == "__main__":
    main()
