import numpy as np
import time
import argparse
import cv2
from multiprocessing import shared_memory, Value, Array, Lock
import threading
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
from televuer import TeleVuerWrapper
from teleop.robot_control.robot_arm import G1_29_ArmController 
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK 
from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller, Dex1_1_Gripper_Controller
#from teleop.robot_control.robot_hand_inspire import Inspire_Controller
# from teleop.robot_control.robot_hand_brainco import Brainco_Controller
from teleop.robot_control.robot_hand_brainco_touch import Brainco_Controller
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

# =========================
# Global state flags
# =========================
#start_signal = False

def on_press(key):
    """Terminal-only key handling.
    r: start program
    q: quit program
    s: toggle recording (if --record)
    a: (optional) sim scene reset
    """
    global running, should_toggle_recording, should_reset_scene
    #if key == 'r':
    #    start_signal = True
    #    logger_mp.info("Program start signal received.")
    if key == 'q':
        stop_listening()
        running = False
    elif key == 's':
        should_toggle_recording = True
    elif key == 'a':
        should_reset_scene = True
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
should_reset_scene = False  # terminal 'a'
set_in_standby = False  # STANDBY에서 한 번만 리셋하기 위한 플래그

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_dir', type=str, default='./utils/data', help='path to save data')
    parser.add_argument('--frequency', type=float, default=60.0, help="main loop frequency (Hz)")

    # basic control parameters
    parser.add_argument('--xr-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device tracking source')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1'], default='G1_29', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire1', 'brainco'], default='brainco', help='Select end effector controller')

    # mode flags
    parser.add_argument('--record', action='store_true', help='Enable data recording')
    parser.add_argument('--motion', action='store_true', help='Enable motion control mode')
    parser.add_argument('--headless', action='store_true', help='Disable OpenCV preview')
    parser.add_argument('--sim', action='store_true', help='Enable Isaac simulation mode')

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
    # TeleVuer (XR) wrapper
    # =========================
    tv_wrapper = TeleVuerWrapper(
        binocular=BINOCULAR,
        #use_hand_tracking=args.xr_mode == "hand",
        use_hand_tracking=True,
        img_shape=tv_img_shape,
        img_shm_name=tv_img_shm.name,
        return_state_data=True,
        return_hand_rot_data=False,
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
            left_hand_tactile_array, right_hand_tactile_array,
            network_interface="enp110s0",
        )
    elif args.ee == "brainco":
        left_hand_pos_array = Array('d', 75, lock=True)
        right_hand_pos_array = Array('d', 75, lock=True)
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 12, lock=False)
        dual_hand_action_array = Array('d', 12, lock=False)

        tactile_field_sizes = [4, 4, 4, 4, 4]
        tactile_total_size = sum(tactile_field_sizes)
        left_hand_tactile_array = Array('d', tactile_total_size, lock=True)
        right_hand_tactile_array = Array('d', tactile_total_size, lock=True)
        hand_ctrl = Brainco_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock,
                                       dual_hand_state_array, dual_hand_action_array, 
                                       left_hand_tactile_array, right_hand_tactile_array,
                                       simulation_mode=args.sim)
    else:
        hand_ctrl = None

    # =========================
    # Simulation hooks
    # =========================
    if args.sim:
        reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
        reset_pose_publisher.Init()
        from teleop.utils.sim_state_topic import start_sim_state_subscribe
        sim_state_subscriber = start_sim_state_subscribe()

    # Controller locomotion (optional)
    if args.xr_mode == "controller" and args.motion:
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        sport_client = LocoClient()
        sport_client.SetTimeout(0.0001)
        sport_client.Init()

    # Recorder
    if args.record:
        recorder = EpisodeWriter(task_dir=args.task_dir, frequency=args.frequency, rerun_log=not args.headless, show_tactile=True)

    
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
            #tele_data = tv_wrapper.get_motion_state_data()
            
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
            
            # D) Read XR data
            tele_data = tv_wrapper.get_motion_state_data()

            if (args.ee == "dex3" or args.ee == "inspire1" or args.ee == "brainco"):# and args.xr_mode == "hand":
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

                # ACTIVE 모드에서 STANDBY로 전환될 때 플래그 리셋
                # logger_mp.info("Now ACTIVE mode.")
            #    hand_ctrl.enter_auto()
                


                if not tele_data.tracking_active:  # tracking_active 사용
                    if lost_since is None:
                        lost_since = now
                    elif now - lost_since > LOST_TIMEOUT:
                        logger_mp.info("Tracking lost → STANDBY")
                        fsm = Mode.STANDBY                   
                else:
                    lost_since = None

                    if pose_filter_enabled and last_good_left_pose is not None:
                        left_jump = np.linalg.norm(
                            tele_data.left_arm_pose[:3, 3] - last_good_left_pose[:3, 3]
                        )
                        right_jump = np.linalg.norm(
                            tele_data.right_arm_pose[:3, 3] - last_good_right_pose[:3, 3]
                        )
                        
                        MAX_FRAME_JUMP = 0.15  # 15cm로 늘림
                        
                        if left_jump > MAX_FRAME_JUMP:
                            #logger_mp.warning(f"Left arm jump {left_jump:.3f}m - using last good pose")
                            tele_data.left_arm_pose = last_good_left_pose.copy()
                        else:
                            last_good_left_pose = tele_data.left_arm_pose.copy()
                            
                        if right_jump > MAX_FRAME_JUMP:
                            #logger_mp.warning(f"Right arm jump {right_jump:.3f}m - using last good pose")
                            tele_data.right_arm_pose = last_good_right_pose.copy()
                        else:
                            last_good_right_pose = tele_data.right_arm_pose.copy()

                    else:
                        # 첫 프레임
                        last_good_left_pose = tele_data.left_arm_pose.copy()
                        last_good_right_pose = tele_data.right_arm_pose.copy()

                    
                    # 정상적인 IK 처리
                    current_lr_arm_q = arm_ctrl.get_current_dual_arm_q()
                    current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()
                    sol_q, sol_tauff = arm_ik.solve_ik(
                        tele_data.left_arm_pose, tele_data.right_arm_pose,
                        current_lr_arm_q, current_lr_arm_dq
                    )
                    arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
            elif fsm == Mode.STANDBY:
                logger_mp.info("Now STANDBY mode.")
                session_alive = getattr(tele_data, 'session_alive', True)
                
                #hand_ctrl.enter_standby_closed() 
                arm_ctrl.go_ready_pose(arm_ik)
                # 스탠바이 진입
                #hand_ctrl.enter_standby_open()     # 계속 오픈 유지
                    # 계속 오픈 유지
                #    logger_mp.info("Hands reset to init position in STANDBY mode")
                
                
                # ① tracking 복귀 확정 → ACTIVE 복귀
                if tracking:
                    if found_since is None:
                        found_since = now
                    elif now - found_since > FOUND_CONFIRM:
                        logger_mp.info("Tracking restored → re-enable and enter ACTIVE")
                        arm_ctrl.speed_gradual_max()
                        is_homed = False
                        found_since = None
                        fsm = Mode.ACTIVE

 
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
                        try:
                            with left_hand_tactile_array.get_lock():
                                left_tactile_data = np.array(left_hand_tactile_array[:])
                            with right_hand_tactile_array.get_lock():
                                right_tactile_data = np.array(right_hand_tactile_array[:])
                        except Exception as E:
                            print(E)
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
