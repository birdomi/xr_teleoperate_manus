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
import json
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
from teleop.robot_control.robot_hand_brainco_replay import Brainco_Controller
from teleop.image_server.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter


import numpy as np

def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data["data"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help="Path to data.json")
    parser.add_argument("--frequency", type=float, default=60.0, help="Replay frequency (Hz)")
    parser.add_argument("--sim", action="store_true", help="Use simulation mode")
    parser.add_argument("--arm", type=str, default="G1_29", choices=["G1_29", "G1_23", "H1_2", "H1"], help="Arm model")
    parser.add_argument('--ee', type=str, default='brainco', choices=['dex1', 'dex3', 'inspire1', 'brainco'], help='Select end effector controller')
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')

    args = parser.parse_args()

    # 로봇 컨트롤러 초기화
    if args.arm == "G1_29":
        arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        arm_ik = G1_29_ArmIK()
    else:
        raise ValueError("Unsupported arm type.")
    
    if args.ee == "inspire1":
        left_hand_pos_array = Array('d', 75, lock = True)      # [input]
        right_hand_pos_array = Array('d', 75, lock = True)     # [input]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
        dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
        # hand_ctrl = Inspire_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim) # sim, dfq hand
        hand_ctrl = Inspire_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, network_interface="enp109s0")   # real, ftp hand
        
    elif args.ee == "brainco":
        left_hand_pos_array = Array('d', 75, lock = True)      # [input]
        right_hand_pos_array = Array('d', 75, lock = True)     # [input]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
        dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
        hand_ctrl = Brainco_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
    
    else:
        pass

    # 데이터 로드
    trajectory = load_data(args.json_path)
    print(f"Loaded {len(trajectory)} frames.")

    try:
        arm_ctrl.speed_gradual_max()
        arm_ctrl.go_initial_pose(arm_ik)
        arm_ctrl.go_initial_pose_2(arm_ik)
        time.sleep(1.0)

        # arm_ctrl.go_initial_pose_social(arm_ik)

        print("replay start")

        for frame in trajectory:
            left_qpos = frame["actions"]["left_arm"]["qpos"]
            right_qpos = frame["actions"]["right_arm"]["qpos"]
            left_ee_qpos = frame["actions"]["left_ee"]["qpos"]
            right_ee_qpos = frame["actions"]["right_ee"]["qpos"]

            # left_ee_state = frame["states"]["left_ee"]["qpos"]
            # right_ee_state = frame["states"]["right_ee"]["qpos"]

            #hand_ctrl.replay(left_ee_qpos, right_ee_qpos)  # for inspire ftp hand
            hand_ctrl.ctrl_dual_hand_replay(left_ee_qpos, right_ee_qpos) # for brainCo hand

            joint_cmd = left_qpos + right_qpos
            tau = arm_ik.get_tauff(np.array(joint_cmd))
            # tau = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            arm_ctrl.ctrl_dual_arm(joint_cmd, tau)
            time.sleep(1.0 / args.frequency)

            if frame['idx'] == 0:
                time.sleep(2.0)

    except KeyboardInterrupt:
        print("Replay interrupted.")
    finally:
        print("replay finish, go home")
        arm_ctrl.go_exit_pose(arm_ik)
        print("end")
        # arm_ctrl.ctrl_dual_arm_go_home()

if __name__ == "__main__":
    main()
