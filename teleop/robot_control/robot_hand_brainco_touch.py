from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_, AudioData_                 # idl
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
import numpy as np
import struct
from enum import IntEnum
import threading
import time
from multiprocessing import Process, Array

import logging_mp
logger_mp = logging_mp.get_logger(__name__)


brainco_Num_Motors = 6
kTopicbraincoLeftCommand = "rt/brainco/left/cmd"
kTopicbraincoLeftState = "rt/brainco/left/state"
kTopicbraincoRightCommand = "rt/brainco/right/cmd"
kTopicbraincoRightState = "rt/brainco/right/state"
kTopicbraincoLeftTactile = "rt/brainco/left/tactile"
kTopicbraincoRightTactile= "rt/brainco/right/tactile"


class Brainco_Controller:
    def __init__(self, left_hand_array, right_hand_array, dual_hand_data_lock = None, dual_hand_state_array = None,
                       dual_hand_action_array = None, left_tactile_array= None, right_tactile_array = None, fps = 100.0, Unit_Test = False, simulation_mode = False):
        logger_mp.info("Initialize Brainco_Controller...")
        self.fps = fps
        self.hand_sub_ready = False
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode

        if not self.Unit_Test:
            self.hand_retargeting = HandRetargeting(HandType.BRAINCO_HAND)
        else:
            self.hand_retargeting = HandRetargeting(HandType.BRAINCO_HAND_Unit_Test)

        if self.simulation_mode:
            ChannelFactoryInitialize(1)
        else:
            ChannelFactoryInitialize(0)

        # initialize handcmd publisher and handstate subscriber
        self.LeftHandCmb_publisher = ChannelPublisher(kTopicbraincoLeftCommand, MotorCmds_)
        self.LeftHandCmb_publisher.Init()
        self.RightHandCmb_publisher = ChannelPublisher(kTopicbraincoRightCommand, MotorCmds_)
        self.RightHandCmb_publisher.Init()

        self.LeftHandState_subscriber = ChannelSubscriber(kTopicbraincoLeftState, MotorStates_)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(kTopicbraincoRightState, MotorStates_)
        self.RightHandState_subscriber.Init()

        self.LeftHandTactile_subscriber = ChannelSubscriber(kTopicbraincoLeftTactile, AudioData_)
        self.LeftHandTactile_subscriber.Init()
        self.RightHandTactile_subscriber = ChannelSubscriber(kTopicbraincoRightTactile, AudioData_)
        self.RightHandTactile_subscriber.Init()

        # Shared Arrays for hand states
        self.left_hand_state_array  = Array('d', brainco_Num_Motors, lock=True)  
        self.right_hand_state_array = Array('d', brainco_Num_Motors, lock=True)

        self.left_hand_tactile_array  = left_tactile_array
        self.right_hand_tactile_array = right_tactile_array

        # initialize subscribe threads (state and tactile are independent)
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        if left_tactile_array is not None and right_tactile_array is not None:
            self.subscribe_tactile_thread = threading.Thread(target=self._subscribe_tactile)
            self.subscribe_tactile_thread.daemon = True
            self.subscribe_tactile_thread.start()

        while not self.hand_sub_ready:
            time.sleep(0.1)
            logger_mp.warning("[brainco_Controller] Waiting to subscribe dds...")
        logger_mp.info("[brainco_Controller] Subscribe dds ok.")

        hand_control_process = Process(target=self.control_process, args=(left_hand_array, right_hand_array,  self.left_hand_state_array, self.right_hand_state_array,
                                                                         dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array))
        hand_control_process.daemon = True
        hand_control_process.start()

        left_q_target  = np.full(brainco_Num_Motors, 0)
        right_q_target = np.full(brainco_Num_Motors, 0)

        # initialize brainco hand's cmd msg
        self.left_hand_msg  = MotorCmds_()
        self.left_hand_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(len(Brainco_Left_Hand_JointIndex))]
        self.right_hand_msg = MotorCmds_()
        self.right_hand_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(len(Brainco_Right_Hand_JointIndex))]

        for idx, id in enumerate(Brainco_Left_Hand_JointIndex):
            self.left_hand_msg.cmds[id].q = 0.0
            self.left_hand_msg.cmds[id].dq = 1.0
        for idx, id in enumerate(Brainco_Right_Hand_JointIndex):
            self.right_hand_msg.cmds[id].q = 0.0
            self.right_hand_msg.cmds[id].dq = 1.0

        logger_mp.info("Initialize brainco_Controller OK!\n")

    def _subscribe_hand_state(self):
        while True:
            left_hand_msg  = self.LeftHandState_subscriber.Read()
            right_hand_msg = self.RightHandState_subscriber.Read()

            self.hand_sub_ready = True
            if left_hand_msg is not None and right_hand_msg is not None:
                # Update left hand state
                for idx, id in enumerate(Brainco_Left_Hand_JointIndex):
                    self.left_hand_state_array[idx] = left_hand_msg.states[id].q
                # Update right hand state
                for idx, id in enumerate(Brainco_Right_Hand_JointIndex):
                    self.right_hand_state_array[idx] = right_hand_msg.states[id].q

    def _subscribe_tactile(self):
        EXPECTED_BYTES = 80  # 20 uint32 × 4 bytes
        while True:
            try:
                left_tactile_msg  = self.LeftHandTactile_subscriber.Read()
                right_tactile_msg = self.RightHandTactile_subscriber.Read()

                if left_tactile_msg is None or right_tactile_msg is None:
                    continue

                left_bytes  = bytes(left_tactile_msg.data)
                right_bytes = bytes(right_tactile_msg.data)

                if len(left_bytes) != EXPECTED_BYTES or len(right_bytes) != EXPECTED_BYTES:
                    logger_mp.warning(f"[tactile] unexpected size: L={len(left_bytes)} R={len(right_bytes)}, expected {EXPECTED_BYTES}")
                    continue

                left_data  = struct.unpack('20I', left_bytes)
                right_data = struct.unpack('20I', right_bytes)

                with self.left_hand_tactile_array.get_lock():
                    self.left_hand_tactile_array[:] = left_data
                with self.right_hand_tactile_array.get_lock():
                    self.right_hand_tactile_array[:] = right_data

            except Exception as e:
                logger_mp.warning(f"[tactile] read error: {e}")

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """
        Set current left, right hand motor state target q
        """
        for idx, id in enumerate(Brainco_Left_Hand_JointIndex):             
            self.left_hand_msg.cmds[id].q = left_q_target[idx]
        for idx, id in enumerate(Brainco_Right_Hand_JointIndex):             
            self.right_hand_msg.cmds[id].q = right_q_target[idx] 

        self.LeftHandCmb_publisher.Write(self.left_hand_msg)
        self.RightHandCmb_publisher.Write(self.right_hand_msg)
        # logger_mp.debug("hand ctrl publish ok.")
        
    def ctrl_dual_hand_replay(self, left_q_target, right_q_target):
        """
        Set current left, right hand motor state target q
        """
        
        for idx, id in enumerate(Brainco_Left_Hand_JointIndex):             
            self.left_hand_msg.cmds[id].q = left_q_target[idx]
        for idx, id in enumerate(Brainco_Right_Hand_JointIndex):             
            self.right_hand_msg.cmds[id].q = right_q_target[idx] 

        self.LeftHandCmb_publisher.Write(self.left_hand_msg)
        self.RightHandCmb_publisher.Write(self.right_hand_msg)
        logger_mp.debug("hand ctrl publish ok.")
    
    def control_process(self, left_hand_array, right_hand_array, left_hand_state_array, right_hand_state_array,
                              dual_hand_data_lock = None, dual_hand_state_array = None, dual_hand_action_array = None):
        self.running = True

        left_q_target  = np.full(brainco_Num_Motors, 0)
        right_q_target = np.full(brainco_Num_Motors, 0)

        # initialize brainco hand's cmd msg
        self.left_hand_msg  = MotorCmds_()
        self.left_hand_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(len(Brainco_Left_Hand_JointIndex))]
        self.right_hand_msg = MotorCmds_()
        self.right_hand_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(len(Brainco_Right_Hand_JointIndex))]

        for idx, id in enumerate(Brainco_Left_Hand_JointIndex):
            self.left_hand_msg.cmds[id].q = 0.0
            self.left_hand_msg.cmds[id].dq = 1.0
        for idx, id in enumerate(Brainco_Right_Hand_JointIndex):
            self.right_hand_msg.cmds[id].q = 0.0
            self.right_hand_msg.cmds[id].dq = 1.0
        
        try:
            while self.running:
                start_time = time.time()
                # get dual hand state
                with left_hand_array.get_lock():
                    left_hand_data  = np.array(left_hand_array[:]).reshape(25, 3).copy()
                with right_hand_array.get_lock():
                    right_hand_data = np.array(right_hand_array[:]).reshape(25, 3).copy()

                # Read left and right q_state from shared arrays
                state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))

                if not np.all(right_hand_data == 0.0) and not np.all(left_hand_data[4] == np.array([-1.13, 0.3, 0.15])): # if hand data has been initialized.
                    ref_left_value = left_hand_data[self.hand_retargeting.left_indices[1,:]] - left_hand_data[self.hand_retargeting.left_indices[0,:]]
                    ref_right_value = right_hand_data[self.hand_retargeting.right_indices[1,:]] - right_hand_data[self.hand_retargeting.right_indices[0,:]]

                    left_q_target  = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[self.hand_retargeting.left_dex_retargeting_to_hardware]
                    right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]

                    # In the official document, the angles are in the range [0, 1] ==> 0.0: fully open  1.0: fully closed
                    # The q_target now is in radians, ranges:
                    #     - idx 0:   0~1.52/
                    #     - idx 1:   0~1.05
                    #     - idx 2~5: 0~1.47
                    # We normalize them using (max - value) / range
                    def normalize(val, min_val, max_val):
                        return 1.0 - np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)

                    for idx in range(brainco_Num_Motors):
                        if idx == 0:
                            left_q_target[idx]  = normalize(left_q_target[idx], 0.0, 1.9) #2.00
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 1.9)
                        elif idx == 1:
                            left_q_target[idx]  = normalize(left_q_target[idx], 0.0, 1.00)#1.0
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 1.00)
                        elif idx >= 2:
                            left_q_target[idx]  = normalize(left_q_target[idx], 0.0, 0.90)
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 0.90)
                # get dual hand action
                action_data = np.concatenate((left_q_target, right_q_target))    
                if dual_hand_state_array and dual_hand_action_array:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data
                # logger_mp.info(f"left_q_target:{left_q_target}")
                self.ctrl_dual_hand(left_q_target, right_q_target)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("brainco_Controller has been closed.")

# according to the official documentation, https://www.brainco-hz.com/docs/revolimb-hand/product/parameters.html
# the motor sequence is as shown in the table below
# ┌──────┬───────┬────────────┬────────┬────────┬────────┬────────┐
# │ Id   │   0   │     1      │   2    │   3    │   4    │   5    │
# ├──────┼───────┼────────────┼────────┼────────┼────────┼────────┤
# │Joint │ thumb │ thumb-aux  |  index │ middle │  ring  │  pinky │
# └──────┴───────┴────────────┴────────┴────────┴────────┴────────┘
class Brainco_Right_Hand_JointIndex(IntEnum):
    kRightHandThumb = 0
    kRightHandThumbAux = 1
    kRightHandIndex = 2
    kRightHandMiddle = 3
    kRightHandRing = 4
    kRightHandPinky = 5

class Brainco_Left_Hand_JointIndex(IntEnum):
    kLeftHandThumb = 0
    kLeftHandThumbAux = 1
    kLeftHandIndex = 2
    kLeftHandMiddle = 3
    kLeftHandRing = 4
    kLeftHandPinky = 5


class Brainco_Left_Hand_Joint(IntEnum):
    kLeftHandThumb = 0
    kLeftHandIndex = 1
    kLeftHandMiddle = 2
    kLeftHandRing = 3
    kLeftHandPinky = 4

