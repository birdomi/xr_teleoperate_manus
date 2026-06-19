import numpy as np
import threading
import time
from enum import IntEnum

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import ( LowCmd_  as hg_LowCmd, LowState_ as hg_LowState) # idl for g1, h1_2
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC

from unitree_sdk2py.idl.unitree_go.msg.dds_ import ( LowCmd_  as go_LowCmd, LowState_ as go_LowState)  # idl for h1
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_

import logging_mp
import pinocchio as pin
import numpy as np
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
from teleop.utils.weighted_moving_filter import WeightedMovingFilter


logger_mp = logging_mp.get_logger(__name__)

kTopicLowCommand_Debug  = "rt/lowcmd"
kTopicLowCommand_Motion = "rt/arm_sdk"
kTopicLowState = "rt/lowstate"

G1_29_Num_Motors = 35
G1_23_Num_Motors = 35
H1_2_Num_Motors = 35
H1_Num_Motors = 20
H2_Num_Motors = 35
 

class MotorState:
    def __init__(self):
        self.q = None
        self.dq = None

class G1_29_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(G1_29_Num_Motors)]

class G1_23_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(G1_23_Num_Motors)]

class H1_2_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(H1_2_Num_Motors)]

class H1_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(H1_Num_Motors)]

class H2_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(H2_Num_Motors)]


class DataBuffer:
    def __init__(self):
        self.data = None
        self.lock = threading.Lock()

    def GetData(self):
        with self.lock:
            return self.data

    def SetData(self, data):
        with self.lock:
            self.data = data


class G1_29_ArmController:
    def __init__(self, motion_mode = False, simulation_mode = False):
        logger_mp.info("Initialize G1_29_ArmController...")
        self.q_target = np.zeros(14)
        self.q_target[3]    = 1.57  # left elbow initial bend
        self.q_target[7+3]  = 1.57  # right elbow  initial bend 
        self.tauff_target = np.zeros(14)        
        self.motion_mode = motion_mode
        self.simulation_mode = simulation_mode
        self.kp_high = 300.0
        self.kd_high = 3.0
        self.kp_low = 80.0
        self.kd_low = 3.0
        self.kp_wrist = 40.0
        self.kd_wrist = 1.5

        self.all_motor_q = None
        self.arm_velocity_limit = 20.0
        self.control_dt = 1.0 / 250.0

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None

        
        # initialize lowcmd publisher and lowstate subscriber
        if self.simulation_mode:
            ChannelFactoryInitialize(1)
        else:
            ChannelFactoryInitialize(0)

        if self.motion_mode:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Motion, hg_LowCmd)
        else:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()

        # initialize subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        while not self.lowstate_buffer.GetData():
            time.sleep(0.1)
            logger_mp.warning("[G1_29_ArmController] Waiting to subscribe dds...")
        logger_mp.info("[G1_29_ArmController] Subscribe dds ok.")

        
        # initialize hg's lowcmd msg
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()

        self.all_motor_q = self.get_current_motor_q()
        logger_mp.info(f"Current all body motor state q:\n{self.all_motor_q} \n")
        logger_mp.info(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        logger_mp.info("Lock all joints except two arms...\n")
        
        
        arm_indices = set(member.value for member in G1_29_JointArmIndex)
        for id in G1_29_JointIndex:
            self.msg.motor_cmd[id].mode = 1
            if id.value in arm_indices:
                if self._Is_wrist_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_wrist
                    self.msg.motor_cmd[id].kd = self.kd_wrist
                else:
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
            else:
                if self._Is_weak_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
                else:
                    self.msg.motor_cmd[id].kp = self.kp_high
                    self.msg.motor_cmd[id].kd = self.kd_high
                self.msg.motor_cmd[id].q  = self.all_motor_q[id]
        logger_mp.info("Lock OK!\n")

        # initialize publish thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self.publish_thread.daemon = True
        self.publish_thread.start()

        logger_mp.info("Initialize G1_29_ArmController OK!")

    def _subscribe_motor_state(self):
        while True:
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = G1_29_LowState()
                for id in range(G1_29_Num_Motors):
                    lowstate.motor_state[id].q  = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                self.lowstate_buffer.SetData(lowstate)
            time.sleep(0.002)

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target

    def _ctrl_motor_state(self):
        if self.motion_mode:
            self.msg.motor_cmd[G1_29_JointIndex.kNotUsedJoint0].q = 1.0;

        while True:
            start_time = time.time()

            with self.ctrl_lock:
                arm_q_target     = self.q_target
                arm_tauff_target = self.tauff_target

            if self.simulation_mode:
                cliped_arm_q_target = arm_q_target
            else:
                cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit = self.arm_velocity_limit)

            for idx, id in enumerate(G1_29_JointArmIndex):
                self.msg.motor_cmd[id].q = cliped_arm_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = arm_tauff_target[idx]   

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)

            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)
            # logger_mp.debug(f"arm_velocity_limit:{self.arm_velocity_limit}")
            # logger_mp.debug(f"sleep_time:{sleep_time}")

    def ctrl_dual_arm(self, q_target, tauff_target):
        '''Set control target values q & tau of the left and right arm motors.'''
        with self.ctrl_lock:
            self.q_target = q_target
            self.tauff_target = tauff_target

    def get_mode_machine(self):
        '''Return current dds mode machine.'''
        return self.lowstate_subscriber.Read().mode_machine
    
    def get_current_motor_q(self):
        '''Return current state q of all body motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_29_JointIndex])
    
    def get_current_dual_arm_q(self):
        '''Return current state q of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_29_JointArmIndex])
    
    def get_current_dual_arm_dq(self):
        '''Return current state dq of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq for id in G1_29_JointArmIndex])
    
    def ctrl_dual_arm_go_home(self):
        '''Move both the left and right arms of the robot to their home position by setting the target joint angles (q) and torques (tau) to zero.'''
        logger_mp.info("[G1_29_ArmController] ctrl_dual_arm_go_home start...")
        max_attempts = 20
        current_attempts = 0
        with self.ctrl_lock:
            self.q_target = np.zeros(14)
            self.q_target[3]    = 1.57  # left elbow initial bend
            self.q_target[7+3]  = 1.57  # right elbow  initial bend
            # self.tauff_target = np.zeros(14)
        tolerance = 0.05  # Tolerance threshold for joint angles to determine "close to zero", can be adjusted based on your motor's precision requirements
        while current_attempts < max_attempts:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                if self.motion_mode:
                    for weight in np.linspace(1, 0, num=101):
                        self.msg.motor_cmd[G1_29_JointIndex.kNotUsedJoint0].q = weight;
                        time.sleep(0.02)
                logger_mp.info("[G1_29_ArmController] both arms have reached the home position.")
                break
            current_attempts += 1
            time.sleep(0.05)

    def speed_gradual_max(self, t = 5.0):
        '''Parameter t is the total time required for arms velocity to gradually increase to its maximum value, in seconds. The default is 5.0.'''
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        '''set arms velocity to the maximum value immediately, instead of gradually increasing.'''
        self.arm_velocity_limit = 30.0

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            G1_29_JointIndex.kLeftAnklePitch.value,
            G1_29_JointIndex.kRightAnklePitch.value,
            # Left arm
            G1_29_JointIndex.kLeftShoulderPitch.value,
            G1_29_JointIndex.kLeftShoulderRoll.value,
            G1_29_JointIndex.kLeftShoulderYaw.value,
            G1_29_JointIndex.kLeftElbow.value,
            # Right arm
            G1_29_JointIndex.kRightShoulderPitch.value,
            G1_29_JointIndex.kRightShoulderRoll.value,
            G1_29_JointIndex.kRightShoulderYaw.value,
            G1_29_JointIndex.kRightElbow.value,
        ]
        return motor_index.value in weak_motors
    
    def _Is_wrist_motor(self, motor_index):
        wrist_motors = [
            G1_29_JointIndex.kLeftWristRoll.value,
            G1_29_JointIndex.kLeftWristPitch.value,
            G1_29_JointIndex.kLeftWristyaw.value,
            G1_29_JointIndex.kRightWristRoll.value,
            G1_29_JointIndex.kRightWristPitch.value,
            G1_29_JointIndex.kRightWristYaw.value,
        ]
        return motor_index.value in wrist_motors
    
    q_target = np.array([
            # 왼팔 7개 관절
            0.0, 1.57, 0.0, 1.57, 0.0, 0.0, 0.0,
            # 오른팔 7개 관절  
            0.0, -1.57, 0.0, 1.57, 0.0, 0.0, 0.0
            ])
    
    def move_dual_arm_to_q_with_gravity(
        self,
        arm_ik,
        q_target,
        t_move: float = 1.5,
        hz: float = 250.0,
        use_current_dq: bool = True,
    ):
        """
        지정한 목표 관절각(q_target)까지 t_move 동안 선형 보간으로 부드럽게 이동.
        매 스텝마다 Pinocchio rnea()를 이용해 중력보상(및 Coriolis/원심항) 토크를 계산해 함께 보냄.
        """
        q_target = np.asarray(q_target).reshape(-1)
        nv = arm_ik.reduced_robot.model.nv
        assert q_target.shape[0] == nv, f"q_target 길이 {q_target.shape[0]} != 모델 nv {nv}"

        # 시작 상태
        q_now = self.get_current_dual_arm_q()
        steps = max(1, int(t_move * hz))
        dt = 1.0 / hz
        last_q = q_now.copy()

        for k in range(1, steps + 1):
            s = k / steps
            q_cmd = (1.0 - s) * q_now + s * q_target

            if use_current_dq:
                dq_cmd = self.get_current_dual_arm_dq()
            else:
                dq_cmd = (q_cmd - last_q) / dt  # 수치미분 근사

            tauff = pin.rnea(
                arm_ik.reduced_robot.model,
                arm_ik.reduced_robot.data,
                q_cmd,
                dq_cmd,
                np.zeros(nv),
            )

            self.ctrl_dual_arm(q_cmd, tauff)

            last_q = q_cmd
            time.sleep(dt)

    def move_lower_body_to_standing(self, t_move: float = 3.0, hz: float = 250.0):
        """
        하체를 서 있는 기본 자세로 t_move 시간 동안 부드럽게 이동시킨다.
        팔 관절은 건드리지 않고, 하체 joint만 업데이트한다.
        """
        import time
        import numpy as np

        # 하체 target posture (라디안)
        q_stand = {
            G1_29_JointIndex.kLeftHipPitch:   -0.2,
            G1_29_JointIndex.kLeftHipRoll:     0.0,
            G1_29_JointIndex.kLeftHipYaw:      0.0,
            G1_29_JointIndex.kLeftKnee:        0.4,
            G1_29_JointIndex.kLeftAnklePitch: -0.2,
            G1_29_JointIndex.kLeftAnkleRoll:   0.0,

            G1_29_JointIndex.kRightHipPitch:  -0.2,
            G1_29_JointIndex.kRightHipRoll:    0.0,
            G1_29_JointIndex.kRightHipYaw:     0.0,
            G1_29_JointIndex.kRightKnee:       0.4,
            G1_29_JointIndex.kRightAnklePitch:-0.2,
            G1_29_JointIndex.kRightAnkleRoll:  0.0,

            G1_29_JointIndex.kWaistYaw:        0.0,
            G1_29_JointIndex.kWaistRoll:       0.0,
            G1_29_JointIndex.kWaistPitch:      0.0,
        }

        # 현재 q → target q 보간
        q_now = self.get_current_motor_q()
        q_target = q_now.copy()
        for jid, q_val in q_stand.items():
            q_target[jid] = q_val

        steps = max(1, int(t_move * hz))
        dt = 1.0 / hz

        for k in range(1, steps + 1):
            s = k / steps
            q_cmd = (1.0 - s) * q_now + s * q_target

            with self.ctrl_lock:
                # 오직 하체 joint만 업데이트!
                for jid in q_stand.keys():
                    self.msg.motor_cmd[jid].mode = 1
                    self.msg.motor_cmd[jid].kp   = self.kp_high
                    self.msg.motor_cmd[jid].kd   = self.kd_high
                    self.msg.motor_cmd[jid].q    = q_cmd[jid]
                    self.msg.motor_cmd[jid].dq   = 0.0
                    self.msg.motor_cmd[jid].tau  = 0.0

            time.sleep(dt)

        logger_mp.info("[G1_29_ArmController] Lower body standing motion completed.")

    def go_ready_pose(self, arm_ik):
        q1 = np.zeros(14)
        q1[3]  -= 0.3 ; q1[7+3] -= 0.3
        self.move_dual_arm_to_q_with_gravity(arm_ik, q1, t_move=1.5)
        
    def go_exit_pose1(self, arm_ik):

        self.go_ready_pose(arm_ik)

        q1 = self.get_current_dual_arm_q()
        q1[1]  = 1.57 ; q1[7+1] = -1.57
        #어깨 90도 좌우로 벌린 자세
        self.move_dual_arm_to_q_with_gravity(arm_ik, q1, t_move=1.5)
        # 팔꿈치 90도 좌우로 벌린 자세
        q1[3]  = 1.57 ; q1[7+3] = 1.57
        self.move_dual_arm_to_q_with_gravity(arm_ik, q1, t_move=1.5)
        q1[1]  = 0.0 ; q1[7+1] = 0.0
        #양팔 옆으로 내린 자세
        self.move_dual_arm_to_q_with_gravity(arm_ik, q1, t_move=1.5)

    # ======================= 공통 헬퍼 =======================
    def _make_bell_sampler(self, q0, qf, max_roll=1.20, extra_elbow=0.80):
        """
        q(u): u∈[0,1] 에서
        - 전체 q0→qf는 min-jerk로
        - 어깨 roll은 종모양(sin(pi*u))으로 좌(+), 우(-) 벌렸다가 0으로 복귀
        - 팔꿈치는 중간(u=0.5)에서 더 굽혀 전방 돌출 억제
        """
        import numpy as np
        # 관절 인덱스(양팔 14자유도 기준)
        L_ROLL, R_ROLL = 1, 8
        L_ELB,  R_ELB  = 3, 10

        q0 = np.asarray(q0).reshape(-1)
        qf = np.asarray(qf).reshape(-1)

        def smoothstep5(s):
            s = np.clip(s, 0.0, 1.0)
            return s**3 * (10 - 15*s + 6*s**2)   # min-jerk, C2 연속

        def bell(s):
            s = np.clip(s, 0.0, 1.0)
            return np.sin(np.pi * s)            # 0→1→0 종모양

        def sample_q(u: float):
            mj = smoothstep5(u)
            w  = bell(u)
            q  = (1.0 - mj) * q0 + mj * qf

            # 어깨 roll: 중간에서 최대 벌림 → 테이블 회피
            q[L_ROLL] +=  max_roll * w
            q[R_ROLL] += -max_roll * w

            # 팔꿈치: 중간에서 더 굽힘 → 전방 돌출 억제
            baseL = (1.0 - mj) * q0[L_ELB] + mj * qf[L_ELB]
            baseR = (1.0 - mj) * q0[R_ELB] + mj * qf[R_ELB]
            q[L_ELB] = baseL + extra_elbow * w
            q[R_ELB] = baseR + extra_elbow * w
            return q

        return sample_q


    def _exec_planned_trajectory(self, sample_q, arm_ik, T=3.0, hz=250.0, hold_final=0.25):
        """
        주어진 sample_q(u)로 계획된 궤적을 실행:
        - dq는 계획값(유한차분)으로 계산 → 속도 연속성↑
        - 마지막 hold로 잔류 진동 제거
        """
        import time
        import numpy as np
        import pinocchio as pin

        nv   = arm_ik.reduced_robot.model.nv
        dt   = 1.0 / hz
        steps= max(1, int(T * hz))

        last_q = sample_q(0.0)
        for k in range(steps + 1):
            t = min(k * dt, T - 1e-9)
            u = t / T

            q_cmd = sample_q(u)
            if k < steps:
                q_next = sample_q((t + dt) / T)
                dq_cmd = (q_next - q_cmd) / dt
            else:
                dq_cmd = np.zeros_like(q_cmd)

            tauff = pin.rnea(
                arm_ik.reduced_robot.model,
                arm_ik.reduced_robot.data,
                q_cmd,
                dq_cmd,
                np.zeros(nv),
            )
            self.ctrl_dual_arm(q_cmd, tauff)
            last_q = q_cmd
            time.sleep(dt)

        # 정착(hold): dq=0 가정, 중력보상만 유지
        if hold_final > 0.0:
            zero = np.zeros(nv)
            end_t = time.time() + hold_final
            while time.time() < end_t:
                tauff = pin.rnea(
                    arm_ik.reduced_robot.model,
                    arm_ik.reduced_robot.data,
                    last_q, zero, zero
                )
                self.ctrl_dual_arm(last_q, tauff)
                time.sleep(dt)


    # ======================= 진입/퇴장: 간단 래퍼 =======================
    def go_initial_pose(self, arm_ik, T: float = 3.0, hz: float = 250.0,
                        max_roll: float = 1.45, extra_elbow: float = 0.40,
                        hold_final: float = 0.25, q_ready=None):
        """
        한 동작으로 테이블을 피해 준비자세로 수렴(90° ‘찍기’ 없이 부드럽게).
        q_ready가 None이면 기본 준비자세(양 팔꿈치 약간 굽힘)를 사용.
        """
        import numpy as np
        # (선택) 하체/팔 초기화가 필요하면 유지, 아니면 제거해도 무방
        #self.move_lower_body_to_standing(t_move=2.0)
        q1 = np.zeros(14)
        q1[3]  += 1.57 ; q1[7+3] += 1.57
        self.move_dual_arm_to_q_with_gravity(arm_ik, q1, t_move=1)

        q0 = self.get_current_dual_arm_q()
        if q_ready is None:
            qf = np.zeros(14); qf[3] = -0.3; qf[10] = -0.3
        else:
            qf = np.asarray(q_ready).reshape(-1); assert qf.shape[0] == 14

        sample_q = self._make_bell_sampler(q0, qf, max_roll, extra_elbow)
        self._exec_planned_trajectory(sample_q, arm_ik, T, hz, hold_final)


    def go_exit_pose(self, arm_ik, T: float = 3.5, hz: float = 250.0,
                    max_roll: float = 1.45, extra_elbow: float = 0.40,
                    hold_final: float = 0.25, q_exit=None):
        """
        현재자세 → (옆으로 부드럽게 벌리며 회피) → 퇴장/홈자세.
        q_exit이 None이면 기본 홈자세(팔꿈치 아주 약간 굽힘)를 사용.
        """
        import numpy as np
        q0 = self.get_current_dual_arm_q()
        if q_exit is None:
            qf = np.zeros(14); qf[3] = 1.57; qf[10] = 1.57; qf[1] = 0.17; qf[8] = -0.17
        else:
            qf = np.asarray(q_exit).reshape(-1); assert qf.shape[0] == 14

        sample_q = self._make_bell_sampler(q0, qf, max_roll, extra_elbow)
        self._exec_planned_trajectory(sample_q, arm_ik, T, hz, hold_final)

    def go_initial_pose_2(self, arm_ik):

        q1 = self.get_current_dual_arm_q()
        q1[0] = -0.349066; q1[1] = 0.698132; q1[2] = 0.087266; q1[3] = 0.436332; q1[4] = 0.296706; q1[5] = -1.221730; q1[6] = -0.087266
        q1[7] = -0.349066; q1[8] = -0.698132; q1[9] = 0.087266; q1[10] = 0.436332; q1[11] = -0.523599; q1[12] = -1.047198; q1[13] = -0.087266
        #어깨 90도 좌우로 벌린 자세
        self.move_dual_arm_to_q_with_gravity(arm_ik, q1, t_move=1.5)
   

class G1_29_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28

class G1_29_JointIndex(IntEnum):
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12
    kWaistRoll = 13
    kWaistPitch = 14

    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28
    
    # not used
    kNotUsedJoint0 = 29
    kNotUsedJoint1 = 30
    kNotUsedJoint2 = 31
    kNotUsedJoint3 = 32
    kNotUsedJoint4 = 33
    kNotUsedJoint5 = 34


class H2_ArmController:
    def __init__(self, motion_mode=False, simulation_mode=False):
        logger_mp.info("Initialize H2_ArmController...")
        self.q_target = np.zeros(14)
        self.tauff_target = np.zeros(14)
        self.motion_mode = motion_mode
        self.simulation_mode = simulation_mode
        self.kp_high = 300.0
        self.kd_high = 5.0
        self.kp_low = 140.0
        self.kd_low = 3.0
        self.kp_wrist = 50.0
        self.kd_wrist = 2.0

        self.all_motor_q = None
        self.arm_velocity_limit = 20.0
        self.control_dt = 1.0 / 250.0

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None
        
        if self.motion_mode:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Motion, hg_LowCmd)
        else:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()

        # initialize subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        while not self.lowstate_buffer.GetData():
            time.sleep(0.1)
            logger_mp.warning("[H2_ArmController] Waiting to subscribe dds...")
        logger_mp.info("[H2_ArmController] Subscribe dds ok.")

        # initialize hg's lowcmd msg
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()

        self.all_motor_q = self.get_current_motor_q()
        logger_mp.debug(f"Current all body motor state q:\n{self.all_motor_q} \n")
        logger_mp.debug(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        logger_mp.info("Lock all joints except two arms...")

        arm_indices = set(member.value for member in H2_JointArmIndex)
        for id in H2_JointIndex:
            self.msg.motor_cmd[id].mode = 1
            if id.value in arm_indices:
                if self._Is_wrist_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_wrist
                    self.msg.motor_cmd[id].kd = self.kd_wrist
                else:
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
            else:
                if self._Is_weak_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
                else:
                    self.msg.motor_cmd[id].kp = self.kp_high
                    self.msg.motor_cmd[id].kd = self.kd_high
            logger_mp.info(
                f"Motor {id.value} ({id.name}): kp={self.msg.motor_cmd[id].kp}, kd={self.msg.motor_cmd[id].kd}"
            )
            self.msg.motor_cmd[id].q = self.all_motor_q[id]
        logger_mp.info("Lock OK!")

        # initialize publish thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self.publish_thread.daemon = True
        self.publish_thread.start()

        logger_mp.info("Initialize H2_ArmController OK!")

    def _subscribe_motor_state(self):
        while True:
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = H2_LowState()
                for id in range(35):
                    lowstate.motor_state[id].q = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                self.lowstate_buffer.SetData(lowstate)
            time.sleep(0.002)

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target

    def _ctrl_motor_state(self):
        if self.motion_mode:
            self.msg.motor_cmd[H2_JointIndex.kNotUsedJoint0].q = 1.0

        while True:
            start_time = time.time()

            with self.ctrl_lock:
                arm_q_target = self.q_target
                arm_tauff_target = self.tauff_target

            if self.simulation_mode:
                cliped_arm_q_target = arm_q_target
            else:
                cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit=self.arm_velocity_limit)

            for idx, id in enumerate(H2_JointArmIndex):
                self.msg.motor_cmd[id].q = cliped_arm_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = arm_tauff_target[idx]

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)

            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)

    def ctrl_dual_arm(self, q_target, tauff_target):
        """Set control target values q & tau of the left and right arm motors."""
        with self.ctrl_lock:
            self.q_target = q_target
            self.tauff_target = tauff_target

    def get_mode_machine(self):
        """Return current dds mode machine."""
        return self.lowstate_subscriber.Read().mode_machine

    def get_current_motor_q(self):
        """Return current state q of all body motors."""
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in H2_JointIndex])

    def get_current_dual_arm_q(self):
        """Return current state q of the left and right arm motors."""
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in H2_JointArmIndex])

    def get_current_dual_arm_dq(self):
        """Return current state dq of the left and right arm motors."""
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq for id in H2_JointArmIndex])

    def ctrl_dual_arm_go_home(self):
        """Move both the left and right arms of the robot to their home position by setting the target joint angles (q) and torques (tau) to zero."""
        logger_mp.info("[H2_ArmController] ctrl_dual_arm_go_home start...")
        max_attempts = 100
        current_attempts = 0
        with self.ctrl_lock:
            self.q_target = np.zeros(14)
        tolerance = 0.05
        while current_attempts < max_attempts:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                if self.motion_mode:
                    for weight in np.linspace(1, 0, num=101):
                        self.msg.motor_cmd[H2_JointIndex.kNotUsedJoint0].q = weight
                        time.sleep(0.02)
                logger_mp.info("[H2_ArmController] both arms have reached the home position.")
                break
            current_attempts += 1
            time.sleep(0.05)

    def speed_gradual_max(self, t=5.0):
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        self.arm_velocity_limit = 30.0

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            H2_JointIndex.kLeftAnklePitch.value,
            H2_JointIndex.kRightAnklePitch.value,
            # Left arm
            H2_JointIndex.kLeftShoulderPitch.value,
            H2_JointIndex.kLeftShoulderRoll.value,
            H2_JointIndex.kLeftShoulderYaw.value,
            H2_JointIndex.kLeftElbow.value,
            # Right arm
            H2_JointIndex.kRightShoulderPitch.value,
            H2_JointIndex.kRightShoulderRoll.value,
            H2_JointIndex.kRightShoulderYaw.value,
            H2_JointIndex.kRightElbow.value,
        ]
        return motor_index.value in weak_motors

    def _Is_wrist_motor(self, motor_index):
        wrist_motors = [
            H2_JointIndex.kLeftWristRoll.value,
            H2_JointIndex.kLeftWristPitch.value,
            H2_JointIndex.kLeftWristyaw.value,
            H2_JointIndex.kRightWristRoll.value,
            H2_JointIndex.kRightWristPitch.value,
            H2_JointIndex.kRightWristYaw.value,
        ]
        return motor_index.value in wrist_motors

class H2_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28


class H2_JointIndex(IntEnum):
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12
    kWaistRoll = 13
    kWaistPitch = 14

    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28

    # Head
    kHeadPitch = 29
    kHeadYaw = 30

    # not used
    kNotUsedJoint0 = 31
    kNotUsedJoint1 = 32
    kNotUsedJoint2 = 33
    kNotUsedJoint3 = 34

if __name__ == "__main__":
    from robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK, H2_ArmIK
    import pinocchio as pin

    ChannelFactoryInitialize(1) # 0 for real robot, 1 for simulation

    # arm_ik = G1_29_ArmIK(Unit_Test = True, Visualization = False)
    # arm = G1_29_ArmController(simulation_mode=True)
    # arm_ik = G1_23_ArmIK(Unit_Test = True, Visualization = False)
    # arm = G1_23_ArmController()
    # arm_ik = H1_2_ArmIK(Unit_Test = True, Visualization = False)
    # arm = H1_2_ArmController()
    # arm_ik = H1_ArmIK(Unit_Test = True, Visualization = True)
    # arm = H1_ArmController()
    arm_ik = H2_ArmIK(Unit_Test = True, Visualization = False)
    arm = H2_ArmController()


    # initial positon
    L_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, +0.25, 0.1]),
    )

    R_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, -0.25, 0.1]),
    )

    rotation_speed = 0.005  # Rotation speed in radians per iteration

    user_input = input("Please enter the start signal (enter 's' to start the subsequent program): \n")
    if user_input.lower() == 's':
        step = 0
        arm.speed_gradual_max()
        while True:
            if step <= 120:
                angle = rotation_speed * step
                L_quat = pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0)  # y axis
                R_quat = pin.Quaternion(np.cos(angle / 2), 0, 0, np.sin(angle / 2))  # z axis

                L_tf_target.translation += np.array([0.001,  0.001, 0.001])
                R_tf_target.translation += np.array([0.001, -0.001, 0.001])
            else:
                angle = rotation_speed * (240 - step)
                L_quat = pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0)  # y axis
                R_quat = pin.Quaternion(np.cos(angle / 2), 0, 0, np.sin(angle / 2))  # z axis

                L_tf_target.translation -= np.array([0.001,  0.001, 0.001])
                R_tf_target.translation -= np.array([0.001, -0.001, 0.001])

            L_tf_target.rotation = L_quat.toRotationMatrix()
            R_tf_target.rotation = R_quat.toRotationMatrix()

            current_lr_arm_q  = arm.get_current_dual_arm_q()
            current_lr_arm_dq = arm.get_current_dual_arm_dq()

            sol_q, sol_tauff = arm_ik.solve_ik(L_tf_target.homogeneous, R_tf_target.homogeneous, current_lr_arm_q, current_lr_arm_dq)

            arm.ctrl_dual_arm(sol_q, sol_tauff)

            step += 1
            if step > 240:
                step = 0
            time.sleep(0.01)
