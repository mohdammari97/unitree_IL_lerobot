# for dex3-1
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, PressSensorState_  # idl
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
# for gripper
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_                           # idl
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

import numpy as np
from enum import IntEnum
import time
import os
import sys
import threading
from multiprocessing import Process, shared_memory, Array, Value, Lock

parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)
from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
from teleop.utils.weighted_moving_filter import WeightedMovingFilter

import logging_mp
logger_mp = logging_mp.get_logger(__name__)
# Import thumb pinch corrector
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)
#from thumb_pinch_corrector import ThumbPinchCorrector

# OpenXR hand joint indices for thumb, index, pinky fingertips
# Note: Human pinky (24) controls robot middle finger since robot has only 3 fingers
unitree_tip_indices = [4, 9, 24] # [thumb, index, pinky] in OpenXR -> [thumb, index, middle] on robot
Dex3_Num_Motors = 7
Dex3_Num_Pressure_Sensors = 12  # Updated to match actual sensor count
# Shared array layout for each hand:
# [q0...q6, dq0...dq6, tau0...tau6, p0...p11] (7+7+7+12=33)
# Indices:
# 0-6:   q (position)
# 7-13:  dq (velocity)
# 14-20: tau (torque)
# 21-32: pressure sensors (12 sensors)
kTopicDex3LeftCommand = "rt/dex3/left/cmd"
kTopicDex3RightCommand = "rt/dex3/right/cmd"
kTopicDex3LeftState = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"


class Dex3_1_Controller:
    def __init__(self, left_hand_array_in, right_hand_array_in, dual_hand_data_lock = None, dual_hand_state_array_out = None,
                       dual_hand_action_array_out = None, fps = 100.0, Unit_Test = False, networkInterface='enxa0cec8616f27', force=False, retargeting_method='dexpilot', simulation_mode=False):
        """
        [note] A *_array type parameter requires using a multiprocessing Array, because it needs to be passed to the internal child process.
        If force=True, shared array layout is [q0...q6, dq0...dq6, tau0...tau6, p0...p11] (33).
        If force=False, shared array layout is [q0...q6, p0...p11] (19).

        left_hand_array: [input] Left hand skeleton data (required from XR device) to hand_ctrl.control_process

        right_hand_array: [input] Right hand skeleton data (required from XR device) to hand_ctrl.control_process

        dual_hand_data_lock: Data synchronization lock for dual_hand_state_array and dual_hand_action_array

        dual_hand_state_array: [output] Return left(7), right(7) hand motor state

        dual_hand_action_array: [output] Return left(7), right(7) hand motor action

        fps: Control frequency

        Unit_Test: Whether to enable unit testing
        """
        logger_mp.info("Initialize Dex3_1_Controller...")
        self.fps = fps
        self.Unit_Test = Unit_Test
        self.force = force
        self.simulation_mode = simulation_mode #TODO Simulation mode mergen 
        if self.force:
            self.shared_array_size = 33  # [q0...q6, dq0...dq6, tau0...tau6, p0...p11]
        else:
            #self.shared_array_size = 19  # [q0...q6, p0...p11]
            self.shared_array_size = 7
        if not self.Unit_Test:
            self.hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3, retargeting_method)
        else:
            self.hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3_Unit_Test, retargeting_method)
            ChannelFactoryInitialize(0, networkInterface)

        # initialize handcmd publisher and handstate subscriber
        self.LeftHandCmb_publisher = ChannelPublisher(kTopicDex3LeftCommand, HandCmd_)
        self.LeftHandCmb_publisher.Init()
        self.RightHandCmb_publisher = ChannelPublisher(kTopicDex3RightCommand, HandCmd_)
        self.RightHandCmb_publisher.Init()

        self.LeftHandState_subscriber = ChannelSubscriber(kTopicDex3LeftState, HandState_)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(kTopicDex3RightState, HandState_)
        self.RightHandState_subscriber.Init()

        # Shared Arrays for hand states
        self.left_hand_state_array  = Array('d', self.shared_array_size, lock=True)
        self.right_hand_state_array = Array('d', self.shared_array_size, lock=True)

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while True:
            if any(self.left_hand_state_array) and any(self.right_hand_state_array):
                break
            time.sleep(0.01)
            logger_mp.warning("[Dex3_1_Controller] Waiting to subscribe dds...")
        logger_mp.info("[Dex3_1_Controller] Subscribe dds ok.")

        hand_control_process = Process(target=self.control_process, args=(left_hand_array_in, right_hand_array_in,  self.left_hand_state_array, self.right_hand_state_array,
                                                                          dual_hand_data_lock, dual_hand_state_array_out, dual_hand_action_array_out))
        hand_control_process.daemon = True
        hand_control_process.start()

        print("Initialize Dex3_1_Controller OK!\n")

    def _subscribe_hand_state(self):
        while True:
            left_hand_msg  = self.LeftHandState_subscriber.Read()
            right_hand_msg = self.RightHandState_subscriber.Read()
            if left_hand_msg is not None and right_hand_msg is not None:
                if self.force:
                    # Fill left hand: q, dq, tau, pressures
                    for idx, id in enumerate(Dex3_1_Left_JointIndex):
                        self.left_hand_state_array[idx] = left_hand_msg.motor_state[id].q
                        self.left_hand_state_array[idx+7] = left_hand_msg.motor_state[id].dq
                        self.left_hand_state_array[idx+14] = left_hand_msg.motor_state[id].tau_est
                    # Get pressure data from press_sensor_state within HandState_
                    if len(left_hand_msg.press_sensor_state) > 0:
                        press_sensor = left_hand_msg.press_sensor_state[0]
                        for id in range(min(Dex3_Num_Pressure_Sensors, len(press_sensor.pressure))):
                            self.left_hand_state_array[21+id] = press_sensor.pressure[id]
                    
                    # Fill right hand: q, dq, tau, pressures
                    for idx, id in enumerate(Dex3_1_Right_JointIndex):
                        self.right_hand_state_array[idx] = right_hand_msg.motor_state[id].q
                        self.right_hand_state_array[idx+7] = right_hand_msg.motor_state[id].dq
                        self.right_hand_state_array[idx+14] = right_hand_msg.motor_state[id].tau_est
                    # Get pressure data from press_sensor_state within HandState_
                    if len(right_hand_msg.press_sensor_state) > 0:
                        press_sensor = right_hand_msg.press_sensor_state[0]
                        for id in range(min(Dex3_Num_Pressure_Sensors, len(press_sensor.pressure))):
                            self.right_hand_state_array[21+id] = press_sensor.pressure[id]
                else:
                    # Only q and pressures
                    for idx, id in enumerate(Dex3_1_Left_JointIndex):
                        self.left_hand_state_array[idx] = left_hand_msg.motor_state[id].q
                    # # Get pressure data from press_sensor_state within HandState_
                    # if len(left_hand_msg.press_sensor_state) > 0:
                    #     press_sensor = left_hand_msg.press_sensor_state[0]
                    #     for id in range(min(Dex3_Num_Pressure_Sensors, len(press_sensor.pressure))):
                    #         self.left_hand_state_array[7+id] = press_sensor.pressure[id]
                    
                    for idx, id in enumerate(Dex3_1_Right_JointIndex):
                        self.right_hand_state_array[idx] = right_hand_msg.motor_state[id].q
                    # Get pressure data from press_sensor_state within HandState_
                    # if len(right_hand_msg.press_sensor_state) > 0:
                    #     press_sensor = right_hand_msg.press_sensor_state[0]
                    #     for id in range(min(Dex3_Num_Pressure_Sensors, len(press_sensor.pressure))):
                    #         self.right_hand_state_array[7+id] = press_sensor.pressure[id]
            time.sleep(0.002)
    
    class _RIS_Mode:
        def __init__(self, id=0, status=0x01, timeout=0):
            self.motor_mode = 0
            self.id = id & 0x0F  # 4 bits for id
            self.status = status & 0x07  # 3 bits for status
            self.timeout = timeout & 0x01  # 1 bit for timeout

        def _mode_to_uint8(self):
            self.motor_mode |= (self.id & 0x0F)
            self.motor_mode |= (self.status & 0x07) << 4
            self.motor_mode |= (self.timeout & 0x01) << 7
            return self.motor_mode

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """set current left, right hand motor state target q"""
        for idx, id in enumerate(Dex3_1_Left_JointIndex):
            self.left_msg.motor_cmd[id].q = left_q_target[idx]
        for idx, id in enumerate(Dex3_1_Right_JointIndex):
            self.right_msg.motor_cmd[id].q = right_q_target[idx]

        self.LeftHandCmb_publisher.Write(self.left_msg)
        self.RightHandCmb_publisher.Write(self.right_msg)
        # print("hand ctrl publish ok.")
    
    def control_process(self, left_hand_array, right_hand_array, left_hand_state_array, right_hand_state_array,
                              dual_hand_data_lock = None, dual_hand_state_array = None, dual_hand_action_array = None):
        self.running = True

        left_q_target  = np.full(Dex3_Num_Motors, 0)
        right_q_target = np.full(Dex3_Num_Motors, 0)

        q = 0.0
        dq = 0.0
        tau = 0.0
        kp = 1.5
        kd = 0.2

        # initialize dex3-1's left hand cmd msg
        self.left_msg  = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Left_JointIndex:
            ris_mode = self._RIS_Mode(id = id, status = 0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.left_msg.motor_cmd[id].mode = motor_mode
            self.left_msg.motor_cmd[id].q    = q
            self.left_msg.motor_cmd[id].dq   = dq
            self.left_msg.motor_cmd[id].tau  = tau
            self.left_msg.motor_cmd[id].kp   = kp
            self.left_msg.motor_cmd[id].kd   = kd

        # initialize dex3-1's right hand cmd msg
        self.right_msg = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Right_JointIndex:
            ris_mode = self._RIS_Mode(id = id, status = 0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.right_msg.motor_cmd[id].mode = motor_mode  
            self.right_msg.motor_cmd[id].q    = q
            self.right_msg.motor_cmd[id].dq   = dq
            self.right_msg.motor_cmd[id].tau  = tau
            self.right_msg.motor_cmd[id].kp   = kp
            self.right_msg.motor_cmd[id].kd   = kd  

        try:
            while self.running:
                start_time = time.time()
                # get dual hand state form OpenXR device
                left_hand_mat  = np.array(left_hand_array[:]).reshape(25, 3).copy() # 25 joints, each with 3D position
                right_hand_mat = np.array(right_hand_array[:]).reshape(25, 3).copy() # 25 joints, each with 3D position

                # Read left and right q_state from shared arrays
                state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))

                if not np.all(right_hand_mat == 0.0) and not np.all(left_hand_mat[4] == np.array([-1.13, 0.3, 0.15])): # if hand data has been initialized.
                    # Get retargeting types
                    left_retargeting_type = self.hand_retargeting.left_retargeting.optimizer.retargeting_type
                    right_retargeting_type = self.hand_retargeting.right_retargeting.optimizer.retargeting_type
                    
                    # Process left hand
                    left_indices = self.hand_retargeting.left_retargeting.optimizer.target_link_human_indices

                    if left_retargeting_type ==  "VECTOR": # Ist das Ein bug? Ist das Absicht mit Position?? 
                        # Vector method: use absolute fingertip positions
                        ref_left_value = left_hand_mat[unitree_tip_indices]
                        # -------- Tune for Vector Retargeting Method --------
                        ref_left_value[0] = ref_left_value[0] * 1.15 # Scale thumb position
                        ref_left_value[1] = ref_left_value[1] * 1.05 # Scale index position
                        ref_left_value[2] = ref_left_value[2] * 0.95 # Scale middle position
                        left_q_target = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[self.hand_retargeting.left_dex_retargeting_to_hardware]

                    elif left_retargeting_type == "DEXPILOT":
                        # DexPilot method: use relative positions (vectors between joints)
                        # DexPilot expects 6 vectors for a 3-finger hand:
                        # Vector 0: index_tip -> thumb_tip
                        # Vector 1: middle_tip -> thumb_tip  
                        # Vector 2: middle_tip -> index_tip
                        # Vector 3: base_link -> thumb_tip
                        # Vector 4: base_link -> index_tip
                        # Vector 5: base_link -> middle_tip
                        
                        origin_indices = left_indices[0, :]  # All origin indices
                        task_indices = left_indices[1, :]    # All task indices
                        
                        # Create a sparse joint position array with OpenXR data at the correct indices
                        # DexPilot's human indices tell us where to place the OpenXR joint data

                        # Place OpenXR joint data at the indices expected by DexPilot
                        # From debug: origin_indices = [8, 12, 12, 0, 0, 0], task_indices = [4, 4, 8, 4, 8, 12]
                        # This means DexPilot expects data at indices: 0, 4, 8, 12
                        # Potential Issue area -------------------
                        # Archive
                        joint_pos = np.zeros((25, 3))
                        joint_pos[0] = left_hand_mat[0]   # wrist -> index 0
                        joint_pos[4] = left_hand_mat[4]   # thumb_tip -> index 4
                        joint_pos[8] = left_hand_mat[9]   # index_tip -> index 8 (OpenXR index 9)
                        joint_pos[12] = left_hand_mat[24] # middle_tip -> index 12 (OpenXR index 14 for middle finger) 24 for pinky finger 
                        #joint_pos = left_hand_mat.copy()  # Use the full OpenXR data directly
                        # Calculate vectors using the indices DexPilot specifies (same as show_realtime_retargeting.py)
                        ref_left_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                        # ------------------------------------
                        # From debug: URDF fixed joint indices are [1, 6] which correspond to:
                        # URDF Joint 1: left_hand_index_1_joint (Hardware Index 6)  
                        # URDF Joint 6: left_hand_thumb_2_joint (Hardware Index 2)
                        # DexPilot optimizes all 7 joints - no fixed_qpos needed
                        
                        dexpilot_output = self.hand_retargeting.left_retargeting.retarget(ref_left_value)
                        # Map from URDF order to Hardware API order
                        # DexPilot returns the full 7-joint array with fixed joints already set
                        # We just need to map from URDF order to Hardware API order
                        left_q_target = np.zeros(Dex3_Num_Motors)
                        # URDF to Hardware mapping:
                        # URDF[0] left_hand_index_0_joint -> Hardware[5]
                        # URDF[1] left_hand_index_1_joint -> Hardware[6] (fixed)
                        # URDF[2] left_hand_middle_0_joint -> Hardware[3]  
                        # URDF[3] left_hand_middle_1_joint -> Hardware[4]
                        # URDF[4] left_hand_thumb_0_joint -> Hardware[0]
                        # URDF[5] left_hand_thumb_1_joint -> Hardware[1]
                        # URDF[6] left_hand_thumb_2_joint -> Hardware[2] (fixed)
                        urdf_to_hardware = [5, 6, 3, 4, 0, 1, 2]
                        for urdf_idx, hw_idx in enumerate(urdf_to_hardware):
                            left_q_target[hw_idx] = dexpilot_output[urdf_idx]
                    else:
                        # Fallback for unknown retargeting types
                        print(f"Warning: Unknown left retargeting type {left_retargeting_type}, using zero positions")
                        left_q_target = np.zeros(Dex3_Num_Motors)
                    
                    # Process right hand
                    right_indices = self.hand_retargeting.right_retargeting.optimizer.target_link_human_indices
                    if right_retargeting_type == "VECTOR":
                        # Vector method: use absolute fingertip positions (original simple method)
                        ref_right_value = right_hand_mat[unitree_tip_indices]
                        ref_right_value[0] = ref_right_value[0] * 1.15
                        ref_right_value[1] = ref_right_value[1] * 1.05
                        ref_right_value[2] = ref_right_value[2] * 0.95
                        right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]
                    elif right_retargeting_type == "DEXPILOT":
                        # DexPilot method: use relative positions (vectors between joints)
                        # DexPilot expects 6 vectors for a 3-finger hand:
                        # Vector 0: index_tip -> thumb_tip
                        # Vector 1: middle_tip -> thumb_tip  
                        # Vector 2: middle_tip -> index_tip
                        # Vector 3: base_link -> thumb_tip
                        # Vector 4: base_link -> index_tip
                        # Vector 5: base_link -> middle_tip

                        origin_indices = right_indices[0, :]  # All origin indices
                        task_indices = right_indices[1, :]    # All task indices
                        
                        # Create a sparse joint position array with OpenXR data at the correct indices
                        # DexPilot's human indices tell us where to place the OpenXR joint data
                        joint_pos = np.zeros((25, 3))
                        
                        # Place OpenXR joint data at the indices expected by DexPilot
                        # From debug: origin_indices = [8, 12, 12, 0, 0, 0], task_indices = [4, 4, 8, 4, 8, 12]
                        # This means DexPilot expects data at indices: 0, 4, 8, 12
                        joint_pos[0] = right_hand_mat[0]   # wrist -> index 0
                        joint_pos[4] = right_hand_mat[4]   # thumb_tip -> index 4
                        joint_pos[8] = right_hand_mat[9]   # index_tip -> index 8 (OpenXR index 9)
                        joint_pos[12] = right_hand_mat[24] # middle_tip -> index 12 (OpenXR index 14)
                        
                        # Calculate vectors using the indices DexPilot specifies (same as show_realtime_retargeting.py)
                        ref_right_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                        
                        # For target_joint_names config, we need to provide fixed_qpos for the non-optimized joints
                        # From debug: URDF fixed joint indices are [1, 6] which correspond to:
                        # URDF Joint 1: right_hand_index_1_joint (Hardware Index 6)  
                        # URDF Joint 6: right_hand_thumb_2_joint (Hardware Index 2)
                        # DexPilot retargeting - same as reference implementation
                        dexpilot_output = self.hand_retargeting.right_retargeting.retarget(ref_right_value)
                        
                        # Apply thumb pinch correction to DexPilot output (optional - can be disabled)
                        # ----------------------
                        # Extract fingertip positions for pinch detection
                        # thumb_tip_pos = right_hand_mat[4]   # OpenXR index 4
                        # index_tip_pos = right_hand_mat[9]   # OpenXR index 9  
                        # middle_tip_pos = right_hand_mat[14] # OpenXR index 14
                        
                        # # Apply correction to the DexPilot output
                        # dexpilot_output = self.right_thumb_corrector.apply_correction(
                        #     dexpilot_output, thumb_tip_pos, index_tip_pos, middle_tip_pos
                        # )
                        # ------------------------------------------------
                        # Map from URDF order to Hardware API order
                        # DexPilot returns the full 7-joint array with fixed joints already set
                        right_q_target = np.zeros(Dex3_Num_Motors)

                        # URDF to Hardware mapping:
                        # URDF[0] right_hand_index_0_joint -> Hardware[5]
                        # URDF[1] right_hand_index_1_joint -> Hardware[6] (fixed)
                        # URDF[2] right_hand_middle_0_joint -> Hardware[3]  
                        # URDF[3] right_hand_middle_1_joint -> Hardware[4]
                        # URDF[4] right_hand_thumb_0_joint -> Hardware[0]
                        # URDF[5] right_hand_thumb_1_joint -> Hardware[1]
                        # URDF[6] right_hand_thumb_2_joint -> Hardware[2] (fixed)
                        urdf_to_hardware = [5, 6, 3, 4, 0, 1, 2]
                        for urdf_idx, hw_idx in enumerate(urdf_to_hardware):
                            right_q_target[hw_idx] = dexpilot_output[urdf_idx]
                    else:
                        # Fallback for unknown retargeting types
                        print(f"Warning: Unknown right retargeting type {right_retargeting_type}, using zero positions")
                        right_q_target = np.zeros(Dex3_Num_Motors)

                # get dual hand action
                action_data = np.concatenate((left_q_target, right_q_target))    
                if dual_hand_state_array and dual_hand_action_array:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data

                self.ctrl_dual_hand(left_q_target, right_q_target)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            print("Dex3_1_Controller has been closed.")
class Dex3_1_Controller_console:
    def __init__(self, left_hand_value_in, right_hand_value_in, left_aButton_in,left_bButton_in,
                 right_aButton_in,right_bButton_in,dual_hand_data_lock = None, dual_hand_state_array_out = None,
                       dual_hand_action_array_out = None, fps = 100.0, Unit_Test = False, simulation_mode = False):
        """
        [note] A *_array type parameter requires using a multiprocessing Array, because it needs to be passed to the internal child process

        left_hand_array_in: [input] Left hand skeleton data (required from XR device) to hand_ctrl.control_process

        right_hand_array_in: [input] Right hand skeleton data (required from XR device) to hand_ctrl.control_process

        dual_hand_data_lock: Data synchronization lock for dual_hand_state_array and dual_hand_action_array

        dual_hand_state_array_out: [output] Return left(7), right(7) hand motor state

        dual_hand_action_array_out: [output] Return left(7), right(7) hand motor action

        fps: Control frequency

        Unit_Test: Whether to enable unit testing

        simulation_mode: Whether to use simulation mode (default is False, which means using real robot)
        """
        logger_mp.info("Initialize Dex3_1_Controller_console")

        self.fps = fps
        self.Unit_Test = Unit_Test  #用来验证算法正确性，controller不用
        self.simulation_mode = simulation_mode
        # if not self.Unit_Test:
        #     self.hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3)
        # else:
        #     self.hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3_Unit_Test)

        # may not use
        if filter and not self.simulation_mode:
            self.smooth_filter = WeightedMovingFilter(np.array([0.5, 0.3, 0.2]), 2)
        else:
            self.smooth_filter = None

        if self.simulation_mode:
            ChannelFactoryInitialize(1)
        else:
            ChannelFactoryInitialize(0)

        # initialize handcmd publisher and handstate subscriber
        self.LeftHandCmb_publisher = ChannelPublisher(kTopicDex3LeftCommand, HandCmd_)
        self.LeftHandCmb_publisher.Init()
        self.RightHandCmb_publisher = ChannelPublisher(kTopicDex3RightCommand, HandCmd_)
        self.RightHandCmb_publisher.Init()

        self.LeftHandState_subscriber = ChannelSubscriber(kTopicDex3LeftState, HandState_)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(kTopicDex3RightState, HandState_)
        self.RightHandState_subscriber.Init()

        # Shared Arrays for hand states
        self.left_hand_state_array  = Array('d', Dex3_Num_Motors, lock=True)  #硬件写入
        self.right_hand_state_array = Array('d', Dex3_Num_Motors, lock=True)

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()
        # waiting for get data
        while True:
            if any(self.left_hand_state_array) and any(self.right_hand_state_array):
                break
            time.sleep(0.01)
            logger_mp.warning("[Dex3_1_Controller] Waiting to subscribe dds...")
        logger_mp.info("[Dex3_1_Controller] Subscribe dds ok.")

        hand_control_process = Process(target=self.control_process, args=(left_hand_value_in, right_hand_value_in, left_aButton_in,left_bButton_in,
                                                                          right_aButton_in,right_bButton_in, self.left_hand_state_array, self.right_hand_state_array,
                                                                          dual_hand_data_lock, dual_hand_state_array_out, dual_hand_action_array_out))
        hand_control_process.daemon = True
        hand_control_process.start()

        logger_mp.info("Initialize Dex3_1_Controller OK!\n")

    def _subscribe_hand_state(self):
        while True:
            left_hand_msg  = self.LeftHandState_subscriber.Read()
            right_hand_msg = self.RightHandState_subscriber.Read()
            if left_hand_msg is not None and right_hand_msg is not None:
                # Update left hand state
                for idx, id in enumerate(Dex3_1_Left_JointIndex):
                    self.left_hand_state_array[idx] = left_hand_msg.motor_state[id].q
                # Update right hand state
                for idx, id in enumerate(Dex3_1_Right_JointIndex):
                    self.right_hand_state_array[idx] = right_hand_msg.motor_state[id].q
            time.sleep(0.002)
    
    class _RIS_Mode:
        def __init__(self, id=0, status=0x01, timeout=0):
            self.motor_mode = 0
            self.id = id & 0x0F  # 4 bits for id
            self.status = status & 0x07  # 3 bits for status
            self.timeout = timeout & 0x01  # 1 bit for timeout

        def _mode_to_uint8(self):
            self.motor_mode |= (self.id & 0x0F)
            self.motor_mode |= (self.status & 0x07) << 4
            self.motor_mode |= (self.timeout & 0x01) << 7
            return self.motor_mode

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """set current left, right hand motor state target q"""
        for idx, id in enumerate(Dex3_1_Left_JointIndex):
            self.left_msg.motor_cmd[id].q = left_q_target[idx]
        for idx, id in enumerate(Dex3_1_Right_JointIndex):
            self.right_msg.motor_cmd[id].q = right_q_target[idx]

        self.LeftHandCmb_publisher.Write(self.left_msg)
        self.RightHandCmb_publisher.Write(self.right_msg)
        # logger_mp.debug("hand ctrl publish ok.")
    
    def control_process(self, left_hand_value_in, right_hand_value_in, 
                        left_aButton_in,left_bButton_in,right_aButton_in,right_bButton_in,left_hand_state_array, right_hand_state_array,
                              dual_hand_data_lock = None, dual_hand_state_array_out = None, dual_hand_action_array_out = None):
        self.running = True
        #（7,0）
        left_q_target  = np.full(Dex3_Num_Motors, 0)
        right_q_target = np.full(Dex3_Num_Motors, 0)
        # 大拇指012：0°~+100°, -35°~+60°, -60°~+60°；
        # 食指34、中指56：0°~+90°，0°~+100°
        # MAX_LIMITS_LEFT  = [-1.05, -0.742, 0.0,   0.0,   0.0,   0.0,   0.0]
        # MIN_LIMITS_LEFT  = [1.05,  1.05,  1.75,   -1.57, -1.75, -1.57, -1.75]

        # MAX_LIMITS_RIGHT = [1.05,  0.742,  0.0,   0.0,   0.0,   0.0,   0.0] 
        # MIN_LIMITS_RIGHT = [-1.05, -1.05, -1.75,  1.57,  1.75,  1.57,  1.75]  

        MAX_LIMITS_LEFT  = [-0.05, -0.742, 0.0,   0.0,   0.0,   0.0,   0.0]
        MIN_LIMITS_LEFT  = [0.05,  1.05,  1.75,   -1.57, -1.75, -1.57, -1.75]

        MAX_LIMITS_RIGHT = [0.05,  0.742,  0.0,   0.0,   0.0,   0.0,   0.0] 
        MIN_LIMITS_RIGHT = [-0.05, -1.05, -1.75,  1.57,  1.75,  1.57,  1.75]  

        mid_limits_left  = [(max_v + min_v) / 2.0 for max_v, min_v in zip(MAX_LIMITS_LEFT,  MIN_LIMITS_LEFT)]
        mid_limits_right = [(max_v + min_v) / 2.0 for max_v, min_v in zip(MAX_LIMITS_RIGHT, MIN_LIMITS_RIGHT)]

        # === mid value ===
        left_target_action  = np.array(mid_limits_left)
        right_target_action = np.array(mid_limits_right)
        DELTA_GRIPPER_CMD = 0.18  
        q = 0.0
        dq = 0.0
        tau = 0.0
        kp = 1.5
        kd = 0.2
        # 相当于封装了motor_cmd
        # initialize dex3-1's left hand cmd msg
        self.left_msg  = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Left_JointIndex:
            ris_mode = self._RIS_Mode(id = id, status = 0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.left_msg.motor_cmd[id].mode = motor_mode
            self.left_msg.motor_cmd[id].q    = q
            self.left_msg.motor_cmd[id].dq   = dq
            self.left_msg.motor_cmd[id].tau  = tau
            self.left_msg.motor_cmd[id].kp   = kp
            self.left_msg.motor_cmd[id].kd   = kd

        # initialize dex3-1's right hand cmd msg
        self.right_msg = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Right_JointIndex:
            ris_mode = self._RIS_Mode(id = id, status = 0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.right_msg.motor_cmd[id].mode = motor_mode  
            self.right_msg.motor_cmd[id].q    = q
            self.right_msg.motor_cmd[id].dq   = dq
            self.right_msg.motor_cmd[id].tau  = tau
            self.right_msg.motor_cmd[id].kp   = kp
            self.right_msg.motor_cmd[id].kd   = kd  
        """
        const float maxLimits_left[7]=  {  1.05 ,  1.05  , 1.75 ,   0   ,  0    , 0     , 0   }; // set max motor value
        const float minLimits_left[7]=  { -1.05 , -0.724 ,   0  , -1.57 , -1.75 , -1.57  ,-1.75}; 
        const float maxLimits_right[7]= {  1.05 , 0.742  ,   0  ,  1.57 , 1.75  , 1.57  , 1.75}; 
        const float minLimits_right[7]= { -1.05 , -1.05  , -1.75,    0  ,  0    ,   0   ,0    }; 
        void rotateMotors(bool isLeftHand) {
        static int _count = 1; // 用来计数让手动起来
        static int dir = 1;    // 控制抓握方向
        const float* maxTorqueLimits = isLeftHand ? maxTorqueLimits_left : maxTorqueLimits_right;
        const float* minTorqueLimits = isLeftHand ? minTorqueLimits_left : minTorqueLimits_right;

        for (int i = 0; i < MOTOR_MAX; i++) {
            RIS_Mode_t ris_mode;
            ris_mode.id = i;        // 设置 id
            ris_mode.status = 0x01; // 设置 status 为 0x01
            ris_mode.timeout = 0x01; // 设置 timeout 为 0x01
            
            uint8_t mode = 0;
            mode |= (ris_mode.id & 0x0F);             // 取低 4 位 id
            mode |= (ris_mode.status & 0x07) << 4;    // 取高 3 位 status 并左移 4 位
            mode |= (ris_mode.timeout & 0x01) << 7;   // 取高 1 位 timeout 并左移 7 位
            msg.motor_cmd()[i].mode(mode);
            msg.motor_cmd()[i].tau(0);
            msg.motor_cmd()[i].kp(0.5);      // 设置控制增益 kp
            msg.motor_cmd()[i].kd(0.1);    // 设置控制增益 kd

            // 计算目标位置 q
            float range = maxTorqueLimits[i] - minTorqueLimits[i]; // 限位范围
            float mid = (maxTorqueLimits[i] + minTorqueLimits[i]) / 2.0; // 中间值
            float amplitude = range / 2.0; // 振幅

            // 使用 _count 动态调整 q 值
            float q = mid + amplitude * sin(_count / 20000.0 * M_PI); // 生成一个随时间变化的正弦波

            // if(i == 0)std::cout << q << std::endl;
            msg.motor_cmd()[i].q(q); // 设置目标位置 q
        }

        handcmd_publisher->Write(msg);
        _count += dir;

        // 控制抓握方向
        if (_count >= 10000) {
            dir = -1;
        }
        if (_count <= -10000) {
            dir = 1;
        }

        usleep(100); // 控制循环频率，避免过快发送命令
    }
        """
        try:
            while self.running:
                start_time = time.time()
                # get dual hand skeletal point state from XR device
                with left_hand_value_in.get_lock():
                    left_hand_value  = left_hand_value_in.value
                with right_hand_value_in.get_lock():
                    right_hand_value = right_hand_value_in.value
                    
                with left_aButton_in.get_lock():
                    left_a_pressed = left_aButton_in.value

                with left_bButton_in.get_lock():
                    left_b_pressed = left_bButton_in.value

                with right_aButton_in.get_lock():
                    right_a_pressed = right_aButton_in.value

                with right_bButton_in.get_lock():
                    right_b_pressed = right_bButton_in.value

                # with left_hand_state_array.get_lock():
                left_state = np.array(left_hand_state_array[:])  # 自动按元素长度转换
                # with right_hand_state_array.get_lock():
                right_state = np.array(right_hand_state_array[:])

                state_data = np.concatenate((left_state, right_state))


                if left_hand_value != 0.0 or right_hand_value != 0.0: # if input data has been initialized.
                    for i in range(7):
                        left_target_action[i]  = np.interp(left_hand_value,  [0, 10], [MIN_LIMITS_LEFT[i],  MAX_LIMITS_LEFT[i]])
                        right_target_action[i] = np.interp(right_hand_value, [0, 10], [MIN_LIMITS_RIGHT[i], MAX_LIMITS_RIGHT[i]])
                    
                    if left_a_pressed:
                        left_target_action[0] = -0.75
                        left_target_action[2] = 1.05
                        left_target_action[3] = 0.0
                        left_target_action[4] = 0.0
                    if left_b_pressed:
                        left_target_action[0] = 0.95
                        left_target_action[2] = 1.05
                        left_target_action[5] = 0.0
                        left_target_action[6] = 0.0

                    if right_a_pressed:
                        right_target_action[0] = -0.75
                        right_target_action[2] = -1.05
                        right_target_action[3] = 0.0
                        right_target_action[4] = 0.0
                    if right_b_pressed:
                        right_target_action[0] = 0.95
                        right_target_action[2] = -1.05
                        right_target_action[5] = 0.0
                        right_target_action[6] = 0.0
                    # clip dual gripper action to avoid overflow
                if not self.simulation_mode:
                    left_q_target  = np.clip(left_target_action,  left_state - DELTA_GRIPPER_CMD, left_state + DELTA_GRIPPER_CMD) 
                    right_q_target = np.clip(right_target_action, right_state - DELTA_GRIPPER_CMD, right_state + DELTA_GRIPPER_CMD)
                else:
                    left_q_target  = left_target_action
                    right_q_target = right_target_action
                action_data = np.array((left_q_target, right_q_target))
                
                #确定如何使用
                # if self.smooth_filter:
                #     self.smooth_filter.add_data(dual_gripper_action)
                #     dual_gripper_action = self.smooth_filter.filtered_data

                #caculate the q
                # get dual hand action
                action_data = np.concatenate((left_q_target, right_q_target))    #（7,），（7,）
                if dual_hand_state_array_out and dual_hand_action_array_out:
                    with dual_hand_data_lock:
                        dual_hand_state_array_out[:] = state_data
                        dual_hand_action_array_out[:] = action_data
                #hand


                self.ctrl_dual_hand(left_q_target, right_q_target)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Dex3_1_Controller has been closed.")



class Dex3_1_Left_JointIndex(IntEnum):
    kLeftHandThumb0 = 0
    kLeftHandThumb1 = 1
    kLeftHandThumb2 = 2
    kLeftHandMiddle0 = 3
    kLeftHandMiddle1 = 4
    kLeftHandIndex0 = 5
    kLeftHandIndex1 = 6

class Dex3_1_Right_JointIndex(IntEnum):
    kRightHandThumb0 = 0
    kRightHandThumb1 = 1
    kRightHandThumb2 = 2
    kRightHandIndex0 = 3
    kRightHandIndex1 = 4
    kRightHandMiddle0 = 5
    kRightHandMiddle1 = 6


# unitree_gripper_indices = [4, 9] # [thumb, index]
# Gripper_Num_Motors = 2
# kTopicGripperCommand = "rt/unitree_actuator/cmd"
# kTopicGripperState = "rt/unitree_actuator/state"


kTopicGripperLeftCommand = "rt/dex1/left/cmd"
kTopicGripperLeftState = "rt/dex1/left/state"
kTopicGripperRightCommand = "rt/dex1/right/cmd"
kTopicGripperRightState = "rt/dex1/right/state"

class Dex1_1_Gripper_Controller:
    def __init__(self, left_gripper_value_in, right_gripper_value_in, dual_gripper_data_lock = None, dual_gripper_state_out = None, dual_gripper_action_out = None, 
                       filter = True, fps = 200.0, Unit_Test = False, simulation_mode = False):
        """
        [note] A *_array type parameter requires using a multiprocessing Array, because it needs to be passed to the internal child process

        left_gripper_value_in: [input] Left ctrl data (required from XR device) to control_thread

        right_gripper_value_in: [input] Right ctrl data (required from XR device) to control_thread

        dual_gripper_data_lock: Data synchronization lock for dual_gripper_state_array and dual_gripper_action_array

        dual_gripper_state_out: [output] Return left(1), right(1) gripper motor state

        dual_gripper_action_out: [output] Return left(1), right(1) gripper motor action

        fps: Control frequency

        Unit_Test: Whether to enable unit testing

        simulation_mode: Whether to use simulation mode (default is False, which means using real robot)
        """

        logger_mp.info("Initialize Dex1_1_Gripper_Controller...")

        self.fps = fps
        self.Unit_Test = Unit_Test
        self.gripper_sub_ready = False
        self.simulation_mode = simulation_mode
        
        if filter and not self.simulation_mode:
            self.smooth_filter = WeightedMovingFilter(np.array([0.5, 0.3, 0.2]), 2)
        else:
            self.smooth_filter = None

        if self.simulation_mode:
            ChannelFactoryInitialize(1)
        else:
            ChannelFactoryInitialize(0)
 
        # initialize handcmd publisher and handstate subscriber
        self.LeftGripperCmb_publisher = ChannelPublisher(kTopicGripperLeftCommand, MotorCmds_)
        self.LeftGripperCmb_publisher.Init()
        self.RightGripperCmb_publisher = ChannelPublisher(kTopicGripperRightCommand, MotorCmds_)
        self.RightGripperCmb_publisher.Init()

        self.LeftGripperState_subscriber = ChannelSubscriber(kTopicGripperLeftState, MotorStates_)
        self.LeftGripperState_subscriber.Init()
        self.RightGripperState_subscriber = ChannelSubscriber(kTopicGripperRightState, MotorStates_)
        self.RightGripperState_subscriber.Init()

        # Shared Arrays for gripper states
        self.left_gripper_state_value = Value('d', 0.0, lock=True)
        self.right_gripper_state_value = Value('d', 0.0, lock=True)

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_gripper_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while not self.gripper_sub_ready:
            time.sleep(0.01)
            logger_mp.warning("[Dex1_1_Gripper_Controller] Waiting to subscribe dds...")
        logger_mp.info("[Dex1_1_Gripper_Controller] Subscribe dds ok.")

        self.gripper_control_thread = threading.Thread(target=self.control_thread, args=(left_gripper_value_in, right_gripper_value_in, self.left_gripper_state_value, self.right_gripper_state_value,
                                                                                         dual_gripper_data_lock, dual_gripper_state_out, dual_gripper_action_out))
        self.gripper_control_thread.daemon = True
        self.gripper_control_thread.start()

        logger_mp.info("Initialize Dex1_1_Gripper_Controller OK!\n")

    def _subscribe_gripper_state(self):
        while True:
            left_gripper_msg  = self.LeftGripperState_subscriber.Read()
            right_gripper_msg  = self.RightGripperState_subscriber.Read()
            self.gripper_sub_ready = True
            if left_gripper_msg is not None and right_gripper_msg is not None:
                self.left_gripper_state_value.value = left_gripper_msg.states[0].q
                self.right_gripper_state_value.value = right_gripper_msg.states[0].q
            time.sleep(0.002)
    
    def ctrl_dual_gripper(self, dual_gripper_action):
        """set current left, right gripper motor cmd target q"""
        self.left_gripper_msg.cmds[0].q  = dual_gripper_action[0]
        self.right_gripper_msg.cmds[0].q = dual_gripper_action[1]

        self.LeftGripperCmb_publisher.Write(self.left_gripper_msg)
        self.RightGripperCmb_publisher.Write(self.right_gripper_msg)
        # logger_mp.debug("gripper ctrl publish ok.")
    
    def control_thread(self, left_gripper_value_in, right_gripper_value_in, left_gripper_state_value, right_gripper_state_value, dual_hand_data_lock = None, 
                             dual_gripper_state_out = None, dual_gripper_action_out = None):
        self.running = True
        DELTA_GRIPPER_CMD = 0.18     # The motor rotates 5.4 radians, the clamping jaw slide open 9 cm, so 0.6 rad <==> 1 cm, 0.18 rad <==> 3 mm
        THUMB_INDEX_DISTANCE_MIN = 5.0
        THUMB_INDEX_DISTANCE_MAX = 7.0
        LEFT_MAPPED_MIN  = 0.0           # The minimum initial motor position when the gripper closes at startup.
        RIGHT_MAPPED_MIN = 0.0           # The minimum initial motor position when the gripper closes at startup.
        # The maximum initial motor position when the gripper closes before calibration (with the rail stroke calculated as 0.6 cm/rad * 9 rad = 5.4 cm).
        LEFT_MAPPED_MAX = LEFT_MAPPED_MIN + 5.40 
        RIGHT_MAPPED_MAX = RIGHT_MAPPED_MIN + 5.40
        left_target_action  = (LEFT_MAPPED_MAX - LEFT_MAPPED_MIN) / 2.0
        right_target_action = (RIGHT_MAPPED_MAX - RIGHT_MAPPED_MIN) / 2.0

        dq = 0.0
        tau = 0.0
        kp = 5.00
        kd = 0.05
        # initialize gripper cmd msg
        self.left_gripper_msg  = MotorCmds_()
        self.left_gripper_msg.cmds = [unitree_go_msg_dds__MotorCmd_()]
        self.right_gripper_msg = MotorCmds_()
        self.right_gripper_msg.cmds = [unitree_go_msg_dds__MotorCmd_()]

        self.left_gripper_msg.cmds[0].dq  = dq
        self.left_gripper_msg.cmds[0].tau = tau
        self.left_gripper_msg.cmds[0].kp  = kp
        self.left_gripper_msg.cmds[0].kd  = kd

        self.right_gripper_msg.cmds[0].dq  = dq
        self.right_gripper_msg.cmds[0].tau = tau
        self.right_gripper_msg.cmds[0].kp  = kp
        self.right_gripper_msg.cmds[0].kd  = kd
        try:
            while self.running:
                start_time = time.time()
                # get dual hand skeletal point state from XR device
                with left_gripper_value_in.get_lock():
                    left_gripper_value  = left_gripper_value_in.value
                with right_gripper_value_in.get_lock():
                    right_gripper_value = right_gripper_value_in.value

                if left_gripper_value != 0.0 or right_gripper_value != 0.0: # if input data has been initialized.
                    # Linear mapping from [0, THUMB_INDEX_DISTANCE_MAX] to gripper action range
                    left_target_action  = np.interp(left_gripper_value, [THUMB_INDEX_DISTANCE_MIN, THUMB_INDEX_DISTANCE_MAX], [LEFT_MAPPED_MIN, LEFT_MAPPED_MAX])
                    right_target_action = np.interp(right_gripper_value, [THUMB_INDEX_DISTANCE_MIN, THUMB_INDEX_DISTANCE_MAX], [RIGHT_MAPPED_MIN, RIGHT_MAPPED_MAX])

                # get current dual gripper motor state
                dual_gripper_state = np.array([left_gripper_state_value.value, right_gripper_state_value.value])

                # clip dual gripper action to avoid overflow
                left_actual_action  = np.clip(left_target_action,  dual_gripper_state[0] - DELTA_GRIPPER_CMD, dual_gripper_state[0] + DELTA_GRIPPER_CMD) 
                right_actual_action = np.clip(right_target_action, dual_gripper_state[1] - DELTA_GRIPPER_CMD, dual_gripper_state[1] + DELTA_GRIPPER_CMD)

                dual_gripper_action = np.array([left_actual_action, right_actual_action])

                if self.smooth_filter:
                    self.smooth_filter.add_data(dual_gripper_action)
                    dual_gripper_action = self.smooth_filter.filtered_data

                if dual_gripper_state_out and dual_gripper_action_out:
                    with dual_hand_data_lock:
                        dual_gripper_state_out[:] = dual_gripper_state - np.array([LEFT_MAPPED_MIN, RIGHT_MAPPED_MIN])
                        dual_gripper_action_out[:] = dual_gripper_action - np.array([LEFT_MAPPED_MIN, RIGHT_MAPPED_MIN])

                self.ctrl_dual_gripper(dual_gripper_action)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Dex1_1_Gripper_Controller has been closed.")

class Gripper_JointIndex(IntEnum):
    kGripper = 0


if __name__ == "__main__":
    import argparse
    from televuer import TeleVuerWrapper
    from teleop.image_server.image_client import ImageClient

    parser = argparse.ArgumentParser()
    parser.add_argument('--xr-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device tracking source')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire1', 'brainco'], help='Select end effector controller')
    args = parser.parse_args()
    logger_mp.info(f"args:{args}\n")

    # image
    img_config = {
        'fps': 30,
        'head_camera_type': 'opencv',
        'head_camera_image_shape': [480, 1280],  # Head camera resolution
        'head_camera_id_numbers': [0],
    }
    ASPECT_RATIO_THRESHOLD = 2.0  # If the aspect ratio exceeds this value, it is considered binocular
    if len(img_config['head_camera_id_numbers']) > 1 or (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        BINOCULAR = True
    else:
        BINOCULAR = False
    # image
    if BINOCULAR and not (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1] * 2, 3)
    else:
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)

    tv_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(tv_img_shape) * np.uint8().itemsize)
    tv_img_array = np.ndarray(tv_img_shape, dtype = np.uint8, buffer = tv_img_shm.buf)
    img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name)
    image_receive_thread = threading.Thread(target = img_client.receive_process, daemon = True)
    image_receive_thread.daemon = True
    image_receive_thread.start()

    # television: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
    tv_wrapper = TeleVuerWrapper(binocular=BINOCULAR, use_hand_tracking=args.xr_mode == "hand", img_shape=tv_img_shape, img_shm_name=tv_img_shm.name, 
                                 return_state_data=True, return_hand_rot_data = False)

# end-effector
    if args.ee == "dex3"and args.xr_mode == "hand":
        left_hand_pos_array = Array('d', 75, lock = True)      # [input]
        right_hand_pos_array = Array('d', 75, lock = True)     # [input]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
        dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
        hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array)
    
    elif args.ee == "dex3" and args.xr_mode == "controller":
        left_hand_value_in = Value('d', 0.0, lock=True)      # [input]
        right_hand_value_in = Value('d', 0.0, lock=True)     # [input]
        left_aButton_in  = Value('b', False, lock=True)         
        left_bButton_in = Value('b', False, lock=True)
        right_aButton_in = Value('b', False, lock=True)
        right_bButton_in = Value('b', False, lock=True)
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
        dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
        hand_ctrl = Dex3_1_Controller_console(left_hand_value_in, right_hand_value_in, left_aButton_in,left_bButton_in,right_aButton_in,right_bButton_in,
                                            dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array)
      
    elif args.ee == "dex1":
        left_gripper_value = Value('d', 0.0, lock=True)        # [input]
        right_gripper_value = Value('d', 0.0, lock=True)       # [input]
        dual_gripper_data_lock = Lock()
        dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
        dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
        gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, dual_gripper_state_array, dual_gripper_action_array)

    user_input = input("Please enter the start signal (enter 's' to start the subsequent program):\n")
    if user_input.lower() == 's':
        while True:
            tele_data = tv_wrapper.get_motion_state_data()
            if args.ee == "dex3" and args.xr_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()

            elif args.ee == "dex3" and args.xr_mode == "controller":
                with left_hand_value_in.get_lock():
                    left_hand_value_in.value = tele_data.left_trigger_value
                with right_hand_value_in.get_lock():
                    right_hand_value_in.value = tele_data.right_trigger_value
                with left_aButton_in.get_lock():
                    left_aButton_in.value = tele_data.tele_state.left_aButton  # True / False
                with left_bButton_in.get_lock():
                    left_bButton_in.value = tele_data.tele_state.left_bButton
                with right_aButton_in.get_lock():
                    right_aButton_in.value = tele_data.tele_state.right_aButton
                with right_bButton_in.get_lock():
                    right_bButton_in.value = tele_data.tele_state.right_bButton
                    

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

            # with dual_hand_data_lock:
            #     logger_mp.info(f"state : {list(dual_hand_state_array)} \naction: {list(dual_hand_action_array)} \n")
            time.sleep(0.01)