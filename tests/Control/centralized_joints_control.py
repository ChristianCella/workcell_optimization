#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import sys
from collections import deque
from scipy.spatial.transform import Rotation as R

''' 
Inverse-dynamics (computed-torque) control on UR5e with optional I/O delays.
- Control law:
    qacc_cmd = qdd_des + Kd*(qd_des - qd_meas) + Kp*(q_des - q_meas)
- Torques are computed via mj_inverse at the current plant state.
- Optional sensor (measurement) delay and/or actuation (command) delay are modeled
  by FIFO buffers in discrete time.
'''

utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils'))
sys.path.append(utils_dir)
import fonts
from transformations import rotm_to_quaternion, get_world_wrench, get_homogeneous_matrix, euler_to_quaternion
from mujoco_utils import set_body_pose, get_collisions, inverse_manipulability, compute_jacobian

# ---- Gains 
Kp = np.array([15, 15, 15, 15, 15, 15], dtype=float)
Kd = np.array([8, 8, 8, 8, 8, 8], dtype=float)

# ---- Delay settings (edit these) ----
DELAY_MS = 0.0 # = or 130 are good tests
SENSING_DELAY = True         # controller uses delayed measurements (q, qd)
ACTUATION_DELAY = False      # apply a delayed torque (command delay)
# -------------------------------------

def euler_to_quaternion(roll, pitch, yaw, degrees=False):
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
    q = r.as_quat()
    return [q[3], q[0], q[1], q[2]]  # [w, x, y, z]

def main():

    # Path setup
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    sys.path.append(base_dir)
    xml_path = os.path.join(base_dir, "ur5e_utils_mujoco/ur5e/ur5e.xml")

    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)

        # --- Discrete delay buffers (computed from sim timestep) ---
        sim_dt = model.opt.timestep
        delay_steps = max(1, int(round((DELAY_MS / 1000.0) / sim_dt)))

        # Buffers for delayed sensing (q, qd) and delayed actuation (tau)
        state_fifo = deque(maxlen=delay_steps + 1)
        torque_fifo = deque(maxlen=delay_steps + 1)

        # List of target joint positions (radians)
        target_qpos_list = [
            np.radians([100, -94.96, 101.82, -95.72, -96.35, -40.97]),
        ]

        # Define body IDs
        base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")
        tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")

        with mujoco.viewer.launch_passive(model, data) as viewer:

            print(f"{fonts.cyan}Initial configuration before tracking{fonts.reset}")
            # (Optional) move base or tool if you want
            _, _, A_w_b = get_homogeneous_matrix(0, 0, 0.0, 0, 0, 0)
            set_body_pose(model, data, base_body_id, A_w_b[:3, 3], rotm_to_quaternion(A_w_b[:3, :3]))
            _, _, A_ee_t = get_homogeneous_matrix(0, 0, 0.0, 0, 0, 0)
            set_body_pose(model, data, tool_body_id, A_ee_t[:3, 3], rotm_to_quaternion(A_ee_t[:3, :3]))

            # Set the initial configuration
            data.qpos[:6] = np.radians([0, -94.96, 101.82, -95.72, -96.35, -40.97])
            data.qvel[:6] = 0.0
            mujoco.mj_forward(model, data)
            data.qacc[:] = 0.0
            viewer.sync()

            # Prime the FIFOs with initial state and zero torque
            q0 = data.qpos[:6].copy()
            qd0 = data.qvel[:6].copy()
            for _ in range(delay_steps + 1):
                state_fifo.append((q0.copy(), qd0.copy()))
                torque_fifo.append(np.zeros(6))

            input("Press Enter to start the robot configuration...")

            for i, desired_qpos in enumerate(target_qpos_list):
                print(f"\n==== Moving to target configuration {i+1} ====")

                last_time = time.perf_counter()
                seconds_per_config = 20.0 
                start_time = last_time

                while time.perf_counter() - start_time < seconds_per_config:
                    mujoco.mj_forward(model, data)

                    current_time = time.perf_counter()
                    dt = current_time - last_time
                    last_time = current_time
                    if dt <= 0.0:
                        dt = sim_dt

                    # ---- Inverse-dynamics control (COMPUTED TORQUE) ----
                    # Desired trajectory for a step: qd_des = 0, qdd_des = 0
                    q_des  = desired_qpos
                    qd_des = np.zeros(6)
                    qdd_des = np.zeros(6)

                    # Push *current* measured plant state to FIFO
                    state_fifo.append((data.qpos[:6].copy(), data.qvel[:6].copy()))

                    # Controller sees either delayed or current measurements
                    if SENSING_DELAY:
                        q_meas, qd_meas = state_fifo[0]  # ~ Td old
                    else:
                        q_meas, qd_meas = data.qpos[:6], data.qvel[:6]

                    # Errors based on (possibly delayed) measurements
                    e  = q_des - q_meas
                    ed = qd_des - qd_meas

                    # (1) Build desired accelerations
                    qacc_cmd = qdd_des + Kd * ed + Kp * e

                    # (2) Ask MuJoCo what torque realizes qacc_cmd at the CURRENT plant state
                    data.qacc[:] = 0.0
                    data.qacc[:6] = qacc_cmd
                    mujoco.mj_inverse(model, data)
                    tau_now = data.qfrc_inverse[:6].copy()

                    # Add to torque FIFO and select what to apply
                    torque_fifo.append(tau_now)
                    if ACTUATION_DELAY:
                        tau_to_apply = torque_fifo[0]   # delayed command
                    else:
                        tau_to_apply = tau_now

                    # (4) Apply torques via actuators
                    data.ctrl[:6] = tau_to_apply

                    # Step the simulation
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(sim_dt)

                    # Display the torques applied by actuators
                    print(f"{time.perf_counter()-start_time:.2f}\t{np.round(data.qfrc_actuator[:6], 2)}")

                print(f"{fonts.red}Final joint angles (deg): {np.round(np.degrees(data.qpos[:6]), 2)}{fonts.reset}")
                print("======================================")

            print("\n--- Finished all configurations. ---")
            input("Press Enter to close the viewer and exit...")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
