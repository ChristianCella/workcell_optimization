#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import sys
from scipy.spatial.transform import Rotation as R

''' 
Test to see if a simple PID controller can stabilize the rboot around a given configuration.
- In this case, you are not computing torques, you are applying a PID controller to compensate for the
    motion that arises due to gravity and an external wrench.
- Torques are read from the joints directly: if the PID works as intended, the values you read should be
    close to the torques that you would compute with a static model (test_wrench.py).
    NOTE: impose the same configuration as in test_wrench.py, otherwise the torques will be different!
'''

# PID gains
Kp = np.array([700, 700, 700, 700, 700, 700])
Kd = np.array([70, 70, 70, 70, 70, 70])
Ki = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

def euler_to_quaternion(roll, pitch, yaw, degrees=False):
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
    q = r.as_quat()
    return [q[3], q[0], q[1], q[2]]  # [w, x, y, z]

def main():

    # Path setup
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(base_dir)
    xml_path = os.path.join(base_dir, "universal_robots_ur5e/scene.xml")

    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)

        # List of target joint positions
        target_qpos_list = [
            np.radians([100, -94.96, 101.82, -95.72, -96.35, -40.97]),
            np.radians([15.99, -62.07, 121.58, -159.82, -99, -90]),
            np.radians([-28.83, -84.36, 102.8, 18.69, -98.83, -101.46]),
            np.radians([-29.77, -160.55, 110.57, -186.16, -98.81, -101.15]),
            np.radians([-124.35, -158.41, 147, -153.21, -155.31, -98.24])
        ]

        # Define body and site IDs
        base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")

        # External wrench (edit as you like)
        external_force_world = np.array([0.0, 5.0, 0.0])
        external_torque_world = np.array([0.0, 0.0, 0.0])

        with mujoco.viewer.launch_passive(model, data) as viewer:
            input("Press Enter to start the robot configuration...")

            for i, desired_qpos in enumerate(target_qpos_list):
                print(f"\n==== Moving to target configuration {i+1} ====")

                # (Optional) move base or tool if you want
                model.body_pos[base_body_id] = [i*0.2, 0.0, 0.5]
                model.body_quat[base_body_id] = euler_to_quaternion(45, 45, 0, degrees=True)
                model.body_pos[tool_body_id] = [0.1, 0.1, 0.1]
                model.body_quat[tool_body_id] = euler_to_quaternion(0, 0, 0, degrees=True)

                # INSTANTANEOUSLY set new pose (reset velocities!)
                data.qpos[:6] = desired_qpos.copy()
                data.qvel[:6] = 0.0
                mujoco.mj_forward(model, data)
                data.qacc[:] = 0.0

                error_integral = np.zeros(6)
                last_time = time.perf_counter()
                seconds_per_config = 10.0  # Hold each pose for 5 seconds
                start_time = last_time

                print("Time\tTorques (Nm)")
                while time.perf_counter() - start_time < seconds_per_config:
                    mujoco.mj_forward(model, data)

                    current_time = time.perf_counter()
                    dt = current_time - last_time
                    last_time = current_time

                    # Apply the external wrench
                    data.xfrc_applied[tool_body_id, :3] = -external_force_world
                    data.xfrc_applied[tool_body_id, 3:] = -external_torque_world

                    # PID Controller
                    q_error = desired_qpos - data.qpos[:6]
                    qd_error = -data.qvel[:6]
                    error_integral += q_error * dt
                    control_torque = Kp * q_error + Ki * error_integral + Kd * qd_error

                    # Set the control input (torques)
                    data.ctrl[:6] = control_torque

                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(model.opt.timestep)

                    # ! Display the torques that are applied to the robot
                    print(f"{time.perf_counter()-start_time:.2f}\t{np.round(data.qfrc_actuator[:6], 2)}")

                print("\nFinal joint angles (deg):", np.round(np.degrees(data.qpos[:6]), 2))
                print("======================================")

            print("\n--- Finished all configurations. ---")
            input("Press Enter to close the viewer and exit...")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
