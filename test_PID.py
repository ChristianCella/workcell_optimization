#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
import time
import os
from scipy.spatial.transform import Rotation as R

# PID gains
Kp = np.array([400, 400, 400, 400, 400, 400])
Kd = np.array([40, 40, 40, 30, 30, 30])
Ki = np.array([7, 7, 7, 7, 7, 7])

def euler_to_quaternion(roll, pitch, yaw, degrees=False):
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
    q = r.as_quat()
    return [q[3], q[0], q[1], q[2]]  # [w, x, y, z]

def main():
    base_dir = os.path.dirname(__file__)
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
        tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")

        # External wrench (edit as you like)
        external_force_world = np.array([0.0, 0.0, 0.0])
        external_torque_world = np.array([0.0, 0.0, 0.0])

        with mujoco.viewer.launch_passive(model, data) as viewer:
            input("Press Enter to start the robot configuration...")

            for i, desired_qpos in enumerate(target_qpos_list):
                print(f"\n==== Moving to target configuration {i+1} ====")

                # (Optional) move base or tool if you want
                # model.body_pos[base_body_id] = [i*0.2, 0.0, 0.5]
                # model.body_quat[base_body_id] = euler_to_quaternion(45, 45, 0, degrees=True)
                # model.body_pos[tool_body_id] = [0.1, 0.1, 0.1]
                # model.body_quat[tool_body_id] = euler_to_quaternion(0, 0, 0, degrees=True)

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

                    # PRINT: what torques are actually being applied at each joint?
                    print(f"{time.perf_counter()-start_time:.2f}\t{np.round(data.qfrc_actuator[:6], 2)}")

                print("\nFinal joint angles (deg):", np.round(np.degrees(data.qpos[:6]), 2))
                print("======================================")

            print("\n--- Finished all configurations. ---")
            input("Press Enter to close the viewer and exit...")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
