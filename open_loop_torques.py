#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
import time
import os
from scipy.spatial.transform import Rotation as R

''' 
Open loop control in torque to compensate gravity and external wrench.
PD (or PID) used to correct minor errors in the inverse dynamics computation.
'''

def euler_to_quaternion(roll, pitch, yaw, degrees=False):
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
    q = r.as_quat()
    return [q[3], q[0], q[1], q[2]]  # [w, x, y, z]

def main():
    # === User switch: True = smooth move (PD+force); False = instant placement with gravity+external compensation ===
    move_robot = True  # <----- SET THIS TO False for instantaneous placement

    base_dir = os.path.dirname(__file__)
    xml_path = os.path.join(base_dir, "universal_robots_ur5e/scene.xml")

    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)

        for i in range(model.nbody):
            print(model.body(i).name)

        # Target joint configurations (in radians)
        target_qpos_list = [
            np.radians([100, -94.96, 101.82, -95.72, -96.35, -40.97]),
            np.radians([15.99, -62.07, 121.58, -159.82, -99, -90]),
            np.radians([-28.83, -84.36, 102.8, 18.69, -98.83, -101.46]),
            np.radians([-29.77, -160.55, 110.57, -186.16, -98.81, -101.15]),
            np.radians([-124.35, -158.41, 147, -153.21, -155.31, -98.24])
        ]

        base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        wrist_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")

        # External force and torque (world frame)
        external_force_world = np.array([0.0, 0.0, 50.0])
        external_torque_world = np.array([20.0, 10.0, 0.0])

        with mujoco.viewer.launch_passive(model, data) as viewer:

            # Wait for enter to be pressed
            input("Press Enter to start the robot configuration...")

            # Start iterating
            for idx, desired_qpos in enumerate(target_qpos_list):

                print(f"\n==== Configuration {idx+1} ====")
                # Move base for fun (or keep fixed if you want)
                model.body_pos[base_body_id] = [idx * 0.3, idx * 0.1, 0.5]
                model.body_quat[base_body_id] = euler_to_quaternion(70, -20, 30, degrees=True)

                if move_robot:
                    # Smoothly move using PD + external force
                    seconds_per_config = 10.0
                    start_time = time.perf_counter()
                    last_time = start_time
                    error_integral = np.zeros(6)

                    while time.perf_counter() - start_time < seconds_per_config:

                        current_time = time.perf_counter()
                        dt = current_time - last_time
                        last_time = current_time

                        mujoco.mj_forward(model, data)
                        r = data.site_xpos[site_id] - data.xpos[wrist_body_id]
                        print(f"The value of r is: {r}")
                        corrected_force = external_force_world
                        corrected_torque = np.cross(r, external_force_world) + external_torque_world
                        data.xfrc_applied[wrist_body_id, :3] = -corrected_force
                        data.xfrc_applied[wrist_body_id, 3:] = -corrected_torque

                        jacp = np.zeros((3, model.nv))
                        jacr = np.zeros((3, model.nv))
                        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
                        J6 = np.vstack([jacp, jacr])[:, :6]
                        wrench_world = np.hstack([external_force_world, external_torque_world])
                        tau_ext = J6.T @ wrench_world

                        gravity_comp = data.qfrc_bias[:6]
                        total_torque = gravity_comp + tau_ext

                        # PD control
                        Kp = np.array([100, 100, 100, 20, 20, 20])
                        Kd = np.array([20, 20, 20, 10, 10, 10])
                        Ki = np.array([3, 3, 3, 1, 1, 1])

                        q_error = desired_qpos - data.qpos[:6]
                        qd_error = 0.0 - data.qvel[:6]
                        error_integral += q_error * dt

                        pid_torque = Kp * q_error + Ki * error_integral + Kd * qd_error

                        data.ctrl[:6] = total_torque + pid_torque

                        mujoco.mj_step(model, data)
                        viewer.sync()
                        time.sleep(model.opt.timestep)
                    data.xfrc_applied[wrist_body_id, :] = 0.0  # Clear

                else:
                    # Instantly set the robot to the new configuration
                    data.qpos[:6] = desired_qpos.copy()
                    data.qvel[:6] = 0.0
                    mujoco.mj_forward(model, data)
                    data.qacc[:] = 0.0

                    seconds_per_config = 3.0
                    start_time = time.perf_counter()
                    while time.perf_counter() - start_time < seconds_per_config:
                        mujoco.mj_forward(model, data)
                        # External force at this pose
                        r = data.site_xpos[site_id] - data.xpos[wrist_body_id]
                        corrected_force = external_force_world
                        corrected_torque = np.cross(r, external_force_world) + external_torque_world
                        data.xfrc_applied[wrist_body_id, :3] = -corrected_force
                        data.xfrc_applied[wrist_body_id, 3:] = -corrected_torque

                        jacp = np.zeros((3, model.nv))
                        jacr = np.zeros((3, model.nv))
                        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
                        J6 = np.vstack([jacp, jacr])[:, :6]
                        wrench_world = np.hstack([external_force_world, external_torque_world])
                        tau_ext = J6.T @ wrench_world

                        gravity_comp = data.qfrc_bias[:6]
                        total_torque = gravity_comp + tau_ext

                        # PD control
                        Kp = np.array([500, 500, 500, 150, 150, 150])
                        Kd = np.array([30, 30, 30, 30, 30, 30])

                        q_error = desired_qpos - data.qpos[:6]
                        qd_error = 0.0 - data.qvel[:6]
                        pd_torque = Kp * q_error + Kd * qd_error

                        # No PD since we do not want movement, just compensation
                        data.ctrl[:6] = total_torque + pd_torque

                        mujoco.mj_step(model, data)
                        viewer.sync()
                        time.sleep(model.opt.timestep)
                    data.xfrc_applied[wrist_body_id, :] = 0.0  # Clear applied force after

                # KPIs, print or collect as you wish
                print("Final joint angles (deg):", np.round(np.degrees(data.qpos[:6]), 2))

            print("\n--- Finished all configurations. ---")
            input("Press Enter to close the viewer and exit...")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
