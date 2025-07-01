#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
import time
import os
from scipy.spatial.transform import Rotation as R

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

        # Target poses for the robot
        target_qpos_list = [
            np.radians([100, -94.96, 101.82, -95.72, -96.35, -40.97]),
        ]

        # Define bodies, geometries and sites
        base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")

        tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")
        ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        force_arrow_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "force_arrow_geom")

        # Forces defined in the world frame and applied at the tool center
        external_force_world = np.array([0.0, 0.0, 5.0])
        external_torque_world = np.array([0.0, 0.0, 0.0])

        with mujoco.viewer.launch_passive(model, data) as viewer:

            input("Press Enter to start the robot configuration...")

            # Scan all the poses
            for idx, desired_qpos in enumerate(target_qpos_list):
                
                # Set the robot base to a new pose
                #model.body_pos[base_body_id] = [idx * 0.3, idx * 0.1, 0.5]
                model.body_pos[base_body_id] = [0.2, 0.2, 0.2]
                model.body_quat[base_body_id] = euler_to_quaternion(45, 0, 0, degrees=True)

                # Set the tool to a new pose with respect to the ee (flange)
                model.body_pos[tool_body_id] = [0.2, 0.2, 0.3]
                model.body_quat[tool_body_id] = euler_to_quaternion(0, 0, 0, degrees=True)

                # Set the position and velocity to 0
                data.qpos[:6] = desired_qpos.copy()
                data.qvel[:6] = 0.0
                mujoco.mj_forward(model, data)
                data.qacc[:] = 0.0

                # Simulate for 10 seconds
                seconds_per_config = 10.0
                start_time = time.perf_counter()
                while time.perf_counter() - start_time < seconds_per_config:
                    mujoco.mj_forward(model, data)

                    # Express torques on the ee in real world coodinates
                    #r = data.xpos[tool_body_id] - data.site_xpos[ee_site_id]
                    corrected_force = external_force_world
                    #corrected_torque = np.cross(r, external_force_world) + external_torque_world
                    corrected_torque = external_torque_world

                    # Apply the wrench in the simulation (of course, opposite to the previous)
                    data.xfrc_applied[tool_body_id, :3] = -corrected_force
                    data.xfrc_applied[tool_body_id, 3:] = -corrected_torque

                    # Get the Jacobians
                    jacp = np.zeros((3, model.nv))
                    jacr = np.zeros((3, model.nv))
                    mujoco.mj_jacSite(model, data, jacp, jacr, tool_site_id)
                    J6 = np.vstack([jacp, jacr])[:, :6]

                    # Wrench aplied at the end-effector in terms of world coordinates
                    wrench_world = np.hstack([external_force_world, external_torque_world])
                    tau_ext = J6.T @ wrench_world

                    # Compute the total torque (gravity + wrench) and add the PD controller
                    gravity_comp = data.qfrc_bias[:6]
                    total_torque = gravity_comp + tau_ext

                    #Kp = np.array([500, 500, 500, 150, 150, 150])
                    #Kd = np.array([30, 30, 30, 30, 30, 30])

                    Kp = np.array([5, 5, 5, 1.5, 1.5, 1.5])
                    Kd = np.array([0.3, 0.3,0.3, 0.8, 0.8, 0.8])

                    q_error = desired_qpos - data.qpos[:6]
                    qd_error = 0.0 - data.qvel[:6]
                    pd_torque = Kp * q_error + Kd * qd_error

                    data.ctrl[:6] = total_torque + pd_torque
                    #data.ctrl[:6] = total_torque

                    # === Update force arrow visualization ===
                    force = external_force_world
                    force_norm = np.linalg.norm(force)
                    if force_norm > 1e-8:
                        force_dir = force / force_norm
                    else:
                        force_dir = np.array([0, 0, 1])
                    arrow_length = 0.4 + 0.001 * force_norm
                    data.mocap_pos[0, :] = data.site_xpos[tool_site_id] + 0.5 * arrow_length * force_dir
                    z_axis = np.array([0, 0, 1])
                    rot, _ = R.align_vectors([force_dir], [z_axis])
                    quat = rot.as_quat()
                    data.mocap_quat[0, :] = [quat[3], quat[0], quat[1], quat[2]]
                    model.geom_size[force_arrow_geom_id][1] = arrow_length / 2
                    # ========================================

                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(model.opt.timestep)
                    #data.xfrc_applied[tool_body_id, :] = 0.0

                print("Final joint angles (deg):", np.round(np.degrees(data.qpos[:6]), 2))

            print("\n--- Finished all configurations. ---")
            input("Press Enter to close the viewer and exit...")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
