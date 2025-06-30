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
    xml_path = os.path.join(base_dir, "universal_robots_ur5e/scene.xml")  # update path as needed

    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)

        target_qpos = np.radians([100, -94.96, 101.82, -95.72, -96.35, -40.97])

        base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        wrist_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")

        # External force and torque (world frame)
        external_force_world = np.array([50.0, 50.0, 20.0])
        external_torque_world = np.array([0.0, 0.0, 0.0])

        with mujoco.viewer.launch_passive(model, data) as viewer:
            print(f"\n--- DEBUG INFO ---")

            # Set base pose and joints
            model.body_pos[base_body_id] = [0.0, 0.0, 0.5]
            model.body_quat[base_body_id] = euler_to_quaternion(0, 0, 0, degrees=True)
            data.qpos[:6] = target_qpos
            data.qvel[:] = 0.0
            data.xfrc_applied[:] = 0.0
            #data.ctrl[:] = 0.0

            mujoco.mj_forward(model, data)

            # Joint and site info
            joint_names = [model.joint(i).name for i in range(6)]
            print("Actuated joint names:", joint_names)
            print("Site name:", model.site(site_id).name)
            print("Site world position:", data.site_xpos[site_id])
            rotmat = data.site_xmat[site_id].reshape(3, 3)
            quat = R.from_matrix(rotmat).as_quat()
            print(f"Site world orientation (quat): {quat}")
            print("wrist_3_link body world position:", data.xpos[wrist_body_id])
            print("Site body name:", model.body(model.site_bodyid[site_id]).name)
            print("Difference (site - body origin):", data.site_xpos[site_id] - data.xpos[wrist_body_id])

            # Jacobian at ee_site
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            np.set_printoptions(precision=4, suppress=True)
            print("jacp (position Jacobian):\n", jacp)
            print("jacr (rotation Jacobian):\n", jacr)

            J6 = np.vstack([jacp, jacr])[:, :6]
            wrench_world = np.hstack([external_force_world, external_torque_world])
            tau_ext = J6.T @ wrench_world

            gravity_comp = data.qfrc_bias[:6]
            total_torque = gravity_comp + tau_ext

            print("tau_ext (external force compensation):", tau_ext)
            print("gravity_comp:", gravity_comp)
            print("total_torque:", total_torque)

            print("\nApplying force and compensation torque for 3 seconds...\n")
            duration = 10.0
            start = time.perf_counter()
            print(f"the position is {data.site_xpos[site_id]}")

            #data.ctrl[:6] = total_torque

            while time.perf_counter() - start < duration:


                # --- Compute vector from body origin to site in world frame ---
                r = data.site_xpos[site_id] - data.xpos[wrist_body_id]
                
                corrected_force = external_force_world
                corrected_torque = np.cross(r, external_force_world) + external_torque_world

                # Apply force/torque at body origin (so as if it's acting at site)
                data.xfrc_applied[wrist_body_id, :3] = -corrected_force
                data.xfrc_applied[wrist_body_id, 3:] = -corrected_torque

                '''
                Mew part: add PD compensator 
                '''

                # In your control loop
                Kp = np.array([200, 200, 200, 100, 100, 100])
                Kd = np.array([20, 20, 20, 10, 10, 10])

                q_error = target_qpos - data.qpos[:6]
                qd_error = 0.0 - data.qvel[:6]
                pd_torque = Kp * q_error + Kd * qd_error

                data.ctrl[:6] = total_torque + pd_torque

                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)

            print("After compensation:")
            print("Final joint angles (deg):", np.round(np.degrees(data.qpos[:6]), 2))
            print("Final joint velocities:", np.round(data.qvel[:6], 4))
            if np.max(np.abs(data.qvel[:6])) < 0.01:
                print("Compensation working! Robot is stationary.")
            else:
                print("Compensation failed! Robot is still moving.")

            # Always clear applied force after loop
            data.xfrc_applied[wrist_body_id, :] = 0.0

    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
