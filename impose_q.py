#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import os
import time

def euler_to_quaternion(roll, pitch, yaw, degrees=False):
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
    q = r.as_quat()
    return [q[3], q[0], q[1], q[2]]  # [w, x, y, z]

def set_body_pose(model, data, body_id, pos, quat):
    model.body_pos[body_id] = pos
    model.body_quat[body_id] = quat
    mujoco.mj_forward(model, data)

def ik_tool_site(model, data, tool_site_id, target_pos, target_quat, max_iters=200, tol=1e-4):
    q = data.qpos[:6].copy()
    for _ in range(max_iters):
        mujoco.mj_forward(model, data)
        pos = data.site_xpos[tool_site_id]
        mat = data.site_xmat[tool_site_id].reshape(3, 3)
        rot_current = R.from_matrix(mat)
        rot_target = R.from_quat(target_quat)
        pos_err = target_pos - pos
        # Axis-angle error for orientation
        rot_err_vec = (rot_target * rot_current.inv()).as_rotvec()
        err = np.hstack([pos_err, rot_err_vec])
        if np.linalg.norm(err) < tol:
            break
        # Get 6D jacobian for the site (world frame)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, tool_site_id)
        J = np.vstack([jacp, jacr])[:, :6]
        # Damped least squares IK step
        dq = np.linalg.pinv(J, rcond=1e-4) @ err
        q += dq
        data.qpos[:6] = q
    return q

def main():
    # Path setup
    base_dir = os.path.dirname(__file__)
    xml_path = os.path.join(base_dir, "universal_robots_ur5e", "scene.xml")

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Get body/site IDs
    tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")
    ref_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "reference_target")
    
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")

    # Define Cartesian target poses (world frame), as (position, quaternion)
    target_poses = [
        (np.array([0.5, 0.0, 0.6]), R.from_euler('xyz', [0, 0, 0], degrees=True).as_quat()),  # identity
        (np.array([0.4, 0.2, 0.5]), R.from_euler('xyz', [90, 0, 0], degrees=True).as_quat()), # tip down
        (np.array([0.3, -0.3, 0.4]), R.from_euler('xyz', [0, 90, 0], degrees=True).as_quat()), # rotate y
        (np.array([0.6, 0.1, 0.7]), R.from_euler('xyz', [45, 0, 90], degrees=True).as_quat()), # arbitrary
    ]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        input("Press Enter to start the poses (one per second)...")
        for pos, quat in target_poses:

            # Set robot and tool
            model.body_pos[base_body_id] = [0.1, 0.1, 0.5]
            model.body_quat[base_body_id] = euler_to_quaternion(45, 45, 0, degrees=True)

            # Set the tool to a new pose with respect to the ee (flange)
            model.body_pos[tool_body_id] = [0.1, 0.1, 0.1]
            model.body_quat[tool_body_id] = euler_to_quaternion(0, 0, 0, degrees=True)
            

            print(f"\n==> Target Cartesian pose: pos {np.round(pos, 3)}, quat {np.round(quat, 3)}")

            # Set the reference body + site (so it's visible)
            set_body_pose(model, data, ref_body_id, pos, [quat[3], quat[0], quat[1], quat[2]])
            mujoco.mj_forward(model, data)

            # Compute IK for the tool_site to reach this pose
            q_sol = ik_tool_site(model, data, tool_site_id, pos, quat)
            data.qpos[:6] = q_sol
            mujoco.mj_forward(model, data)
            print(f"IK joint solution (deg): {np.round(np.degrees(q_sol), 2)}")
            print(f"Tool site reached: {np.round(data.site_xpos[tool_site_id], 3)}")
            print(f"Error norm: {np.linalg.norm(data.site_xpos[tool_site_id] - pos):.5f}")
            print("Actual applied torques:", data.qfrc_actuator[:6])

            viewer.sync()
            time.sleep(5.0)  # show each pose for 1s

        print("\n--- Finished all target poses. ---")
        input("Press Enter to close the viewer...")

if __name__ == "__main__":
    main()
