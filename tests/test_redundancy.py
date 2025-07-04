#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
import time, os, sys

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

def get_tool_z_direction(data, tool_site_id):
    mat = data.site_xmat[tool_site_id].reshape(3, 3)
    return mat[:, 2]  # Extract the z axis of the tool wrt world

def five_dof_error(pos_des, z_dir_des, pos, z_dir):
    pos_err = pos_des - pos
    err_axis = np.cross(z_dir, z_dir_des)
    err_proj = err_axis - np.dot(err_axis, z_dir_des) * z_dir_des
    return np.concatenate([pos_err, err_proj[:2]])

def five_dof_jacobian(model, data, tool_site_id, z_dir_des):
    nv = model.nv
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, tool_site_id)
    mat = data.site_xmat[tool_site_id].reshape(3, 3)
    z_axis = mat[:, 2] # Extract the z axis
    J_dir = np.zeros((3, nv))
    for i in range(nv):
        J_dir[:, i] = np.cross(jacr[:, i], z_axis)
    J_dir_proj = J_dir[:2, :] # Take teh first two rows, all the columns
    J5 = np.vstack([jacp, J_dir_proj])
    return J5[:, :6]

def manipulability(J):
    return np.sqrt(np.linalg.det(J @ J.T))

def grad_manipulability(q, model, data, tool_site_id, eps=1e-6):
    w0 = manipulability(get_full_jacobian(model, data, tool_site_id))
    grad = np.zeros_like(q)
    for i in range(len(q)):
        dq = np.zeros_like(q)
        dq[i] = eps
        q_pert = q + dq
        data.qpos[:len(q_pert)] = q_pert
        mujoco.mj_forward(model, data)
        Jp = get_full_jacobian(model, data, tool_site_id)
        w1 = manipulability(Jp)
        grad[i] = (w1 - w0) / eps
    return grad

def get_full_jacobian(model, data, tool_site_id):
    nv = model.nv
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, tool_site_id)
    J = np.vstack([jacp, jacr])
    return J[:, :6]

def ik5dof_tool_site(model, data, tool_site_id, target_pos, z_dir_des, max_iters=500, tol=1e-6):
    q = data.qpos[:6].copy()
    for _ in range(max_iters):
        mujoco.mj_forward(model, data)
        pos = data.site_xpos[tool_site_id]
        z_dir = get_tool_z_direction(data, tool_site_id)
        err = five_dof_error(target_pos, z_dir_des, pos, z_dir)
        if np.linalg.norm(err) < tol:
            break
        J5 = five_dof_jacobian(model, data, tool_site_id, z_dir_des)
        dq = np.linalg.pinv(J5, rcond=1e-4) @ err
        q += dq
        data.qpos[:6] = q
    return q

def maximize_manipulability(model, data, tool_site_id, target_pos, z_dir_des, q_init, nsteps=500, alpha=0.02):
    q = q_init.copy()
    for i in range(nsteps):
        data.qpos[:6] = q # ? Impose the previous joint configuration
        mujoco.mj_forward(model, data)
        J5 = five_dof_jacobian(model, data, tool_site_id, z_dir_des)
        J5_pinv = np.linalg.pinv(J5, rcond=1e-4)
        N = np.eye(6) - J5_pinv @ J5
        grad_w = grad_manipulability(q, model, data, tool_site_id)
        dq_null = N @ grad_w
        if np.linalg.norm(dq_null) > 1e-6:
            dq_null = dq_null / np.linalg.norm(dq_null)
        q = q + alpha * dq_null

    # Project back to the constraint manifold
    q = ik5dof_tool_site(model, data, tool_site_id, target_pos, z_dir_des, max_iters=5, tol=1e-4)
    return q

def main():
    verbose = True
    show_pose_duration = 5.0

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(base_dir)
    xml_path = os.path.join(base_dir, "universal_robots_ur5e", "scene.xml")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")
    ref_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "reference_target")   
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")

    target_poses = [
        (np.array([-0.3, -0.4, 0.05]), R.from_euler('xyz', [180, 0, 180], degrees=True).as_quat()),  # arbitrary orientation
    ]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        input("Press Enter to start the pose...")

        for pos, quat in target_poses:
            model.body_pos[base_body_id] = [0.1, 0.1, 0.2]
            model.body_quat[base_body_id] = euler_to_quaternion(45, 0, 0, degrees=True)
            model.body_pos[tool_body_id] = [0.1, 0.1, 0.25]
            model.body_quat[tool_body_id] = euler_to_quaternion(0, 0, 0, degrees=True)

            set_body_pose(model, data, ref_body_id, pos, [quat[3], quat[0], quat[1], quat[2]])
            mujoco.mj_forward(model, data)

            # ! 1) Find an intiial guess for the joint configuration
            q_init = ik_tool_site(model, data, tool_site_id, pos, quat)
            print(f"Initial 6-DoF IK solution: {np.round(q_init, 3)}")
            data.qpos[:6] = q_init
            mujoco.mj_forward(model, data)

            if verbose:
                print(f"\n[Init 6-DoF IK] Tool site: {np.round(data.site_xpos[tool_site_id],3)}")
                print(f"z_dir: {np.round(get_tool_z_direction(data, tool_site_id),3)}")
                print(f"Error norm: {np.linalg.norm(data.site_xpos[tool_site_id] - pos):.5f}")
                print(f"Manipulability: {manipulability(get_full_jacobian(model, data, tool_site_id)):.5f}")

            # Show initial guess for 5 seconds
            print("\nShowing initial guess (6-DoF solution) for 5 seconds...")
            viewer.sync()
            time.sleep(1)

            # ! 2) Resolution of redundancy
            target_pos = data.site_xpos[tool_site_id].copy()
            z_dir_ik = get_tool_z_direction(data, tool_site_id).copy()

            # --- Nullspace maximization (5-DoF: position + z-direction) ---
            q_manip = maximize_manipulability(model, data, tool_site_id, target_pos, z_dir_ik, q_init)
            print(f"Optimal configuration: {np.round(q_manip, 3)}")
            data.qpos[:6] = q_manip
            mujoco.mj_forward(model, data)
            if verbose:
                print(f"After nullspace search, tool site: {np.round(data.site_xpos[tool_site_id],3)}, z_dir: {np.round(get_tool_z_direction(data, tool_site_id),3)}")
                print(f"Manipulability: {manipulability(get_full_jacobian(model, data, tool_site_id)):.5f}")

            viewer.sync()
            time.sleep(show_pose_duration)

        if verbose: print("\n--- Finished ---")
        input("Press Enter to close the viewer...")

if __name__ == "__main__":
    main()
