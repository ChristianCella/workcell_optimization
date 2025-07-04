#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize, NonlinearConstraint
import time, os, sys

''' 
This script demonstrates how to use nonlinear constrained optimization to solve the redundancy problem in a 6-DOF robot arm.
Compare the results with the previous test_redundancy.py script.
'''

def euler_to_quaternion(roll, pitch, yaw, degrees=False):
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
    q = r.as_quat()
    return [q[3], q[0], q[1], q[2]]  # [w, x, y, z]

def set_body_pose(model, data, body_id, pos, quat):
    model.body_pos[body_id] = pos
    model.body_quat[body_id] = quat
    mujoco.mj_forward(model, data)

def get_tool_z_direction(data, tool_site_id):
    mat = data.site_xmat[tool_site_id].reshape(3, 3)
    return mat[:, 2]

def five_dof_error(q, model, data, tool_site_id, target_pos, z_dir_des):
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    pos = data.site_xpos[tool_site_id]
    z_dir = get_tool_z_direction(data, tool_site_id)
    pos_err = target_pos - pos
    err_axis = np.cross(z_dir, z_dir_des)
    err_proj = err_axis - np.dot(err_axis, z_dir_des) * z_dir_des
    return np.concatenate([pos_err, err_proj[:2]])

def get_full_jacobian(model, data, tool_site_id):
    nv = model.nv
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, tool_site_id)
    J = np.vstack([jacp, jacr])
    return J[:, :6]

def manipulability(q, model, data, tool_site_id):
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    J = get_full_jacobian(model, data, tool_site_id)
    # Robust for rank-deficient J
    JJt = J @ J.T
    return 1/np.sqrt(np.linalg.det(JJt)) if np.linalg.matrix_rank(JJt) == 6 else 0.0

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

    # Target pose: arbitrary orientation, only z-axis alignment matters!
    target_poses = [
        (np.array([-0.3, -0.4, 0.05]), R.from_euler('xyz', [180, 0, 180], degrees=True).as_quat()),
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

            # Find an initial guess for the joint configuration
            def ik_tool_site(model, data, tool_site_id, target_pos, target_quat, max_iters=200, tol=1e-4):
                q = data.qpos[:6].copy()
                for _ in range(max_iters):
                    mujoco.mj_forward(model, data)
                    pos_now = data.site_xpos[tool_site_id]
                    mat = data.site_xmat[tool_site_id].reshape(3, 3)
                    rot_current = R.from_matrix(mat)
                    rot_target = R.from_quat(target_quat)
                    pos_err = target_pos - pos_now
                    rot_err_vec = (rot_target * rot_current.inv()).as_rotvec()
                    err = np.hstack([pos_err, rot_err_vec])
                    if np.linalg.norm(err) < tol:
                        break
                    jacp = np.zeros((3, model.nv))
                    jacr = np.zeros((3, model.nv))
                    mujoco.mj_jacSite(model, data, jacp, jacr, tool_site_id)
                    J = np.vstack([jacp, jacr])[:, :6]
                    dq = np.linalg.pinv(J, rcond=1e-4) @ err
                    q += dq
                    data.qpos[:6] = q
                return q

            q_init = ik_tool_site(model, data, tool_site_id, pos, quat)
            data.qpos[:6] = q_init
            mujoco.mj_forward(model, data)

            viewer.sync()
            time.sleep(show_pose_duration)

            R_target = R.from_quat(quat).as_matrix()
            z_dir_des = R_target[:, 2].copy()

            if verbose:
                print(f"\nInitial guess: {np.round(q_init, 3)}")
                print(f"Tool site: {np.round(data.site_xpos[tool_site_id],3)}")
                print(f"z_dir: {np.round(get_tool_z_direction(data, tool_site_id),3)}")
                print(f"Manipulability: {1/manipulability(q_init, model, data, tool_site_id):.5f}")

            # --- Nonlinear constrained optimization: maximize manipulability ---
            def obj(q):
                return manipulability(q, model, data, tool_site_id)  # minimize negative => maximize

            def constraint(q):
                # Want five_dof_error == 0
                return five_dof_error(q, model, data, tool_site_id, pos, z_dir_des)

            # Nonlinear constraint: all 5 elements should be zero
            cons = [NonlinearConstraint(constraint, 0, 0)]
            #bounds = [(-np.pi, np.pi)] * 6  # Set bounds for joints

            res = minimize(
                obj,
                q_init,
                method='trust-constr',
                constraints=cons,
                #bounds=bounds,
                options={'verbose': 2, 'maxiter': 100}
            )
            q_opt = res.x
            data.qpos[:6] = q_opt
            mujoco.mj_forward(model, data)

            if verbose:
                print(f"\nOptimized configuration: {np.round(q_opt, 3)}")
                print(f"Tool site: {np.round(data.site_xpos[tool_site_id],3)}")
                print(f"z_dir: {np.round(get_tool_z_direction(data, tool_site_id),3)}")
                print(f"Manipulability: {1/manipulability(q_opt, model, data, tool_site_id):.5f}")

            viewer.sync()
            time.sleep(show_pose_duration)

        if verbose: print("\n--- Finished ---")
        input("Press Enter to close the viewer...")

if __name__ == "__main__":
    main()
