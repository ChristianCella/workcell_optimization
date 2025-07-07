#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import differential_evolution
import time, os, sys

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
    JJt = J @ J.T
    if np.linalg.matrix_rank(JJt) < 6 or np.linalg.det(JJt) < 1e-12:
        return 0.0
    return np.sqrt(np.linalg.det(JJt))

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
        (np.array([0.3, 0.4, 0.2]), R.from_euler('xyz', [180, 30, 45], degrees=True).as_quat()),
    ]

    # Use actual joint bounds from the model:
    bounds = list(zip(model.jnt_range[:6,0], model.jnt_range[:6,1]))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        input("Press Enter to start the pose...")

        for pos, quat in target_poses:
            model.body_pos[base_body_id] = [-0.1, -0.1, 0.0]
            model.body_quat[base_body_id] = euler_to_quaternion(0, 0, 0, degrees=True)
            model.body_pos[tool_body_id] = [0.0, 0.0, 0.0]
            model.body_quat[tool_body_id] = euler_to_quaternion(0, 0, 0, degrees=True)

            set_body_pose(model, data, ref_body_id, pos, [quat[3], quat[0], quat[1], quat[2]])
            mujoco.mj_forward(model, data)

            R_target = R.from_quat(quat).as_matrix()
            z_dir_des = R_target[:, 2].copy()

            PENALTY = 1e8  # You may want to tune this

            def penalty_objective(q):
                # Negative manipulability (since we minimize)
                m = manipulability(q, model, data, tool_site_id)
                constraint_violation = np.sum(five_dof_error(q, model, data, tool_site_id, pos, z_dir_des)**2)
                return -m + PENALTY * constraint_violation

            print("\nRunning global optimizer (differential evolution). This may take some time...")

            result = differential_evolution(
                penalty_objective,
                bounds,
                strategy='best1bin',
                popsize=20,
                maxiter=120,
                tol=1e-5,
                polish=True,
                updating='deferred'
            )
            q_opt = result.x
            data.qpos[:6] = q_opt
            mujoco.mj_forward(model, data)

            # Show and print results
            print("\nOptimized configuration (global):", np.round(q_opt, 3))
            print("Tool site:", np.round(data.site_xpos[tool_site_id],3))
            print("z_dir:", np.round(get_tool_z_direction(data, tool_site_id),3))
            print("Manipulability:", manipulability(q_opt, model, data, tool_site_id))
            print("Constraint violation (should be near zero):", np.round(five_dof_error(q_opt, model, data, tool_site_id, pos, z_dir_des), 6))

            viewer.sync()
            time.sleep(show_pose_duration)

        if verbose: print("\n--- Finished ---")
        input("Press Enter to close the viewer...")

if __name__ == "__main__":
    main()
