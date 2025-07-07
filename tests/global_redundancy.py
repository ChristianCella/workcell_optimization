#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import differential_evolution, minimize
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
    # Penalize sign (anti-alignment) via dot product (see below)
    err_axis = np.cross(z_dir, z_dir_des)
    err_proj = err_axis - np.dot(err_axis, z_dir_des) * z_dir_des
    return np.concatenate([pos_err, err_proj[:2]])

def five_dof_error_with_sign(q, model, data, tool_site_id, target_pos, z_dir_des):
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    pos = data.site_xpos[tool_site_id]
    z_dir = get_tool_z_direction(data, tool_site_id)
    pos_err = target_pos - pos
    err_axis = np.cross(z_dir, z_dir_des)
    err_proj = err_axis - np.dot(err_axis, z_dir_des) * z_dir_des
    # Penalize anti-alignment
    sign_penalty = max(0, -np.dot(z_dir, z_dir_des))  # Only positive when dot < 0
    return np.concatenate([pos_err, err_proj[:2], [sign_penalty]])

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

# Cost weights (tune as desired)
W_POSE = 1e3
W_JOINT_DISP = 1e-1
W_LIMITS = 1e-1
W_ELBOW = 1e-1
W_MANIP = -1.0
W_COLLISION = 100
W_ANTIALIGN = 1e6  # Strong penalty for z misalignment

def table_geom_collision_cost(q, model, data, min_z=0.03):
    """
    Check all geoms for penetration below z=min_z.
    Returns penalty: 0 if all are OK, or huge penalty if any are below.
    """
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    penalty = 0.0
    for g in range(model.ngeom):
        geom_z = data.geom_xpos[g][2]
        # Optional: ignore world/floor geoms (skip index 0 or 1, depending on your model)
        # If geom is attached to world/floor, skip.
        if model.geom_bodyid[g] == 0:
            continue
        if geom_z < min_z:
            penalty += (min_z - geom_z)**2
    return penalty

def self_collision_cost(q, model, data):
    """
    Returns a large penalty if there are any self-collisions
    (excluding robot-floor or world collisions).
    """
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    for i in range(data.ncon):
        c = data.contact[i]
        # Get the two geom ids involved in the contact
        g1, g2 = c.geom1, c.geom2
        # Get the body ids for each geom
        b1, b2 = model.geom_bodyid[g1], model.geom_bodyid[g2]
        # Ignore world (body 0 or -1), or ground collision, only care about self-collision
        if b1 > 0 and b2 > 0 and b1 != b2:
            # Not the world, not the same body (so not self-contact within the same body)
            return 1.0  # 1 collision found, could return more info if needed
    return 0.0

def bioik_cost(q, model, data, tool_site_id, target_pos, z_dir_des, q_seed, joint_lims, elbow_pref=None, min_z=0.03):
    # Pose error
    pose_err = five_dof_error(q, model, data, tool_site_id, target_pos, z_dir_des)
    cost_pose = np.sum(pose_err**2)

    # Joint displacement (from seed)
    cost_joint_disp = np.sum((q - q_seed)**2)

    # Joint limits (penalize being near limits)
    lower, upper = joint_lims[:,0], joint_lims[:,1]
    mid = 0.5 * (lower + upper)
    range_ = upper - lower
    cost_limits = np.sum(((2 * np.abs(q - mid) - 0.5 * range_)**2))

    # Elbow cost (for UR robots, joint 4 is "elbow")
    if elbow_pref is not None:
        el, eh = elbow_pref
        cost_elbow = (2 * np.abs(q[3] - 0.5*(eh + el)) - 0.5*(eh - el))**2
    else:
        cost_elbow = 0.0

    # Manipulability (maximize)
    manip = manipulability(q, model, data, tool_site_id)

    # Table collision penalty (geom-based)
    collision_penalty = table_geom_collision_cost(q, model, data, min_z=min_z)  # All geoms above table

    # Self-collision penalty
    selfcol_penalty = self_collision_cost(q, model, data)

    # Penalize anti-alignment (z_dir·z_dir_des < 0)
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    z_dir = get_tool_z_direction(data, tool_site_id)
    dot_z = np.dot(z_dir, z_dir_des)
    anti_align_penalty = 0.0
    if dot_z < 0:
        anti_align_penalty = -dot_z  # Bigger penalty the more negative

    # Weighted sum (BioIK philosophy)
    cost = (W_POSE * cost_pose +
            W_JOINT_DISP * cost_joint_disp +
            W_LIMITS * cost_limits +
            W_ELBOW * cost_elbow +
            W_MANIP * manip +
            W_COLLISION * (collision_penalty + selfcol_penalty) +
            W_ANTIALIGN * anti_align_penalty)
    return cost

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
        (np.array([-0.3, -0.4, 0.2]), R.from_euler('xyz', [180, 20, 45], degrees=True).as_quat()),
    ]

    joint_lims = model.jnt_range[:6]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        input("Press Enter to start the pose...")

        for pos, quat in target_poses:
            model.body_pos[base_body_id] = [0.0, 0.0, 0.0]
            model.body_quat[base_body_id] = euler_to_quaternion(0, 0, 0, degrees=True)
            model.body_pos[tool_body_id] = [0.0, 0.0, 0.0]
            model.body_quat[tool_body_id] = euler_to_quaternion(0, 0, 0, degrees=True)

            set_body_pose(model, data, ref_body_id, pos, [quat[3], quat[0], quat[1], quat[2]])
            mujoco.mj_forward(model, data)

            R_target = R.from_quat(quat).as_matrix()
            z_dir_des = R_target[:, 2].copy()

            # Elbow preference range (optional)
            elbow_pref = (joint_lims[3,0], joint_lims[3,0]+0.4*(joint_lims[3,1]-joint_lims[3,0]))

            print("\nGlobal (BioIK-style) search (this may take a while)...")
            def cost_wrap(q):
                return bioik_cost(q, model, data, tool_site_id, pos, z_dir_des, q_seed=np.zeros(6),
                                  joint_lims=joint_lims, elbow_pref=elbow_pref, min_z=0.03)

            bounds = list(zip(joint_lims[:,0], joint_lims[:,1]))

            result = differential_evolution(
                cost_wrap,
                bounds,
                popsize=20,
                maxiter=80,
                polish=True,
                updating='deferred'
            )
            q_global = result.x

            print("Refining globally found configuration with local gradient search...")
            def local_cost(q):
                return bioik_cost(q, model, data, tool_site_id, pos, z_dir_des, q_seed=q_global,
                                  joint_lims=joint_lims, elbow_pref=elbow_pref, min_z=0.03)

            res = minimize(
                local_cost,
                q_global,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-8, 'maxiter': 200}
            )
            q_opt = res.x

            # Apply and show solution
            data.qpos[:6] = q_opt
            mujoco.mj_forward(model, data)
            print("\nFinal configuration:", np.round(q_opt, 3))
            print("Tool site:", np.round(data.site_xpos[tool_site_id],3))
            print("z_dir:", np.round(get_tool_z_direction(data, tool_site_id),3))
            print("Pose cost (should be ~0):", np.sum(five_dof_error_with_sign(q_opt, model, data, tool_site_id, pos, z_dir_des)**2))
            print("Manipulability:", manipulability(q_opt, model, data, tool_site_id))

            viewer.sync()
            time.sleep(show_pose_duration)

        if verbose: print("\n--- Finished ---")
        input("Press Enter to close the viewer...")

if __name__ == "__main__":
    main()
