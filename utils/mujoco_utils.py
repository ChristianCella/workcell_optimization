import mujoco
import numpy as np
from transformations import rotm_to_quaternion, quaternion_to_euler

def get_cartesian_pose(frame_id, data, representation):
    position = data.xpos[frame_id]
    rotation_matrix = data.xmat[frame_id].reshape(3, 3)
    quaternion = rotm_to_quaternion(rotation_matrix)
    euler_angles = quaternion_to_euler(quaternion, degrees=True)
    if representation == "quaternion":
        return position, quaternion
    elif representation == "euler":
        return position, euler_angles
    elif representation == "rotation_matrix":
        return position, rotation_matrix

def set_body_pose(model, data, body_id, pos, quat):
    model.body_pos[body_id] = pos
    model.body_quat[body_id] = quat
    mujoco.mj_forward(model, data)

def compute_jacobian(model, data, tool_site_id):
    Jp = np.zeros((3, model.nv))
    Jr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, Jp, Jr, tool_site_id)
    Jac = np.vstack([Jp, Jr])[:, :6]
    return Jac

def get_collisions(model, data, verbose):
    # Step the simulator once so that contacts get populated
    mujoco.mj_forward(model, data)

    if data.ncon == 0:
        if verbose: print("No collisions detected.")
    else:
        if verbose: print(f"{data.ncon} collision(s) detected:")
        for i in range(data.ncon):
            c = data.contact[i]
            # lookup names via mj_id2name
            name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            if verbose: print(f"  • {name1} ↔ {name2}")
    return data.ncon

def inverse_manipulability(q, model, data, tool_site_id):
    data.qpos[:model.nv] = q; mujoco.mj_forward(model, data)
    Jp = np.zeros((3,model.nv)); Jr = np.zeros((3,model.nv))
    mujoco.mj_jacSite(model, data, Jp, Jr, tool_site_id)
    J = np.vstack([Jp, Jr])[:,:6]
    JJt = J @ J.T
    det = np.linalg.det(JJt)
    return 1e12 if det <= 1e-12 else 1.0/np.sqrt(det)

def directional_inverse_manipulability(q, model, data, tool_site_id, u_z):
    data.qpos[:model.nv] = q; mujoco.mj_forward(model, data)
    Jp = np.zeros((3,model.nv)); Jr = np.zeros((3,model.nv))
    mujoco.mj_jacSite(model, data, Jp, Jr, tool_site_id)
    J = np.vstack([Jp, Jr])[:,:6]
    dir_inv_man = u_z.T @ J @ J.T @ u_z
    return 1e12 if dir_inv_man <= 1e-12 else 1.0/np.sqrt(dir_inv_man)

def setup_target_frames(model, data, ref_body_ids, target_poses):
    for i, (pos, quat) in enumerate(target_poses):
        set_body_pose(model, data, ref_body_ids[i],
                      pos, [quat[3], quat[0], quat[1], quat[2]])
    mujoco.mj_forward(model, data)

def solve_ik_dls(model, data, tool_tip_site_id, target_pos, target_rot,
                 q_init, max_iter=100, tol=1e-5, lam=0.05):
    """
    Damped Least Squares IK solver.
    target_pos : (3,)   desired position in world frame
    target_rot : (3,3)  desired rotation matrix in world frame
    q_init     : (6,)   initial joint configuration (use q_old!)
    """
    q = q_init.copy()

    for _ in range(max_iter):
        # Forward kinematics
        data.qpos[:6] = q
        mujoco.mj_forward(model, data)

        # Position error
        curr_pos = data.site_xpos[tool_tip_site_id]
        err_pos  = target_pos - curr_pos

        # Orientation error (from rotation matrix difference)
        curr_rot = data.site_xmat[tool_tip_site_id].reshape(3, 3)
        R_err    = target_rot @ curr_rot.T
        # Convert skew-symmetric part to axis-angle error vector
        err_rot  = 0.5 * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1]
        ])

        # Full 6D error
        err = np.concatenate([err_pos, err_rot])
        if np.linalg.norm(err) < tol:
            break

        # Jacobian (6 x n_joints)
        J = compute_jacobian(model, data, tool_tip_site_id)

        # DLS step: Δq = Jᵀ (J Jᵀ + λ²I)⁻¹ Δx
        JJT  = J @ J.T
        dq   = J.T @ np.linalg.solve(JJT + lam**2 * np.eye(6), err)
        q   += dq

    return q

def joint_displacement(q1, q2):
    """Euclidean distance accounting for joint angle wrapping."""
    diff = q1 - q2
    diff = (diff + np.pi) % (2 * np.pi) - np.pi 
    return np.linalg.norm(diff)

