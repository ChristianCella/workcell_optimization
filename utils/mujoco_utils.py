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

def compute_jacobian(model, data, rob_par, tool_site_id):
    Jp = np.zeros((3, rob_par.nu))
    Jr = np.zeros((3, rob_par.nu))
    mujoco.mj_jacSite(model, data, Jp, Jr, tool_site_id)
    Jac = np.vstack([Jp, Jr])[:, :rob_par.nu]
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
            if verbose: print(f" - {name1} <-> {name2}")
            if verbose: print(f"Penetration depth: {c.dist}")
    return data.ncon

def inverse_manipulability(q, model, data, rob_par, tool_site_id):
    data.qpos[:rob_par.nu] = q 
    mujoco.mj_forward(model, data)
    J = compute_jacobian(model, data, rob_par, tool_site_id)
    JJt = J @ J.T
    det = np.linalg.det(JJt)
    return 1e12 if det <= 1e-6 else 1.0/np.sqrt(det)

def setup_target_frames(model, data, ref_body_ids, target_poses):
    for i, (pos, quat) in enumerate(target_poses):
        set_body_pose(model, data, ref_body_ids[i],
                      pos, [quat[3], quat[0], quat[1], quat[2]])
    mujoco.mj_forward(model, data)

def solve_ik_dls(model, data, rob_par, tool_tip_site_id, target_pos, target_rot,
                 q_init, max_iter=100, tol=1e-5, lam_max=0.1, eps=1e-6,
                 dq_max=0.05):  # <-- max joint change per iteration in rad
    """
    DLS with per-iteration joint velocity clamping.
    dq_max: maximum allowed joint displacement per iteration (rad).
            Smaller = more stable but slower convergence.
            Typical range: 0.01 -- 0.1 rad
    """
    q   = q_init.copy()
    res = 1

    for _ in range(max_iter):
        data.qpos[:rob_par.nu] = q
        mujoco.mj_forward(model, data)

        curr_pos = data.site_xpos[tool_tip_site_id]
        err_pos  = target_pos - curr_pos

        curr_rot = data.site_xmat[tool_tip_site_id].reshape(3, 3)
        R_err    = target_rot @ curr_rot.T
        err_rot  = 0.5 * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1]
        ])

        err      = np.concatenate([err_pos, err_rot])
        err_norm = np.linalg.norm(err)

        if err_norm < tol:
            res = 0
            break

        J = compute_jacobian(model, data, rob_par, tool_tip_site_id)

        lam = lam_max * (err_norm / (err_norm + eps))
        JJT = J @ J.T
        dq  = J.T @ np.linalg.solve(JJT + lam**2 * np.eye(6), err)

        # Clamp: if any joint would move more than dq_max,
        # scale the entire dq vector down proportionally
        max_dq = np.abs(dq).max()
        if max_dq > dq_max:
            dq *= dq_max / max_dq

        q += dq

    return q, res

def joint_displacement(q1, q2):
    """Euclidean distance accounting for joint angle wrapping."""
    diff = q1 - q2
    diff = (diff + np.pi) % (2 * np.pi) - np.pi 
    return np.linalg.norm(diff)

def solve_ik_clik(model, data, rob_par, tool_tip_site_id, target_pos, target_rot,
                  q_init, max_iter=100, tol=1e-5, gain=0.5, dt=0.01):
    """
    Closed-Loop Inverse Kinematics (CLIK).
    
    Instead of computing a direct joint displacement like DLS, CLIK computes
    a joint velocity that steers the end-effector toward the target, then
    integrates it over a small timestep dt. The feedback gain controls how
    aggressively the error is reduced at each step.

    Args:
        gain  : proportional feedback gain (0 < gain <= 1). Higher = faster
                convergence but may overshoot. Typical range: 0.1 -- 1.0
        dt    : integration timestep (s). Smaller = smoother but slower.
                Typical range: 0.001 -- 0.1
    """
    q   = q_init.copy()
    res = 1

    for _ in range(max_iter):
        data.qpos[:rob_par.nu] = q
        mujoco.mj_forward(model, data)

        # Position error
        curr_pos = data.site_xpos[tool_tip_site_id]
        err_pos  = target_pos - curr_pos

        # Orientation error (axis-angle from rotation matrix)
        curr_rot = data.site_xmat[tool_tip_site_id].reshape(3, 3)
        R_err    = target_rot @ curr_rot.T
        err_rot  = 0.5 * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1]
        ])

        err      = np.concatenate([err_pos, err_rot])
        err_norm = np.linalg.norm(err)

        if err_norm < tol:
            res = 0
            break

        J = compute_jacobian(model, data, rob_par, tool_tip_site_id)

        # CLIK: joint velocity = J^+ * (gain * err / dt)
        # J^+ = J^T (J J^T)^{-1}  — Moore-Penrose pseudoinverse
        # Regularised to avoid issues near singularities
        lam  = 1e-4   # small fixed regularisation
        JJT  = J @ J.T
        dq_dt = J.T @ np.linalg.solve(JJT + lam * np.eye(6), gain * err / dt)

        # Integrate
        q += dq_dt * dt

    return q, res

import numpy as np
from scipy.optimize import minimize


def solve_ik_scipy(target_pos, target_rot, q_init, joint_limits,
                   fk_func, tol=1e-6, w_posture=1e-3):
    """
    IK via scipy L-BFGS-B minimization — same approach as IKPy.

    Args:
        target_pos   : (3,)   desired end-effector position
        target_rot   : (3,3)  desired end-effector rotation matrix
        q_init       : (nu,)  seed configuration — use q_prev for path tracking
        joint_limits : (nu,2) array of [lower, upper] bounds per joint
        fk_func      : callable q -> (pos (3,), rot (3,3))
                       your existing FK using MuJoCo
        tol          : convergence tolerance on the cost
        w_posture    : weight on posture regularisation (pulls toward q_init)
                       higher = stays closer to seed = less branch switching

    Returns:
        q_sol : (nu,) solution
        err   : float final pose error norm
    """

    def cost(q):
        pos, rot = fk_func(q)

        # Position error
        err_pos = target_pos - pos                        # (3,)

        # Orientation error (same as your DLS formula)
        R_err   = target_rot @ rot.T
        err_rot = 0.5 * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1]
        ])                                                # (3,)

        # Posture regularisation — pulls toward q_init
        err_posture = q - q_init                         # (nu,)

        return (err_pos @ err_pos
                + err_rot @ err_rot
                + w_posture * (err_posture @ err_posture))

    bounds = [(lo, hi) for lo, hi in joint_limits]

    result = minimize(
        cost,
        q_init,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": tol, "gtol": 1e-8, "maxiter": 200}
    )

    q_sol     = result.x
    pos, rot  = fk_func(q_sol)
    err_pos   = np.linalg.norm(target_pos - pos)
    R_err     = target_rot @ rot.T
    err_rot   = 0.5 * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1]
    ])
    err_norm  = np.sqrt(err_pos**2 + np.linalg.norm(err_rot)**2)

    return q_sol, err_norm

def make_fk(model, data, rob_par, tool_tip_site_id):
    """
    Returns a callable q -> (pos, rot) using MuJoCo FK.
    """
    def fk(q):
        data.qpos[:rob_par.nu] = q
        mujoco.mj_forward(model, data)
        pos = data.site_xpos[tool_tip_site_id].copy()
        rot = data.site_xmat[tool_tip_site_id].reshape(3, 3).copy()
        return pos, rot
    return fk



