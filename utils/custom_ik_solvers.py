from mujoco_utils import compute_jacobian, joint_displacement
from transformations import rotm2euler

import mujoco, time
import numpy as np
import mink
from mink import Configuration, FrameTask, solve_ik, PostureTask
from scipy.optimize import minimize
import threading, copy

def solve_ik_dls(model, data, rob_par, tool_tip_site_id, target_pos, target_rot,
                 q_init, max_iter=100, tol=1e-5, lam_max=0.1, eps=1e-6,
                 dq_max=0.05):
    q   = q_init.copy()
    res = 1

    # Joint limits from MuJoCo model
    q_min = model.jnt_range[:rob_par.nu, 0]
    q_max = model.jnt_range[:rob_par.nu, 1]

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

        # Clamp step size
        max_dq = np.abs(dq).max()
        if max_dq > dq_max:
            dq *= dq_max / max_dq

        q += dq

        # Clamp to joint limits
        q = np.clip(q, q_min, q_max)

    return q, res

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

def ik_mink(ee_site, model, target_pos, target_rot, q_init):
    task = FrameTask(
        frame_name=ee_site,
        frame_type="body",
        position_cost=1.0,
        orientation_cost=1.0,
    )

    target_pos = np.array(target_pos)
    rot = rotm2euler(target_rot, degrees=False)
    rotation = mink.SO3.from_rpy_radians(
        roll=rot[0],
        pitch=rot[1],
        yaw=rot[2],
    )
    #print(f"The rotation is: {rotation}")

    configuration = Configuration(model, q=q_init)
    mujoco.mj_forward(model, configuration.data)
    posture_task = PostureTask(model, cost=1e-2)
    posture_task.set_target(q_init)

    target_pose = mink.SE3.from_rotation_and_translation(
        rotation=rotation,
        translation=target_pos,
    )

    task.set_target(target_pose)

    dt = 0.01
    start = time.perf_counter()
    for _ in range(300):
        vel = solve_ik(
            configuration,
            [task, posture_task],
            dt=dt,
            solver="ecos",
        )
        configuration.integrate_inplace(vel, dt)

    #print(f"Time taken: {time.perf_counter() - start:.4f} seconds")
    #print("qpos:")
    #print(configuration.q)
    q = np.array(configuration.q)
    res = 0 if joint_displacement(q, q_init) < 1e-4 else 1
    return q, res

def solve_ik_tracik(model, data, rob_par, tool_tip_site_id, target_pos, target_rot,
                    q_init, timeout=1.0, tol=1e-5, lam_max=0.1, eps=1e-6, dq_max=0.05):
    """
    Pure Python TRAC-IK implementation.
    Drop-in replacement for solve_ik_dls — same signature, same return values.

    Returns:
        q_sol : (nu,) best solution found
        res   : 0 if converged (err < tol), 1 otherwise — same as solve_ik_dls
    """

    q_min = model.jnt_range[:rob_par.nu, 0]
    q_max = model.jnt_range[:rob_par.nu, 1]

    # Each solver gets its own MuJoCo data copy — thread safety
    data_a = copy.copy(data)
    data_b = copy.copy(data)

    # Shared result — initialised with q_init so we always return something
    result      = {"q": q_init.copy(), "err": np.inf}
    result_lock = threading.Lock()
    stop_event  = threading.Event()

    # ------------------------------------------------------------------ #
    # Helper: compute pose error                                          #
    # ------------------------------------------------------------------ #
    def pose_error(q, d):
        d.qpos[:rob_par.nu] = q
        mujoco.mj_forward(model, d)
        curr_pos = d.site_xpos[tool_tip_site_id].copy()
        curr_rot = d.site_xmat[tool_tip_site_id].reshape(3, 3).copy()
        err_pos  = target_pos - curr_pos
        R_err    = target_rot @ curr_rot.T
        err_rot  = 0.5 * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1]
        ])
        return np.concatenate([err_pos, err_rot])

    # ------------------------------------------------------------------ #
    # Solver A — DLS with joint limit clamping and singularity detection  #
    # ------------------------------------------------------------------ #
    def solver_dls():
        q = q_init.copy()

        for _ in range(10000):
            if stop_event.is_set():
                break

            err      = pose_error(q, data_a)
            err_norm = np.linalg.norm(err)

            with result_lock:
                if err_norm < result["err"]:
                    result["q"]   = q.copy()
                    result["err"] = err_norm

            if err_norm < tol:
                stop_event.set()
                return

            J = compute_jacobian(model, data_a, rob_par, tool_tip_site_id)

            # Singularity detection — stop iterating if Jacobian is degenerate
            # Continuing would drive the solver deeper into the singularity
            det_J = abs(np.linalg.det(J))
            if det_J < 1e-6:
                break

            # Minimum damping floor prevents underdamping near singularities
            # where err_norm is small but J is ill-conditioned
            lam = max(lam_max * (err_norm / (err_norm + eps)), 1e-2)

            JJT = J @ J.T
            dq  = J.T @ np.linalg.solve(JJT + lam**2 * np.eye(6), err)

            max_dq = np.abs(dq).max()
            if max_dq > dq_max:
                dq *= dq_max / max_dq

            q += dq
            q  = np.clip(q, q_min, q_max)

    # ------------------------------------------------------------------ #
    # Solver B — L-BFGS-B global optimizer                                #
    # ------------------------------------------------------------------ #
    def solver_sqp():
        w_posture = 1e-3

        def cost(q):
            if stop_event.is_set():
                return 0.0
            err         = pose_error(q, data_b)
            err_posture = q - q_init
            return (err @ err
                    + w_posture * (err_posture @ err_posture))

        bounds = [(lo, hi) for lo, hi in zip(q_min, q_max)]

        seeds = [q_init]
        rng   = np.random.default_rng(seed=42)
        for _ in range(5):
            seeds.append(rng.uniform(q_min, q_max))

        for q_seed in seeds:
            if stop_event.is_set():
                break

            res = minimize(
                cost, q_seed,
                method="L-BFGS-B",
                bounds=bounds,
                options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 100}
            )

            q_sol    = np.clip(res.x, q_min, q_max)
            err_norm = np.linalg.norm(pose_error(q_sol, data_b))

            with result_lock:
                if err_norm < result["err"]:
                    result["q"]   = q_sol.copy()
                    result["err"] = err_norm

            if err_norm < tol:
                stop_event.set()
                return

    # ------------------------------------------------------------------ #
    # Run both solvers concurrently                                        #
    # ------------------------------------------------------------------ #
    thread_a = threading.Thread(target=solver_dls, daemon=True)
    thread_b = threading.Thread(target=solver_sqp, daemon=True)

    thread_a.start()
    thread_b.start()

    stop_event.wait(timeout=timeout)
    stop_event.set()

    thread_a.join(timeout=0.01)
    thread_b.join(timeout=0.01)

    # Mirror solve_ik_dls return: (q, res) where res=0 means converged
    q_sol = result["q"]
    res   = 0 if result["err"] < tol else 1

    return q_sol, res
