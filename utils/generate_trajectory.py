import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import numpy as np
import os, sys
import mujoco
from mujoco_utils import get_collisions

#* Base directrory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

#! TOPPRA
def create_trajectory(
    q_path,
    rob_params,
    dt=0.002,
    solver_wrapper="ecos",
    save_data=False,
    robot_to_use=None,
    ik_solver_to_use=None,
    v_scaling=1.0,
    a_scaling=1.0
):

    q_path = np.asarray(q_path, dtype=np.float64)

    q_dot_max = np.asarray(rob_params.q_dot_max, dtype=np.float64)
    q_ddot_max = np.asarray(rob_params.q_ddot_max, dtype=np.float64)

    # Replace np.linspace with joint-space arc length
    dq = np.diff(q_path, axis=0)
    seg_lengths = np.linalg.norm(dq, axis=1)
    arc_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])
    arc_lengths /= arc_lengths[-1]  # normalize to [0, 1]

    path = ta.SplineInterpolator(arc_lengths, q_path)

    '''
    # Define geometric path
    path = ta.SplineInterpolator(
        np.linspace(0, 1, len(q_path)),
        q_path,
        bc_type="clamped"
    )
    '''

    # Define velocity and acceleration limits
    vlim = v_scaling * np.vstack([-q_dot_max, q_dot_max]).T.astype(np.float64)
    alim = a_scaling * np.vstack([-q_ddot_max, q_ddot_max]).T.astype(np.float64)

    pc_vel = constraint.JointVelocityConstraint(vlim)
    pc_acc = constraint.JointAccelerationConstraint(alim)

    # Solve TOPPRA
    instance = algo.TOPPRA(
        [pc_vel, pc_acc],
        path,
        solver_wrapper=solver_wrapper
    )

    jnt_traj = instance.compute_trajectory()

    if jnt_traj is None:
        raise RuntimeError("TOPPRA failed to compute a valid trajectory.")

    # Sample trajectory
    duration = jnt_traj.duration
    t_fine = np.arange(0, duration + dt, dt)

    q_traj = jnt_traj(t_fine)
    qd_traj = jnt_traj(t_fine, 1)
    qdd_traj = jnt_traj(t_fine, 2)

    print(f"Trajectory duration : {duration:.3f} s")
    print(f"Trajectory points   : {len(t_fine)} samples at {1 / dt:.0f} Hz")
    print(f"q_traj shape        : {q_traj.shape}")

    # Optional save
    results_dir = os.path.join(base_dir, "workcell_optimization/results")
    if save_data:
        if results_dir is None:
            raise ValueError("results_dir must be provided when save_data=True.")

        os.makedirs(results_dir, exist_ok=True)

        suffix = ""
        if robot_to_use is not None and ik_solver_to_use is not None:
            suffix = f"_{robot_to_use}_{ik_solver_to_use}"

        header = ",".join([f"q{i+1}" for i in range(q_path.shape[1])])

        np.savetxt(
            os.path.join(results_dir, f"q_traj{suffix}.csv"),
            q_traj,
            delimiter=",",
            header=header,
            comments=""
        )

        np.savetxt(
            os.path.join(results_dir, f"qd_traj{suffix}.csv"),
            qd_traj,
            delimiter=",",
            header=header,
            comments=""
        )

        np.savetxt(
            os.path.join(results_dir, f"qdd_traj{suffix}.csv"),
            qdd_traj,
            delimiter=",",
            header=header,
            comments=""
        )

        print(f"Trajectories saved to {results_dir}")

    return q_traj, qd_traj, qdd_traj, t_fine, duration

import numpy as np

#! TOTG
def compute_time_stamps_totg(q_path, q_dot_max, q_ddot_max, v_scaling=1.0, a_scaling=1.0, dt=0.002):
    """
    Time Optimal Trajectory Generation with trapezoidal velocity profiles.
    Replicates MoveIt's IterativeParabolicTimeParameterization logic.
    """
    q_path = np.asarray(q_path, dtype=np.float64)
    n = len(q_path)
    vmax = v_scaling * np.asarray(q_dot_max)
    amax = a_scaling * np.asarray(q_ddot_max)

    # Step 1: compute per-segment joint displacements
    dq = np.diff(q_path, axis=0)  # (n-1, njoints)

    # Step 2: compute minimum time for each segment based on velocity and acceleration limits
    t_seg = np.zeros(n - 1)
    for i in range(n - 1):
        t_v = np.max(np.abs(dq[i]) / vmax)       # time limited by velocity
        t_a = np.max(np.sqrt(np.abs(dq[i]) / (0.5 * amax)))  # time limited by acceleration
        t_seg[i] = max(t_v, t_a, 1e-6)

    # Step 3: forward pass - enforce acceleration limits between segments
    for i in range(1, n - 1):
        v_prev = dq[i-1] / t_seg[i-1]
        v_curr = dq[i]   / t_seg[i]
        dv = np.abs(v_curr - v_prev)
        t_acc = np.max(dv / amax)
        if t_acc > t_seg[i]:
            t_seg[i] = t_acc

    # Step 4: backward pass
    for i in range(n - 3, -1, -1):
        v_curr = dq[i]   / t_seg[i]
        v_next = dq[i+1] / t_seg[i+1]
        dv = np.abs(v_next - v_curr)
        t_acc = np.max(dv / amax)
        if t_acc > t_seg[i]:
            t_seg[i] = t_acc

    # Step 5: build time stamps
    t_waypoints = np.concatenate([[0], np.cumsum(t_seg)])
    duration = t_waypoints[-1]
    print(f"Trajectory duration : {duration:.3f} s")

    # Step 6: sample at uniform dt using cubic spline
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(t_waypoints, q_path)
    t_fine = np.arange(0, duration + dt, dt)
    q_traj  = cs(t_fine)
    qd_traj = cs(t_fine, 1)
    qdd_traj = cs(t_fine, 2)

    print(f"Trajectory points   : {len(t_fine)} samples at {1/dt:.0f} Hz")
    print(f"q_traj shape        : {q_traj.shape}")

    return q_traj, qd_traj, qdd_traj, t_fine, duration

