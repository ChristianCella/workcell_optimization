import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import numpy as np
import os

#* Base directrory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

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

    # Define geometric path
    path = ta.SplineInterpolator(
        np.linspace(0, 1, len(q_path)),
        q_path
    )

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