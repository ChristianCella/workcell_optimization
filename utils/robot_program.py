import numpy as np
import os
import sys

#* Base directory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

#* Directory for config params
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scene_manager')))
from config import *
rob_params = rob_par

def generate_ur_script(
    q_traj,
    dt=0.002,
    lookahead_time=0.1,
    gain=300,
    robot_to_use="ur5e",
    ik_solver_to_use="dls",
    output_dir=None
):
    """
    Generate a URScript .script file from a joint trajectory.

    q_traj  : (N, 6) joint positions in radians
    dt      : timestep in seconds (must match TOPP-RA sampling, e.g. 1/500)
    lookahead_time : servoj lookahead time in seconds (0.03 - 0.2)
    gain    : servoj gain (100 - 500)
    """

    if output_dir is None:
        output_dir = os.path.join(base_dir, "workcell_optimization/results")
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, f"trajectory_{robot_to_use}_{ik_solver_to_use}.script")

    n_points = len(q_traj)
    duration = n_points * dt

    # UR5e rated safety caps — servoj never actually uses these
    # for motion profiling, they are just hard limits
    a_safe = 1.4   # rad/s^2  (UR5e max is ~8.0, use conservative value)
    v_safe = 1.05  # rad/s    (UR5e max is ~3.14, use conservative value)

    lines = []

    # ── Header ──────────────────────────────────────────────────────────────
    lines.append(f"# ============================================================")
    lines.append(f"# Auto-generated URScript trajectory")
    lines.append(f"# Robot   : {robot_to_use}")
    lines.append(f"# Solver  : {ik_solver_to_use}")
    lines.append(f"# Points  : {n_points}")
    lines.append(f"# dt      : {dt} s  ({1/dt:.0f} Hz)")
    lines.append(f"# Duration: {duration:.3f} s")
    lines.append(f"# lookahead_time: {lookahead_time}")
    lines.append(f"# gain          : {gain}")
    lines.append(f"# ============================================================")
    lines.append("")

    # ── Program ─────────────────────────────────────────────────────────────
    lines.append("def trajectory():")
    lines.append("")

    # ── 1. Move safely to home first ─────────────────────────────────────
    home = rob_params.home_configuration
    home_str = ", ".join([f"{float(v):.6f}" for v in home])
    lines.append(f"  # Step 1: move to home configuration")
    lines.append(f"  movej([{home_str}], a=0.5, v=0.3, r=0)")
    lines.append(f"  sleep(1.0)")
    lines.append("")

    # ── 2. Move to first trajectory waypoint ─────────────────────────────
    q0 = q_traj[0]
    q0_str = ", ".join([f"{float(v):.6f}" for v in q0])
    lines.append(f"  # Step 2: move to start of trajectory")
    lines.append(f"  movej([{q0_str}], a=0.5, v=0.3, r=0)")
    lines.append(f"  sleep(0.5)")
    lines.append("")

    # ── 3. Execute trajectory via servoj ─────────────────────────────────
    lines.append(f"  # Step 3: execute trajectory via servoj at {1/dt:.0f} Hz")
    lines.append(f"  # The actual motion profile is defined by the")
    lines.append(f"  # sequence of positions — TOPP-RA already respects")
    lines.append(f"  # joint velocity and acceleration limits.")
    lines.append(f"  # a and v below are safety caps only.")
    lines.append("")

    for i in range(n_points):
        q = q_traj[i]
        q_str = ", ".join([f"{float(v):.6f}" for v in q])
        lines.append(
            f"  servoj([{q_str}], "
            f"a={a_safe}, "
            f"v={v_safe}, "
            f"t={dt:.4f}, "
            f"lookahead_time={lookahead_time}, "
            f"gain={gain})"
        )

    lines.append("")

    # ── 4. Stop servo and return home ─────────────────────────────────────
    lines.append(f"  # Step 4: stop servo motion cleanly")
    lines.append(f"  stopj(a=1.0)")
    lines.append(f"  sleep(0.5)")
    lines.append("")
    lines.append(f"  # Step 5: return to home configuration")
    lines.append(f"  movej([{home_str}], a=0.5, v=0.3, r=0)")
    lines.append("")
    lines.append("end")
    lines.append("")
    lines.append("# Entry point")
    lines.append("trajectory()")

    # ── Write file ───────────────────────────────────────────────────────
    script_content = "\n".join(lines)
    with open(filename, "w") as f:
        f.write(script_content)

    print(f"URScript saved to  : {filename}")
    print(f"  Points           : {n_points}")
    print(f"  Duration         : {duration:.3f} s")
    print(f"  Frequency        : {1/dt:.0f} Hz")
    print(f"  lookahead_time   : {lookahead_time}")
    print(f"  gain             : {gain}")

    return filename


if __name__ == "__main__":

    results_dir = os.path.join(base_dir, "workcell_optimization/results")
    dt = 1.0 / rob_params.freq

    # Load trajectory
    q_traj = np.loadtxt(
        os.path.join(results_dir, f"q_traj_{robot_to_use}_{ik_solver_to_use}.csv"),
        delimiter=",",
        skiprows=1
    )

    print(f"Loaded trajectory: {q_traj.shape[0]} points at {1/dt:.0f} Hz")
    print(f"Duration: {q_traj.shape[0] * dt:.3f} s")

    generate_ur_script(
        q_traj=q_traj,
        dt=dt,
        lookahead_time=0.1,
        gain=300,
        robot_to_use=robot_to_use,
        ik_solver_to_use=ik_solver_to_use,
        output_dir=results_dir
    )