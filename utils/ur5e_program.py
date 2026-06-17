import numpy as np
import os
import sys
import re

#* Base directory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

#* Directory for config params
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scene_manager')))
from config import *
rob_params = rob_par


def generate_ur_script(
    robot_to_use="ur5e",
    ik_solver_to_use="dls",
    dt=0.002,
    lookahead_time=0.1,
    gain=300,
    output_dir=None,
    results_dir=None
):
    """
    Scans the results directory for all trajectory files matching
    q_traj_{robot_to_use}_*.csv, loads them in order, and generates
    a single URScript .script file that executes them sequentially.

    Each trajectory is preceded by a movej to its first waypoint and
    followed by a 1-second sleep after the last servoj command.
    """

    if results_dir is None:
        results_dir = os.path.join(base_dir, "workcell_optimization/results")
    if output_dir is None:
        output_dir = results_dir
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Scan results directory for matching trajectory files                 #
    # Pattern: q_traj_{robot_to_use}_{number}.csv                         #
    # Excludes any file starting with 'prog'                              #
    # ------------------------------------------------------------------ #
    pattern = re.compile(rf"^q_traj_{re.escape(robot_to_use)}_(\d+)\.csv$")

    traj_files = []
    for fname in os.listdir(results_dir):
        if fname.startswith("prog"):
            continue
        match = pattern.match(fname)
        if match:
            traj_idx = int(match.group(1))
            traj_files.append((traj_idx, os.path.join(results_dir, fname)))

    if not traj_files:
        raise FileNotFoundError(
            f"No trajectory files found matching 'q_traj_{robot_to_use}_*.csv' "
            f"in {results_dir}"
        )

    # Sort by trajectory index
    traj_files.sort(key=lambda x: x[0])
    print(f"Found {len(traj_files)} trajectory file(s) for robot '{robot_to_use}':")
    for idx, fpath in traj_files:
        print(f"  [{idx}] {os.path.basename(fpath)}")

    # ------------------------------------------------------------------ #
    # Load all trajectories                                                #
    # ------------------------------------------------------------------ #
    trajectories = []
    for idx, fpath in traj_files:
        q_traj = np.loadtxt(fpath, delimiter=",", skiprows=1)
        if q_traj.ndim == 1:
            q_traj = q_traj.reshape(1, -1)
        trajectories.append((idx, q_traj))
        print(f"  Loaded trajectory {idx}: {q_traj.shape[0]} points")

    # ------------------------------------------------------------------ #
    # Build URScript                                                       #
    # ------------------------------------------------------------------ #
    filename = os.path.join(output_dir, f"trajectory_{robot_to_use}_{ik_solver_to_use}.script")

    total_points = sum(q.shape[0] for _, q in trajectories)
    total_duration = total_points * dt

    a_safe = 1.4
    v_safe = 1.05

    lines = []

    # ── Header ──────────────────────────────────────────────────────────
    lines.append(f"# ============================================================")
    lines.append(f"# Auto-generated URScript trajectory")
    lines.append(f"# Robot      : {robot_to_use}")
    lines.append(f"# Solver     : {ik_solver_to_use}")
    lines.append(f"# Trajectories: {len(trajectories)}")
    lines.append(f"# Total pts  : {total_points}")
    lines.append(f"# dt         : {dt} s  ({1/dt:.0f} Hz)")
    lines.append(f"# Duration   : {total_duration:.3f} s")
    lines.append(f"# lookahead_time: {lookahead_time}")
    lines.append(f"# gain          : {gain}")
    lines.append(f"# ============================================================")
    lines.append("")

    # ── Program ─────────────────────────────────────────────────────────
    lines.append("def trajectory():")
    lines.append("")

    # ── Step 1: move to home ─────────────────────────────────────────────
    home = rob_params.home_configuration
    home_str = ", ".join([f"{float(v):.6f}" for v in home])
    lines.append(f"  # Step 1: move to home configuration")
    lines.append(f"  movej([{home_str}], a=0.5, v=0.3, r=0)")
    lines.append(f"  sleep(1.0)")
    lines.append("")

    # ── Steps 2+: one block per trajectory ───────────────────────────────
    for traj_idx, (file_idx, q_traj) in enumerate(trajectories):
        n_points = q_traj.shape[0]
        duration = n_points * dt

        lines.append(f"  # ── Trajectory {file_idx} ──────────────────────────────")
        lines.append(f"  # Points: {n_points}  Duration: {duration:.3f} s")
        lines.append("")

        # movej to home and then first waypoint of this trajectory
        q0     = q_traj[0]
        q0_str = ", ".join([f"{float(v):.6f}" for v in q0])
        lines.append(f"  # Step 1: move to home configuration")
        lines.append(f"  movej([{home_str}], a=0.5, v=0.3, r=0)")
        lines.append(f"  sleep(1.0)")
        lines.append("")
        lines.append(f"  # Move to first waypoint of trajectory {file_idx}")
        lines.append(f"  movej([{q0_str}], a=0.5, v=0.3, r=0)")
        lines.append(f"  sleep(0.5)")
        lines.append("")

        # servoj commands for all points
        lines.append(f"  # Execute trajectory {file_idx} via servoj at {1/dt:.0f} Hz")
        for i in range(n_points):
            q     = q_traj[i]
            q_str = ", ".join([f"{float(v):.6f}" for v in q])
            lines.append(
                f"  servoj([{q_str}], "
                f"a={a_safe}, "
                f"v={v_safe}, "
                f"t={dt:.4f}, "
                f"lookahead_time={lookahead_time}, "
                f"gain={gain})"
            )

        # 1-second wait after last servoj of this trajectory
        lines.append(f"  sleep(1.0)  # wait after end of trajectory {file_idx}")
        lines.append("")

    # ── Final: stop and return home ───────────────────────────────────────
    lines.append(f"  # Stop servo motion cleanly")
    lines.append(f"  stopj(a=1.0)")
    lines.append(f"  sleep(0.5)")
    lines.append("")
    lines.append(f"  # Return to home configuration")
    lines.append(f"  movej([{home_str}], a=0.5, v=0.3, r=0)")
    lines.append("")
    lines.append("end")
    lines.append("")
    lines.append("# Entry point")
    lines.append("trajectory()")

    # ── Write file ────────────────────────────────────────────────────────
    script_content = "\n".join(lines)
    with open(filename, "w", encoding="utf-8") as f:   # <-- add encoding="utf-8"
        f.write(script_content)

    print(f"\nURScript saved to  : {filename}")
    print(f"  Trajectories     : {len(trajectories)}")
    print(f"  Total points     : {total_points}")
    print(f"  Total duration   : {total_duration:.3f} s")
    print(f"  Frequency        : {1/dt:.0f} Hz")
    print(f"  lookahead_time   : {lookahead_time}")
    print(f"  gain             : {gain}")

    return filename


if __name__ == "__main__":

    results_dir = os.path.join(base_dir, "workcell_optimization/results")
    dt = 1.0 / rob_params.freq

    generate_ur_script(
        robot_to_use=robot_to_use,
        ik_solver_to_use=ik_solver_to_use,
        dt=dt,
        lookahead_time=0.1,
        gain=300,
        output_dir=results_dir,
        results_dir=results_dir
    )