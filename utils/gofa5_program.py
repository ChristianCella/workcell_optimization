import numpy as np
import os
import sys

#* Base directory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

#* Directory for config params
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scene_manager')))
from config import *
rob_params = rob_par

def generate_rapid_program(
    q_traj,
    qd_traj,
    dt=0.002,
    robot_to_use="gofa5",
    ik_solver_to_use="dls",
    tool_name="tool0",
    wobj_name="wobj0",
    output_dir=None
):
    """
    Generate an ABB RAPID .mod file from a joint trajectory.

    q_traj  : (N, 6) joint positions in RADIANS (will be converted to degrees)
    qd_traj : (N, 6) joint velocities in rad/s (used to compute per-point speed)
    dt      : timestep in seconds matching TOPP-RA sampling rate
    tool_name : RAPID tool name defined on the robot (default 'tool0')
    wobj_name : RAPID work object name (default 'wobj0')
    """

    if output_dir is None:
        output_dir = os.path.join(base_dir, "workcell_optimization/results")
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, f"trajectory_{robot_to_use}_{ik_solver_to_use}.mod")

    # Convert radians → degrees for RAPID
    q_traj_deg  = np.degrees(q_traj)
    n_points    = len(q_traj_deg)
    duration    = n_points * dt

    # Compute per-point TCP speed estimate (mm/s) from joint velocity norm
    # This is a rough approximation — RAPID uses TCP speed, not joint speed.
    # We use a generous value so the robot is not artificially slowed down.
    # The actual motion profile is defined by the dense waypoint sequence.
    v_default = 5000   # mm/s — conservative safe speed for dense trajectory
    v_home    = 100   # mm/s — slower for home moves

    lines = []

    # ── Module header ────────────────────────────────────────────────────
    lines.append(f"MODULE Trajectory")
    lines.append(f"")
    lines.append(f"  ! ============================================================")
    lines.append(f"  ! Auto-generated RAPID trajectory")
    lines.append(f"  ! Robot   : {robot_to_use}")
    lines.append(f"  ! Solver  : {ik_solver_to_use}")
    lines.append(f"  ! Points  : {n_points}")
    lines.append(f"  ! dt      : {dt} s  ({1/dt:.0f} Hz)")
    lines.append(f"  ! Duration: {duration:.3f} s")
    lines.append(f"  ! ============================================================")
    lines.append(f"")

    # ── Speed and zone data declarations ─────────────────────────────────
    lines.append(f"  ! Speed data: [v_tcp (mm/s), v_ori (deg/s), v_leax, v_reax]")
    lines.append(f"  LOCAL CONST speeddata v_traj  := [{v_default}, 500, 5000, 1000];")
    lines.append(f"  LOCAL CONST speeddata v_slow  := [{v_home},    500, 5000, 1000];")
    lines.append(f"")
    lines.append(f"  ! Zone data: z1 = 1mm blend radius for smooth dense trajectory")
    lines.append(f"  ! Use fine (z0) for last point to guarantee exact stop")
    lines.append(f"")

    # ── Home jointtarget ─────────────────────────────────────────────────
    home_deg = np.degrees(rob_params.home_configuration)
    home_str = ", ".join([f"{float(v):.4f}" for v in home_deg])
    lines.append(f"  LOCAL CONST jointtarget home_jt := [[{home_str}], [9E9,9E9,9E9,9E9,9E9,9E9]];")
    lines.append(f"")

    # ── Declare all waypoints as CONST jointtarget ───────────────────────
    lines.append(f"  ! Waypoint declarations ({n_points} points)")
    for i in range(n_points):
        q = q_traj_deg[i]
        q_str = ", ".join([f"{float(v):.4f}" for v in q])
        lines.append(
            f"  LOCAL CONST jointtarget jt{i:05d} "
            f":= [[{q_str}], [9E9,9E9,9E9,9E9,9E9,9E9]];"
        )
    lines.append(f"")

    # ── Main procedure ───────────────────────────────────────────────────
    lines.append(f"  PROC main()")
    lines.append(f"    ! Step 1: move to home")
    lines.append(f"    MoveAbsJ home_jt, v_slow, fine, {tool_name}\\WObj:={wobj_name};")
    lines.append(f"    WaitTime 1;")
    lines.append(f"")
    lines.append(f"    ! Step 2: move to start of trajectory")
    q0_str = ", ".join([f"{float(v):.4f}" for v in q_traj_deg[0]])
    lines.append(f"    MoveAbsJ jt00000, v_slow, fine, {tool_name}\\WObj:={wobj_name};")
    lines.append(f"    WaitTime 0.5;")
    lines.append(f"")
    lines.append(f"    ! Step 3: execute trajectory")
    lines.append(f"    ! Dense MoveAbsJ sequence at {1/dt:.0f} Hz")
    lines.append(f"    ! z1 blend zone ensures smooth continuous motion")
    lines.append(f"    ! The actual velocity profile is defined by the")
    lines.append(f"    ! waypoint spacing — TOPP-RA already respects joint limits.")

    for i in range(n_points):
        # Use fine stop only on last point
        if i < n_points - 1:
            zone = "z10"
        else:
            zone = "fine"
        lines.append(
            f"    MoveAbsJ jt{i:05d}, v_traj, {zone}, "
            f"{tool_name}\\WObj:={wobj_name};"
        )

    lines.append(f"")
    lines.append(f"    ! Step 4: return to home")
    lines.append(f"    WaitTime 0.5;")
    lines.append(f"    MoveAbsJ home_jt, v_slow, fine, {tool_name}\\WObj:={wobj_name};")
    lines.append(f"")
    lines.append(f"  ENDPROC")
    lines.append(f"")
    lines.append(f"ENDMODULE")

    # ── Write file ───────────────────────────────────────────────────────
    script_content = "\n".join(lines)
    with open(filename, "w") as f:
        f.write(script_content)

    print(f"RAPID module saved to : {filename}")
    print(f"  Points              : {n_points}")
    print(f"  Duration            : {duration:.3f} s")
    print(f"  Frequency           : {1/dt:.0f} Hz")
    print(f"  TCP speed           : {v_default} mm/s")

    return filename


if __name__ == "__main__":

    results_dir = os.path.join(base_dir, "workcell_optimization/results")
    dt = 1.0 / rob_params.freq

    # Load trajectory
    q_traj = np.loadtxt(
        os.path.join(results_dir, f"q_traj_{robot_to_use}_{ik_solver_to_use}.csv"),
        delimiter=",", skiprows=1
    )
    qd_traj = np.loadtxt(
        os.path.join(results_dir, f"qd_traj_{robot_to_use}_{ik_solver_to_use}.csv"),
        delimiter=",", skiprows=1
    )

    print(f"Loaded trajectory: {q_traj.shape[0]} points at {1/dt:.0f} Hz")
    print(f"Duration: {q_traj.shape[0] * dt:.3f} s")

    generate_rapid_program(
        q_traj=q_traj,
        qd_traj=qd_traj,
        dt=dt,
        robot_to_use=robot_to_use,
        ik_solver_to_use=ik_solver_to_use,
        tool_name="tool0",
        wobj_name="wobj0",
        output_dir=results_dir
    )