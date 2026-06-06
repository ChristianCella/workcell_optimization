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

    q_traj    : (N, 6) joint positions in RADIANS (will be converted to degrees)
    qd_traj   : (N, 6) joint velocities in rad/s
    dt        : timestep in seconds — derived from TOPP-RA sampling frequency.
                Each MoveAbsJ segment is assigned \T:=dt so the controller
                spends exactly dt seconds on every move.
    tool_name : RAPID tool name defined on the robot (default 'tool0')
    wobj_name : RAPID work object name (default 'wobj0')
    """

    if output_dir is None:
        output_dir = os.path.join(base_dir, "workcell_optimization/results")
    os.makedirs(output_dir, exist_ok=True)

    filename   = os.path.join(output_dir, f"trajectory_{robot_to_use}_{ik_solver_to_use}.mod")

    # Convert radians → degrees for RAPID
    q_traj_deg = np.degrees(q_traj)
    n_points   = len(q_traj_deg)
    duration   = n_points * dt

    # dt comes directly from the TOPP-RA sampling frequency (1/freq).
    # \T:=dt tells the OmniCore controller that each segment must take
    # exactly dt seconds, faithfully replaying the TOPP-RA timing.
    T_segment = dt                  # seconds per move segment
    freq      = 1.0 / dt            # Hz — for comments only

    v_default = 5000   # mm/s — upper bound; overridden at runtime by \T
    v_home    = 100    # mm/s — slower for home moves

    lines = []

    # ── Module header ─────────────────────────────────────────────────────
    lines += [
        "MODULE Trajectory",
        "",
        "  ! ============================================================",
        f"  ! Auto-generated RAPID trajectory",
        f"  ! Robot   : {robot_to_use}",
        f"  ! Solver  : {ik_solver_to_use}",
        f"  ! Points  : {n_points}",
        f"  ! dt      : {dt} s  ({freq:.0f} Hz)",
        f"  ! Duration: {duration:.3f} s",
        f"  ! \\T/segment: {T_segment:.4f} s  (= 1 / {freq:.0f} Hz)",
        "  ! ============================================================",
        "",
    ]

    # ── Speed data ────────────────────────────────────────────────────────
    lines += [
        "  ! Speed data: [v_tcp (mm/s), v_ori (deg/s), v_leax, v_reax]",
        f"  LOCAL CONST speeddata v_traj := [{v_default}, 500, 5000, 1000];",
        f"  LOCAL CONST speeddata v_slow := [{v_home},    500, 5000, 1000];",
        "",
        f"  ! \\T:={T_segment:.4f} overrides v_traj at runtime and enforces",
        f"  ! the TOPP-RA segment duration (1 / {freq:.0f} Hz = {T_segment:.4f} s).",
        f"  ! z0 blend zone — fine on last point only.",
        "",
    ]

    # ── Home jointtarget ──────────────────────────────────────────────────
    home_deg = np.degrees(rob_params.home_configuration)
    home_str = ", ".join([f"{float(v):.4f}" for v in home_deg])
    lines += [
        f"  LOCAL CONST jointtarget home_jt := [[{home_str}], [9E9,9E9,9E9,9E9,9E9,9E9]];",
        "",
    ]

    # ── Waypoint declarations ─────────────────────────────────────────────
    lines.append(f"  ! Waypoint declarations ({n_points} points)")
    for i in range(n_points):
        q_str = ", ".join([f"{float(v):.4f}" for v in q_traj_deg[i]])
        lines.append(
            f"  LOCAL CONST jointtarget jt{i:05d} "
            f":= [[{q_str}], [9E9,9E9,9E9,9E9,9E9,9E9]];"
        )
    lines.append("")

    # ── Main procedure ────────────────────────────────────────────────────
    lines += [
        "  PROC main()",
        "    ! Step 1: move to home",
        f"    MoveAbsJ home_jt, v_slow, fine, {tool_name}\\WObj:={wobj_name};",
        "    WaitTime 1;",
        "",
        "    ! Step 2: move to start of trajectory (slow, exact stop)",
        f"    MoveAbsJ jt00000, v_slow, fine, {tool_name}\\WObj:={wobj_name};",
        "    WaitTime 0.5;",
        "",
        "    ! Step 3: execute time-stamped trajectory",
        f"    ! Each segment takes exactly \\T:={T_segment:.4f} s ({freq:.0f} Hz)",
        "    ! v_traj is required by the parser but overridden by \\T at runtime.",
        "    ! z0 blend zone — fine on last point for exact stop.",
    ]

    for i in range(n_points):
        zone = "fine" if i == n_points - 1 else "z0"
        lines.append(
            f"    MoveAbsJ jt{i:05d}, v_traj\\T:={T_segment:.4f}, {zone}, "
            f"{tool_name}\\WObj:={wobj_name};"
        )

    lines += [
        "",
        "    ! Step 4: return to home",
        "    WaitTime 0.5;",
        f"    MoveAbsJ home_jt, v_slow, fine, {tool_name}\\WObj:={wobj_name};",
        "",
        "  ENDPROC",
        "",
        "ENDMODULE",
    ]

    # ── Write file ────────────────────────────────────────────────────────
    with open(filename, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"RAPID module saved to : {filename}")
    print(f"  Points              : {n_points}")
    print(f"  Duration            : {duration:.3f} s")
    print(f"  Frequency           : {freq:.0f} Hz")
    print(f"  \\T per segment      : {T_segment:.4f} s")

    return filename


if __name__ == "__main__":

    results_dir = os.path.join(base_dir, "workcell_optimization/results")
    dt = 1.0 / rob_params.freq

    q_traj = np.loadtxt(
        os.path.join(results_dir, f"q_traj_{robot_to_use}_{ik_solver_to_use}.csv"),
        delimiter=",", skiprows=1
    )
    qd_traj = np.loadtxt(
        os.path.join(results_dir, f"qd_traj_{robot_to_use}_{ik_solver_to_use}.csv"),
        delimiter=",", skiprows=1
    )

    print(f"Loaded trajectory : {q_traj.shape[0]} points at {1/dt:.0f} Hz")
    print(f"Duration          : {q_traj.shape[0] * dt:.3f} s")

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