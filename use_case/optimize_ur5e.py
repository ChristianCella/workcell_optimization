#!/usr/bin/env python3
"""
CMA-ES optimization of workpiece (x, y, z) position for the UR5e robot
on the 'reconstructed' piece.

Objective function:
    - If any waypoint is unreachable OR in collision  → large penalty
    - Otherwise                                       → mean gravity torque
      across all trajectory waypoints and all joints

The optimizer converges toward the workpiece placement that:
    1. Has a fully reachable, collision-free path
    2. Minimises mean gravity loading on the joints
"""

import os
import sys
import time
import warnings
import numpy as np
import mujoco
import cma  # pip install cma

# ── project imports (same pattern as the original main.py) ──────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scene_manager')))
from create_scene import merge_robot_and_tool, inject_robot_tool_into_scene, add_instance
from config import *
rob_params = rob_par
from parameters import TestIK
ik_params = TestIK()

utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(utils_dir)
import fonts
from transformations import rotm_to_quaternion, rotm2euler, get_homogeneous_matrix
from mujoco_utils import set_body_pose, get_collisions, inverse_manipulability
from generate_path import create_path
from generate_trajectory import create_trajectory

# ── hard-wired choices ───────────────────────────────────────────────────────
ROBOT    = "ur5e"
PIECE    = "reconstructed"
TOOL     = tool_to_use          # keep whatever is in config
IK_SOLVER = "dls"

# CMA-ES search space: [x, y, z] of the piece in world frame
# Adjust these bounds to match your workcell geometry
X_BOUNDS = (-0.8,  0.8)   # metres
Y_BOUNDS = (-0.8,  0.8)
Z_BOUNDS = (-0.3,  0.5)

# Penalty returned when path is infeasible
INFEASIBLE_PENALTY = 1e6

# ── build the MuJoCo model once and reuse it ─────────────────────────────────

def build_model():
    """
    Replicates the scene-assembly steps from main.py.
    Returns (model, data, ids_dict) ready for simulation.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    robot_and_tool_file_name = f"temp_{ROBOT}_with_tool.xml"
    output_scene_filename    = "final_scene_cmaes.xml"

    _ = merge_robot_and_tool(
        robot_filename=rob_name,
        robot_folder=rob_folder,
        tool_filename=tool_name,
        base_dir=base_dir,
        output_robot_tool_filename=robot_and_tool_file_name,
    )

    merged_scene_path = inject_robot_tool_into_scene(
        robot_tool_filename=robot_and_tool_file_name,
        output_scene_filename=output_scene_filename,
        base_dir=base_dir,
        robot_folder=rob_folder,
    )

    add_instance(
        base_scene_path=merged_scene_path,
        instance_path=os.path.join(base_dir, "ur5e_utils_mujoco/pieces", pie_name),
        output_path=merged_scene_path,
        mesh_source_dir=os.path.join(base_dir, "ur5e_utils_mujoco/pieces"),
        mesh_target_dir=os.path.join(base_dir, f"ur5e_utils_mujoco/{rob_folder}/assets"),
    )

    model_path = os.path.join(base_dir, "ur5e_utils_mujoco", output_scene_filename)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Collect body / site IDs
    ids = {
        "piece":     mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{PIECE}"),
        "base":      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base"),
        "tool_base": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_base"),
        "tool_tip":  mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame"),
        "tool_site": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_site"),
    }

    # UR5e robot base (fixed, from original code)
    _, _, A_w_b = get_homogeneous_matrix(0.0, 0.0, 0.0, 0.0, 0.0, 180.0)
    set_body_pose(model, data, ids["base"], A_w_b[:3, 3], rotm_to_quaternion(A_w_b[:3, :3]))
    data.qpos[:rob_params.nu] = rob_params.home_configuration.tolist()

    # Tool geometry (UR5e fixed convention)
    _, _, A_wl3_ee = get_homogeneous_matrix(0.0, 0.1, 0.0, -90.0, 0.0, 0.0)

    if TOOL == "welding_gun":
        _, _, A_ee_t1 = get_homogeneous_matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        set_body_pose(model, data, ids["tool_base"], A_ee_t1[:3, 3], rotm_to_quaternion(A_ee_t1[:3, :3]))
        _, _, A_t1_t = get_homogeneous_matrix(0.0, -0.083033, 0.31549, 45.0, 0.0, 0.0)
    elif TOOL == "screwdriver":
        _, _, A_ee_t1 = get_homogeneous_matrix(0.0, 0.0, 0.0, 0.0, 0.0, -45.0)
        set_body_pose(model, data, ids["tool_base"], A_ee_t1[:3, 3], rotm_to_quaternion(A_ee_t1[:3, :3]))
        _, _, A_t1_t = get_homogeneous_matrix(0, -0.195, 0.028, 90.0, 0.0, 0.0)
    elif TOOL == "painting_gun":
        _, _, A_ee_t1 = get_homogeneous_matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        set_body_pose(model, data, ids["tool_base"], A_ee_t1[:3, 3], rotm_to_quaternion(A_ee_t1[:3, :3]))
        _, _, A_t1_t = get_homogeneous_matrix(0.0, 0.0, 0.21, 0.0, 0.0, 0.0)
    else:
        raise ValueError(f"Unknown tool: {TOOL}")

    A_ee_t = A_ee_t1 @ A_t1_t
    set_body_pose(model, data, ids["tool_tip"], A_ee_t[:3, 3], rotm_to_quaternion(A_ee_t[:3, :3]))

    return model, data, ids, A_w_b, A_ee_t, A_wl3_ee


def get_cartesian_path(model, data):
    """Extract ordered list of (pos, euler) from the point_*_traj bodies."""
    frame_ids = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and name.startswith("point_") and name.endswith("_traj"):
            frame_ids.append(i)

    cartesian_path = []
    for fid in frame_ids:
        pos = data.body(fid).xpos.copy()
        rot = data.body(fid).xmat.reshape(3, 3).copy()
        euler = rotm2euler(rot, degrees=False)
        cartesian_path.append((pos, euler))

    return cartesian_path


def gravity_torques(q, model, data):
    """
    Return the gravity-compensation torque vector for configuration q.
    Uses mujoco's qfrc_bias which contains Coriolis + gravity (with qvel=0
    it is purely gravity).
    """
    data.qpos[:rob_params.nu] = q
    data.qvel[:rob_params.nu] = 0.0
    mujoco.mj_forward(model, data)
    # qfrc_bias = C(q,qdot)*qdot + g(q); with qdot=0 → pure gravity
    return data.qfrc_bias[:rob_params.nu].copy()


# ── objective function ────────────────────────────────────────────────────────

def objective(xyz, model, data, ids, A_w_b, A_ee_t, A_wl3_ee):
    """
    Given a candidate workpiece position xyz = [x, y, z]:
      1. Place the piece at that position (fixed orientation from config)
      2. Solve IK for all waypoints
      3. If infeasible → return INFEASIBLE_PENALTY
      4. Else          → return mean |gravity torque| across all waypoints
    """
    x, y, z = xyz

    # Clamp to bounds (CMA-ES may propose out-of-bound candidates)
    x = float(np.clip(x, *X_BOUNDS))
    y = float(np.clip(y, *Y_BOUNDS))
    z = float(np.clip(z, *Z_BOUNDS))

    # Place the piece — keep the orientation fixed (same as original config)
    _, _, A_w_p = get_homogeneous_matrix(x, y, z, 0.0, 0.0, -90.0)
    set_body_pose(model, data, ids["piece"],
                  A_w_p[:3, 3], rotm_to_quaternion(A_w_p[:3, :3]))

    # Reset robot to home so IK starts from a consistent state
    data.qpos[:rob_params.nu] = rob_params.home_configuration.tolist()
    mujoco.mj_forward(model, data)

    # Get Cartesian path (positions depend on piece placement)
    cartesian_path = get_cartesian_path(model, data)

    if len(cartesian_path) == 0:
        return INFEASIBLE_PENALTY

    # Solve IK — suppress verbose output during optimisation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            q_path, reach, cols, _ = create_path(
                cartesian_path, model, data, rob_params,
                ids["tool_site"], A_w_b, A_ee_t, A_wl3_ee,
                save_data=False,        # never save during optimisation
            )
        except Exception:
            return INFEASIBLE_PENALTY

    # Feasibility check: any unreachable or collision → heavy penalty
    if any(r == 1 for r in reach):
        n_unreachable = sum(r == 1 for r in reach)
        # Soft penalty proportional to number of bad waypoints
        # so CMA-ES can still make progress toward feasibility
        return INFEASIBLE_PENALTY + n_unreachable * 1e4

    if any(c > 0 for c in cols):
        n_cols = sum(c > 0 for c in cols)
        return INFEASIBLE_PENALTY + n_cols * 1e4

    # Compute mean absolute gravity torque over the path
    torque_norms = []
    for q in q_path:
        tau = gravity_torques(q, model, data)
        torque_norms.append(np.mean(np.abs(tau)))

    return float(np.mean(torque_norms))


# ── CMA-ES optimisation ───────────────────────────────────────────────────────

def run_cmaes():

    print(f"\n{fonts.green}{'='*60}")
    print(" CMA-ES workpiece position optimisation")
    print(f" Robot : {ROBOT}   Piece : {PIECE}   Tool : {TOOL}")
    print(f"{'='*60}{fonts.reset}\n")

    # Build the model once
    model, data, ids, A_w_b, A_ee_t, A_wl3_ee = build_model()

    # Initial guess: original position from the main script
    x0 = np.array([0.0, 0.65, -0.3])

    # Initial standard deviation (search radius in metres)
    sigma0 = 0.15

    # CMA-ES options
    opts = cma.CMAOptions()
    opts['tolx']         = 1e-4
    opts['tolfun']       = 1e-4
    opts['maxiter']      = 100
    opts['popsize']      = 20
    opts['verbose']      = 3
    opts['bounds']       = [
        [X_BOUNDS[0], Y_BOUNDS[0], Z_BOUNDS[0]],
        [X_BOUNDS[1], Y_BOUNDS[1], Z_BOUNDS[1]],
    ]

    # Wrap objective to pass fixed arguments
    def obj_wrapper(xyz):
        val = objective(xyz, model, data, ids, A_w_b, A_ee_t, A_wl3_ee)
        feasible = val < INFEASIBLE_PENALTY
        tag = f"{fonts.green}feasible  cost={val:.4f}{fonts.reset}" if feasible \
              else f"{fonts.red}infeasible penalty={val:.0f}{fonts.reset}"
        print(f"  xyz=[{xyz[0]:+.4f}, {xyz[1]:+.4f}, {xyz[2]:+.4f}]  →  {tag}")
        return val

    t_start = time.perf_counter()
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    best_xyz   = x0.copy()
    best_cost  = np.inf

    while not es.stop():
        candidates = es.ask()
        fitnesses  = [obj_wrapper(c) for c in candidates]
        es.tell(candidates, fitnesses)
        es.disp()

        # Track best feasible solution
        for c, f in zip(candidates, fitnesses):
            if f < best_cost:
                best_cost = f
                best_xyz  = np.array(c)

    elapsed = time.perf_counter() - t_start

    print(f"\n{fonts.green}{'='*60}")
    print(" CMA-ES optimisation complete")
    print(f" Elapsed : {elapsed:.1f} s")
    print(f" Best xyz: [{best_xyz[0]:+.4f}, {best_xyz[1]:+.4f}, {best_xyz[2]:+.4f}]")
    if best_cost < INFEASIBLE_PENALTY:
        print(f" Best mean |gravity torque|: {best_cost:.4f} N·m")
    else:
        print(f" {fonts.red}No feasible solution found — try widening bounds or increasing maxiter{fonts.reset}")
    print(f"{'='*60}{fonts.reset}\n")

    return best_xyz, best_cost


# ── optional: visualise the best solution ────────────────────────────────────

def visualise_best(best_xyz):
    """
    Re-run the full pipeline with the optimised piece position and open
    the MuJoCo viewer so the user can inspect the result.
    """
    print(f"\n{fonts.green}Visualising best solution …{fonts.reset}")

    model, data, ids, A_w_b, A_ee_t, A_wl3_ee = build_model()

    x, y, z = best_xyz
    _, _, A_w_p = get_homogeneous_matrix(x, y, z, 0.0, 0.0, -90.0)
    set_body_pose(model, data, ids["piece"],
                  A_w_p[:3, 3], rotm_to_quaternion(A_w_p[:3, :3]))

    data.qpos[:rob_params.nu] = rob_params.home_configuration.tolist()
    mujoco.mj_forward(model, data)

    cartesian_path = get_cartesian_path(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        input("Press Enter to solve IK …")

        q_path, reach, cols, _ = create_path(
            cartesian_path, model, data, rob_params,
            ids["tool_site"], A_w_b, A_ee_t, A_wl3_ee,
            save_data=False,
        )

        unreachable = [i for i, v in enumerate(reach) if v == 1]
        if unreachable:
            print(f"{fonts.red}Unreachable waypoints: {unreachable}{fonts.reset}")
        else:
            print(f"{fonts.green}All waypoints reachable!{fonts.reset}")

        if any(c > 0 for c in cols):
            print(f"{fonts.red}Collisions detected at some waypoints!{fonts.reset}")
        else:
            print(f"{fonts.green}No collisions!{fonts.reset}")

        # Compute and print per-joint mean gravity torque
        torques = np.array([gravity_torques(q, model, data) for q in q_path])
        mean_tau = np.mean(np.abs(torques), axis=0)
        print("\nMean |gravity torque| per joint (N·m):")
        for j, t in enumerate(mean_tau):
            print(f"  Joint {j+1}: {t:.4f}")
        print(f"  Overall mean: {np.mean(mean_tau):.4f}")

        # Time-optimal trajectory
        q_traj, _, _, _, _ = create_trajectory(
            q_path=q_path,
            rob_params=rob_params,
            dt=1 / rob_params.freq,
            solver_wrapper="ecos",
            save_data=False,
            robot_to_use=ROBOT,
            ik_solver_to_use=IK_SOLVER,
            v_scaling=v_red_per,
            a_scaling=a_red_per,
        )

        input("Press Enter to animate the trajectory …")

        data.qpos[:rob_params.nu] = rob_params.home_configuration.tolist()
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(1.0)

        dt = 1.0 / rob_params.freq
        t0 = time.perf_counter()
        for i, q in enumerate(q_traj):
            data.qpos[:rob_params.nu] = q
            mujoco.mj_forward(model, data)
            viewer.sync()
            target = t0 + (i + 1) * dt
            sleep_t = target - time.perf_counter()
            if sleep_t > 0:
                time.sleep(sleep_t)

        input("Press Enter to exit …")

#! Tets genetic algorithm
if __name__ == "__main__":
    best_xyz, best_cost = run_cmaes()

    if best_cost < INFEASIBLE_PENALTY:
        ans = input("\nVisualize best solution? [y/N] ").strip().lower()
        if ans == "y":
            visualise_best(best_xyz)
    else:
        print("Optimisation did not converge to a feasible solution.")
        print("Suggestions:")
        print("  • Widen X_BOUNDS / Y_BOUNDS / Z_BOUNDS")
        print("  • Increase opts['maxiter']")
        print("  • Increase opts['popsize']")
        print("  • Check that the piece mesh and trajectory frames are correct")