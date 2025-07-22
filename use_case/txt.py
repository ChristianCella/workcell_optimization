#!/usr/bin/env python3
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

import tensorflow as tf
tf.random.set_seed(444)

import cma  # pycma
import mujoco
import mujoco.viewer

# ───── Helpers ─────────────────────────────────────────────────────────────────

def euler_to_quaternion(roll, pitch, yaw, degrees=False):
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
    x, y, z, w = r.as_quat()
    return [w, x, y, z]

def set_body_pose(model, data, body_id, pos, quat_wxyz):
    model.body_pos[body_id]  = pos
    model.body_quat[body_id] = quat_wxyz
    mujoco.mj_forward(model, data)

def setup_target_frames(model, data, ref_body_ids, target_poses):
    for bid, (pos, quat_xyzw) in zip(ref_body_ids, target_poses):
        set_body_pose(
            model, data, bid,
            pos,
            [quat_xyzw[3], quat_xyzw[0],
             quat_xyzw[1], quat_xyzw[2]]
        )
    mujoco.mj_forward(model, data)

def ik_tool_site(model, data, tool_site_id, target_pos, target_quat_xyzw,
                 max_iters=200, tol=1e-4):
    """Simple damped‐least‐squares IK for a 6‐DOF chain."""
    q = data.qpos[:6].copy()
    for _ in range(max_iters):
        mujoco.mj_forward(model, data)

        # position error
        p_err = target_pos - data.site_xpos[tool_site_id]
        # orientation error via rotvec
        mat = data.site_xmat[tool_site_id].reshape(3, 3)
        current = R.from_matrix(mat)
        target  = R.from_quat(target_quat_xyzw)
        r_err   = (target * current.inv()).as_rotvec()

        err = np.hstack([p_err, r_err])
        if np.linalg.norm(err) < tol:
            break

        # Jacobian
        Jp = np.zeros((3, model.nv))
        Jr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, Jp, Jr, tool_site_id)
        J6 = np.vstack([Jp, Jr])[:, :6]

        dq = np.linalg.pinv(J6, rcond=1e-4) @ err
        q += dq
        data.qpos[:6] = q

    return q

# ───── Build the simulator wrapper ─────────────────────────────────────────────

def make_simulator(xml_path, pieces_target_poses, wrench_world):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")
    ref_body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"reference_target_1"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"reference_target_2")
    ]
    tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")

    def run_simulation(params: np.ndarray) -> float:
        x_base, y_base = float(params[0]), float(params[1])

        model.body_pos[base_body_id]  = [x_base, y_base, 0.0]
        model.body_quat[base_body_id] = euler_to_quaternion(0, 0, 0, True)
        model.body_pos[tool_body_id]  = [0.0, 0.0, 0.0]
        model.body_quat[tool_body_id] = euler_to_quaternion(0, 0, 0, True)
        mujoco.mj_forward(model, data)

        setup_target_frames(model, data, ref_body_ids, pieces_target_poses)

        norms = []
        for i, (pos, quat) in enumerate(pieces_target_poses):
            print(f"      → solving IK for target frame {i}, pose {pos}, {quat}")
            q_sol = ik_tool_site(model, data, tool_site_id, pos, quat)
            data.qpos[:6] = q_sol
            mujoco.mj_forward(model, data)

            Jp = np.zeros((3, model.nv))
            Jr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, Jp, Jr, tool_site_id)
            J6 = np.vstack([Jp, Jr])[:, :6]

            tau_g   = data.qfrc_bias[:6]
            tau_ext = J6.T @ wrench_world
            tau_tot = tau_g + tau_ext
            norms.append(np.linalg.norm(tau_tot))

        fitness = float(np.mean(norms))
        print(f"      ← simulation fitness: {fitness:.6f}")
        return fitness

    return run_simulation, model, data, base_body_id

# ───── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    print(f"Base directory: {base_dir}")
    sys.path.append(base_dir)
    xml_path = os.path.join(base_dir, "ur5e_utils_mujoco/scene.xml")

    pieces_target_poses = [
        (np.array([0.2, -0.2, 0.2]), R.from_euler('xyz',[180,0,0],True).as_quat()),
        (np.array([0.3,  0.1, 0.2]), R.from_euler('xyz',[180,0,0],True).as_quat()),
    ]
    f = np.array([0.0, 0.0, 10.0])
    m = np.zeros(3)
    wrench_world = np.hstack([f, m])

    run_sim, model, data, base_body_id = make_simulator(
        xml_path, pieces_target_poses, wrench_world
    )

    dim      = 2
    x0       = np.zeros(dim)
    sigma0   = 0.5
    popsize  = 20
    max_gens = 200

    opts = {
        "popsize": popsize,
        "bounds": [[-0.5, -0.5], [ 0.5,  0.5]],
        "verb_disp": 0,
    }

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        input("Press Enter to start optimization…")

        for gen in range(max_gens):
            print(f"\nGeneration {gen}")
            sols = es.ask()

            fitnesses = []
            for idx, sol in enumerate(sols):
                print(f"    • chromosome {idx}: {sol}")
                fitnesses.append(run_sim(sol))

            es.tell(sols, fitnesses)
            es.logger.add()
            es.disp()

            # 1) move the base to the best
            i_best    = int(np.argmin(fitnesses))
            x_b, y_b  = sols[i_best][:2]
            model.body_pos[base_body_id]  = [x_b, y_b, 0.0]
            model.body_quat[base_body_id] = euler_to_quaternion(0, 0, 0, True)
            mujoco.mj_forward(model, data)

            # 2) **for each** target frame, re-run IK & display
            for idx, (pos, quat) in enumerate(pieces_target_poses):
                print(f"   ▶ visualizing frame {idx}")
                q_vis = ik_tool_site(model, data, 
                                     mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_site"),
                                     pos, quat)
                data.qpos[:6] = q_vis
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.2)

            if es.stop():
                break

        res = es.result
        print("\nOptimization terminated:")
        print("  best solution:", res.xbest)
        print(f"  best fitness : {res.fbest:.6f}")

        input("Done — press Enter to exit…")
