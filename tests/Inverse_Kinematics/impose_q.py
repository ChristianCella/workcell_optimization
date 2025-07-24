#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
import time
import sys
import os
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(__file__))
from inference_parallelized import FastIKFlowSolver, solve_ik_fast, solve_ik_rotational_sweep


def euler_to_quaternion(roll, pitch, yaw, degrees=False):
    from scipy.spatial.transform import Rotation as R
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
    q = r.as_quat()
    return [q[3], q[0], q[1], q[2]]  # [w, x, y, z]

def rotm_to_quaternion(rotm):
    from scipy.spatial.transform import Rotation as R
    q = R.from_matrix(rotm).as_quat()
    return [q[3], q[0], q[1], q[2]]  # [w, x, y, z]

def set_body_pose(model, data, body_id, pos, quat):
    model.body_pos[body_id] = pos
    model.body_quat[body_id] = quat
    mujoco.mj_forward(model, data)

def main():
    # variables
    verbose = True
    use_ikflow = False  # set to False to test a hard-coded joint configuration 
    show_pose_duration = 1.5  # seconds to show each pose

    # Path setup  
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    sys.path.append(base_dir)
    xml_path = os.path.join(base_dir, "ur5e_utils_mujoco/scene.xml")

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Get body/site IDs
    ee_site_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    piece_body_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "reference_target_1")
    base_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    tool_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")
    screwdriver_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "screw_top")

    #! Move bodies in space (simulation of 'changing' the layout)

    # Set robot base (matrix A^w_b)
    t_w_b = np.array([0, 0, 0])
    R_w_b = R.from_euler('xyz', [np.radians(0), np.radians(0), np.radians(0)], degrees=False).as_matrix()
    A_w_b = np.eye(4)
    A_w_b[:3, 3] = t_w_b
    A_w_b[:3, :3] = R_w_b

    model.body_pos[base_body_id] = A_w_b[:3, 3]
    model.body_quat[base_body_id] = rotm_to_quaternion(A_w_b[:3, :3])

    # Set the farme 'screw_top to a new pose wrt flange' and move the screwdriver there
    t_ee_t1 = np.array([0, 0.1, -0.1])
    R_ee_t1 = R.from_euler('xyz', [np.radians(0), np.radians(0), np.radians(0)], degrees=False).as_matrix()
    A_ee_t1 = np.eye(4)
    A_ee_t1[:3, 3] = t_ee_t1
    A_ee_t1[:3, :3] = R_ee_t1

    model.body_pos[screwdriver_body_id] = A_ee_t1[:3, 3]
    model.body_quat[screwdriver_body_id] = rotm_to_quaternion(A_ee_t1[:3, :3])

    # Fixed transformation 'tool top (t1) => tool tip (t)'
    t_t1_t = np.array([0, 0.0, 0.26])
    R_t1_t = R.from_euler('xyz', [np.radians(0), np.radians(0), np.radians(0)], degrees=False).as_matrix()
    A_t1_t = np.eye(4)
    A_t1_t[:3, 3] = t_t1_t
    A_t1_t[:3, :3] = R_t1_t

    # Update the position of the tool tip
    A_ee_t = A_ee_t1 @ A_t1_t  # combine the two transformations

    model.body_pos[tool_body_id] = A_ee_t[:3, 3]
    model.body_quat[tool_body_id] = rotm_to_quaternion(A_ee_t[:3, :3])

    # Piece in the world (define A^w_p) => this is also used to put teh frame in space  
    theta_w_p_0 = np.radians(0)  # initial angle
    t_w_p = np.array([0.3, -0.174, 0.7])
    R_w_p = R.from_euler('xyz', [np.radians(0), 0, theta_w_p_0], degrees=False).as_matrix()
    A_w_p = np.eye(4)
    A_w_p[:3, 3] = t_w_p
    A_w_p[:3, :3] = R_w_p

    # End-effector with respect to wrist3
    t_wl3_ee = np.array([0, 0.1, 0])
    R_wl3_e = R.from_euler('xyz', [np.radians(-90), 0, 0], degrees=False).as_matrix()
    A_wl3_ee = np.eye(4)
    A_wl3_ee[:3, 3] = t_wl3_ee
    A_wl3_ee[:3, :3] = R_wl3_e

    #! Make inference on the nornmalizing flow (ikflow)
    N = 20
    N_disc = 18
    fast_ik_solver = FastIKFlowSolver()    
    
    counter_start_inference = time.time()

    sols_ok, fk_ok = [], []
    for i in range(N_disc): # 0, 1, 2, ... N_disc-1
        R_w_p_rotated = R.from_euler('xyz', [np.radians(0), 0, theta_w_p_0 + i * 2 * np.pi / N_disc], degrees=False).as_matrix()
        A_w_p_rotated = np.eye(4)
        A_w_p_rotated[:3, 3] = t_w_p
        A_w_p_rotated[:3, :3] = R_w_p_rotated
        A_b_wl3 = np.linalg.inv(A_w_b) @ A_w_p_rotated @ np.linalg.inv(A_ee_t)@ np.linalg.inv(A_wl3_ee)
        quat_pose = rotm_to_quaternion(A_b_wl3[:3, :3])
        target = np.array([
            A_b_wl3[0, 3], A_b_wl3[1, 3], A_b_wl3[2, 3],   # position
            quat_pose[0], quat_pose[1], quat_pose[2], quat_pose[3]  # quaternion
        ], dtype=np.float64)
        tgt_tensor = torch.from_numpy(target.astype(np.float32))
        sols_disc, fk_disc = solve_ik_fast(tgt_tensor, N=N, fast_solver=fast_ik_solver)
        sols_ok.append(sols_disc)
        fk_ok.append(fk_disc)

    counter_end_inference = time.time()
    if verbose: print(f"--- Inference took {counter_end_inference - counter_start_inference:.2f} seconds for {N} samples ---")

    # bring solutions back to host for numpy()
    counter_start_cpu = time.time()
    sols_ok = torch.cat(sols_ok, dim=0)  # -> (ΣKi, 7)
    fk_ok   = torch.cat(fk_ok,   dim=0)  # -> (ΣKi, 7)
    sols_np = sols_ok.cpu().numpy()
    fk_np   = fk_ok.cpu().numpy()
    counter_end_cpu = time.time()
    if verbose: print(f"--- Bringing solutions to cpu took {counter_end_cpu - counter_start_cpu:.2f} seconds ---")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        input("Press Enter to start visualizing IK-flow solutions…")

        # first update the marker in the new pose (NOTE: in terms of world coordinates)
        quat_frame = rotm_to_quaternion(A_w_p[:3, :3])
        set_body_pose(model, data, piece_body_id,
                      t_w_p.tolist(),
                      [quat_frame[0], quat_frame[1], quat_frame[2], quat_frame[3]])
        mujoco.mj_forward(model, data)

        # loop over each valid IK solution
        if use_ikflow:
            for i, (q, x) in enumerate(zip(sols_np, fk_np), 1):
                if verbose:
                    print(f"[OK] sol {i:2d}: q={np.round(q,3)}  →  x={np.round(x,3)}")

                # apply joint solution
                data.qpos[:6] = q.tolist()
                mujoco.mj_forward(model, data)

                viewer.sync()
                time.sleep(show_pose_duration)

            if verbose:
                print("\n--- Finished all IK-flow solutions. ---")
            input("Press Enter to close the viewer…")
        else:
            # hard-coded joint configuration for testing
            q = np.radians([100, -94.96, 101.82, -95.72, -96.35, 180])
            #q = np.array([np.radians(-8.38), np.radians(-68.05), np.radians(-138), np.radians(-64), np.radians(90), np.radians(90)])
            data.qpos[:6] = q.tolist()
            mujoco.mj_forward(model, data)
            viewer.sync()

            # Compute torques to compensate gravity
            gravity_comp = data.qfrc_bias[:6]
            

            if verbose:
                print(f"[OK] Hard-coded joint configuration: q={np.round(q,3)}")
                print(f"Gravity compensation torques: {np.round(gravity_comp, 3)}")
                print("\n--- Finished hard-coded joint configuration. ---")
            input("Press Enter to close the viewer…")

if __name__ == "__main__":
    main()
