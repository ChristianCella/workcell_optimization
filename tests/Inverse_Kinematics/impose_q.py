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
    use_ikflow = True  # set to False to test a hard-coded joint configuration 
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

    #! Move bodies in space (simulation of 'changing' the layout)

    # Set robot base (matrix A^w_b)
    t_w_b = np.array([-0.1, -0.1, 0.1])
    R_w_b = R.from_euler('xyz', [np.radians(45), np.radians(60), 0], degrees=False).as_matrix()
    A_w_b = np.eye(4)
    A_w_b[:3, 3] = t_w_b
    A_w_b[:3, :3] = R_w_b

    model.body_pos[base_body_id] = A_w_b[:3, 3]
    model.body_quat[base_body_id] = rotm_to_quaternion(A_w_b[:3, :3])

    # Set the tool to a new pose with respect to the ee (define A^ee_t)
    t_ee_t = np.array([0, 0, 0.1])
    R_ee_t = R.from_euler('xyz', [np.radians(20), np.radians(0), np.radians(0)], degrees=False).as_matrix()
    A_ee_t = np.eye(4)
    A_ee_t[:3, 3] = t_ee_t
    A_ee_t[:3, :3] = R_ee_t

    model.body_pos[tool_body_id] = A_ee_t[:3, 3]
    model.body_quat[tool_body_id] = rotm_to_quaternion(A_ee_t[:3, :3])

    # Piece in the world (define A^w_p)
    t_w_p = np.array([0.3, -0.174, 0.7])
    R_w_p = R.from_euler('xyz', [np.radians(0), 0, np.radians(0)], degrees=False).as_matrix()
    A_w_p = np.eye(4)
    A_w_p[:3, 3] = t_w_p
    A_w_p[:3, :3] = R_w_p

    # End-effector with respect to wrist3
    t_wl3_ee = np.array([0, 0.1, 0])
    R_wl3_e = R.from_euler('xyz', [np.radians(-90), 0, 0], degrees=False).as_matrix()
    A_wl3_ee = np.eye(4)
    A_wl3_ee[:3, 3] = t_wl3_ee
    A_wl3_ee[:3, :3] = R_wl3_e

    # Compute the target pose (robot base => last DH link)
    A_b_wl3 = np.linalg.inv(A_w_b) @ A_w_p @ np.linalg.inv(A_ee_t)@ np.linalg.inv(A_wl3_ee)
    quat_pose = rotm_to_quaternion(A_b_wl3[:3, :3])
    target = np.array([
        A_b_wl3[0, 3], A_b_wl3[1, 3], A_b_wl3[2, 3],   # position
         quat_pose[0], quat_pose[1], quat_pose[2], quat_pose[3]  # quaternion
    ], dtype=np.float64)

    #! Make inference on the nornmalizing flow (ikflow)
    N = 1000
    fast_ik_solver = FastIKFlowSolver()    
    tgt_tensor = torch.from_numpy(target.astype(np.float32))
    counter_start_inference = time.time()
    sols_ok, fk_ok = solve_ik_fast(tgt_tensor, N=N, fast_solver=fast_ik_solver)
    counter_end_inference = time.time()
    if verbose: print(f"--- Inference took {counter_end_inference - counter_start_inference:.2f} seconds for {N} samples ---")

    # bring solutions back to host for numpy()
    counter_start_cpu = time.time()
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

if __name__ == "__main__":
    main()
