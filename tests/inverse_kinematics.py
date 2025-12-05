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

#* ur5e directory
ur5e_utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ur5e_utils_mujoco'))

#* Utils directory
utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(utils_dir)
import fonts
from transformations import rotm_to_quaternion, get_homogeneous_matrix, quaternion_to_euler, get_cartesian_pose
from mujoco_utils import set_body_pose, get_collisions, inverse_manipulability, scene_manager
from ikflow_inference import FastIKFlowSolver, solve_ik_fast
from constant_parameters import TestIkFlow

#* Database directory
database_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../database'))
sys.path.append(database_dir)
from query_db import get_tcp_frame

def main():

    # Acquire parameters
    params = TestIkFlow()
    tool_id = "gripper_hande"  # Default tool

    # Path setup 
    model_path = scene_manager(params.mode, params.n_targets, ur5e_utils_dir, "bringup_ur5e.xml", "extension.xml")

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Get body/site IDs
    base_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base") #* Robot base
    target_body_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "reference_target_1") #* target

    tool_base_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_base") #* Base of the tool
    tool_base_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tool_base_site')
    tool_tip_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_tip") #* 'movable' frame in the tool

    ext_base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ext_base") #* Base of the extension tool

    # Define the pose of the target in the world frame
    if params.use_database:
        tcp_frame, wrench_values, tool_id = get_tcp_frame("23")
        q_frame = [tcp_frame[6], tcp_frame[3], tcp_frame[4], tcp_frame[5]] 
        theta_w_p_x_0, theta_w_p_y_0, theta_w_p_z_0 = quaternion_to_euler(q_frame, degrees=False)
        print(f"The angles are: {theta_w_p_x_0}, {theta_w_p_y_0}, {theta_w_p_z_0}")
        t_w_p = np.array([tcp_frame[0], tcp_frame[1], tcp_frame[2]])
        R_w_p = R.from_euler('XYZ', [theta_w_p_x_0, theta_w_p_y_0, theta_w_p_z_0], degrees=False).as_matrix()
        A_w_p = np.eye(4)
        A_w_p[:3, 3] = t_w_p
        A_w_p[:3, :3] = R_w_p
        if tool_id == "gripper_hande":
            gripper_length = params.hande_offset
        else:   
            gripper_length = params.extension_offset + params.hande_offset

    else:
        t_w_p = np.array([params.x_tar, params.y_tar, params.z_tar]) 
        R_w_p = R.from_euler('XYZ', [np.radians(params.theta_x_tar), np.radians(params.theta_y_tar), np.radians(params.theta_z_tar)], degrees=False).as_matrix()
        A_w_p = np.eye(4)
        A_w_p[:3, 3] = t_w_p
        A_w_p[:3, :3] = R_w_p
        gripper_length = params.extension_offset + params.hande_offset
        tool_id = "extension"
        #gripper_length = params.hande_offset
        theta_w_p_x_0 = np.radians(params.theta_x_tar)
        theta_w_p_y_0 = np.radians(params.theta_y_tar)
        theta_w_p_z_0 = np.radians(params.theta_z_tar)

    # Set robot base (matrix A^w_b)
    _, _, A_w_b = get_homogeneous_matrix(0.0, 0.0, 0.0, 0, 0, 0)
    set_body_pose(model, data, base_body_id, A_w_b[:3, 3], rotm_to_quaternion(A_w_b[:3, :3]))

    # Update the target pose in the simulator
    set_body_pose(model, data, target_body_id, A_w_p[:3, 3], rotm_to_quaternion(A_w_p[:3, :3]))

    # Set the base of the tool with respect to the flange
    _, _, A_ee_t1 = get_homogeneous_matrix(0.0, 0.0, 0.0, 0, 0, 0)
    set_body_pose(model, data, tool_base_body_id, A_ee_t1[:3, 3], rotm_to_quaternion(A_ee_t1[:3, :3]))

    # Fixed transformation 'tool base (t1) => tool tip (t)'
    _, _, A_t1_t = get_homogeneous_matrix(0, 0, gripper_length, 0, 0, 0)
    set_body_pose(model, data, tool_tip_body_id, A_t1_t[:3, 3], rotm_to_quaternion(A_t1_t[:3, :3])) #* Tool tip update

    # Compute the transformation 'end-effector (ee) => tool tip (t)'
    A_ee_t = A_ee_t1 @ A_t1_t

    # End-effector with respect to wrist3 (this is ALWAYS fixed)
    t_wl3_ee = np.array([0, 0.1, 0])
    R_wl3_e = R.from_euler('XYZ', [np.radians(-90), 0, 0], degrees=False).as_matrix()
    A_wl3_ee = np.eye(4)
    A_wl3_ee[:3, 3] = t_wl3_ee
    A_wl3_ee[:3, :3] = R_wl3_e

    # Constant matrices for the extension tool
    _, _, A_eb_et = get_homogeneous_matrix(0, 0, 0.2, 0, 0, 0)

    #! Make inference on the nornmalizing flow (ikflow)   
    fast_ik_solver = FastIKFlowSolver()       
    counter_start_inference = time.time()

    # Loop through the discrete configurations
    sols_ok, fk_ok = [], []
    for i in range(params.N_disc): # 0, 1, 2, ... N_disc-1
        R_w_p_rotated = R.from_euler('XYZ', [theta_w_p_x_0, theta_w_p_y_0, theta_w_p_z_0 + i * 2 * np.pi / params.N_disc], degrees=False).as_matrix()
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
        sols_disc, fk_disc = solve_ik_fast(tgt_tensor, N = params.N_samples, fast_solver=fast_ik_solver) # Find N solutions for this target
        sols_ok.append(sols_disc)
        fk_ok.append(fk_disc)

    counter_end_inference = time.time()
    print(f"--- Inference took {counter_end_inference - counter_start_inference:.2f} seconds for {params.N_samples} samples ---")

    # bring solutions back to host for numpy()
    counter_start_cpu = time.time()
    sols_ok = torch.cat(sols_ok, dim=0)  # -> (ΣKi, 7)
    fk_ok   = torch.cat(fk_ok,   dim=0)  # -> (ΣKi, 7)
    sols_np = sols_ok.cpu().numpy()
    fk_np   = fk_ok.cpu().numpy()
    counter_end_cpu = time.time()
    if params.verbose: print(f"--- Bringing solutions to cpu took {counter_end_cpu - counter_start_cpu:.2f} seconds ---")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        input("Press Enter to start visualizing IK-flow solutions…")

        # loop over each valid IK solution
        cost = 1e12
        best_cost = 1e12
        start_inference = time.time()

        for i, (q, x) in enumerate(zip(sols_np, fk_np), 1):

            # apply the i-th joint solution
            data.qpos[:6] = q.tolist()
            mujoco.mj_forward(model, data)

            # Now, update the scene also for the extension tool (if any)
            if params.mode == "full":
                if tool_id != "gripper_hande":
                    pos, quat = get_cartesian_pose(tool_tip_body_id, data)
                    eul = quaternion_to_euler(quat, degrees=False)
                    _, _, A_w_et = get_homogeneous_matrix(pos[0], pos[1], pos[2], np.degrees(eul[0]), np.degrees(eul[1]), np.degrees(eul[2]))
                    A_w_eb = A_w_et @ np.linalg.inv(A_eb_et)
                    set_body_pose(model, data, ext_base_body_id, A_w_eb[:3, 3], rotm_to_quaternion(A_w_eb[:3, :3]))
                    mujoco.mj_forward(model, data)
                else:
                    _, _, A_w_et = get_homogeneous_matrix(2, 2, 2, 0, 0, 0)
                    A_w_eb = A_w_et @ np.linalg.inv(A_eb_et)
                    set_body_pose(model, data, ext_base_body_id, A_w_eb[:3, 3], rotm_to_quaternion(A_w_eb[:3, :3]))
                    mujoco.mj_forward(model, data)
            viewer.sync()

            # Evaluate collisions and manipulability
            n_cols = get_collisions(model, data, params.verbose)
            sigma_manip = inverse_manipulability(q, model, data, tool_base_site_id)
            #time.sleep(params.show_pose_duration)          
            if params.verbose: print(f"Number of collisions detected: {n_cols}; inverse manipulability: {sigma_manip:.3f}")
            input(f"Press a key to change move on...")

            # Compute the Cartesian pose of the tool tip (just a check)
            pos_tt = data.xpos[tool_tip_body_id] 
            rot_tt = data.xmat[tool_tip_body_id].reshape(3, 3) 
            quat_tt = rotm_to_quaternion(rot_tt)
            print(f"{fonts.green}Tool tip Cartesian pose: pos={np.round(pos_tt,3)}, quat={np.round(quat_tt,3)}{fonts.reset}")

            # Compute the metric for the evaluation
            if n_cols > 0:
                cost = 1e12
            else:
                cost = sigma_manip

            # Save the configuration with best inverse manipulability
            if cost < best_cost:
                best_cost = cost
                best_q = q

        print(f"N of samples{len(sols_np)}; total time: {time.time() - start_inference:.2f} seconds")
        input("Press Enter to apply the best configuration…")

        # Optimization is over => apply the best configuration for visualization purposes
        data.qpos[:6] = best_q.tolist()
        mujoco.mj_forward(model, data)
        viewer.sync()

        input("Press Enter to close the viewer…")

if __name__ == "__main__":
    main()
