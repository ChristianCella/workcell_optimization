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

#* Directory for scene creation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scene_manager')))
from create_scene import create_reference_frames,  merge_robot_and_tool, inject_robot_tool_into_scene, add_instance
from config import *
rob_params = rob_par
from parameters import TestIK
ik_params = TestIK()

#* Directory for the utilities
utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(utils_dir)
import fonts
from transformations import rotm_to_quaternion, rotm2euler, get_homogeneous_matrix
from mujoco_utils import set_body_pose, get_collisions, inverse_manipulability, compute_jacobian
from ikflow_inference import FastIKFlowSolver, solve_ik_fast
fast_ik_solver = FastIKFlowSolver()

def main():

    # Path setup 
    robot_folder = rob_folder
    robot_name = rob_name
    tool_filename = tool_name
    robot_and_tool_file_name = f"temp_{robot_to_use}_with_tool.xml"
    output_scene_filename = "final_scene.xml"
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    # Create a single xml (robot + tool)
    _ = merge_robot_and_tool(
        robot_filename =robot_name, 
        robot_folder=robot_folder, 
        tool_filename=tool_filename, 
        base_dir=base_dir, 
        output_robot_tool_filename=robot_and_tool_file_name
        )
    
    # Add the robot + tool to the scene
    _ = inject_robot_tool_into_scene(
        robot_tool_filename=robot_and_tool_file_name, 
        output_scene_filename=output_scene_filename, 
        base_dir=base_dir,
        robot_folder=robot_folder
    )
    
    # Add one reference frame
    temp_xml_name = create_reference_frames(base_dir, "ur5e_utils_mujoco/" + output_scene_filename, 1)
    model_path = os.path.join(base_dir, "ur5e_utils_mujoco", temp_xml_name)

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Get body/site IDs
    piece_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "reference_target_1")
    base_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    tool_base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_base")
    tool_tip_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")
    tool_tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tool_site')

    # Set robot base
    _, _, A_w_b = get_homogeneous_matrix(0.0, 0.0, 0.0, 0, 0, 0)
    set_body_pose(model, data, base_body_id, A_w_b[:3, 3], rotm_to_quaternion(A_w_b[:3, :3]))

    # Set the Cartesian target (piece to manipulate)
    _, _, A_w_p = get_homogeneous_matrix(-0.55, 0.135, 0.03, 180, 0, 90)
    set_body_pose(model, data, piece_body_id, A_w_p[:3, 3], rotm_to_quaternion(A_w_p[:3, :3]))

    # Set the tool
    #_, _, A_ee_t1 = get_homogeneous_matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) # Welding gun
    _, _, A_ee_t1 = get_homogeneous_matrix(0.0, 0.0, 0.0, 0.0, 0.0, -45.0) # Screwdriver
    set_body_pose(model, data, tool_base_body_id, A_ee_t1[:3, 3], rotm_to_quaternion(A_ee_t1[:3, :3])) # Update tool base
    #_, _, A_t1_t = get_homogeneous_matrix(0.0, -0.083033, 0.31549, 45.0, 0.0, 0.0) # Welding gun
    _, _, A_t1_t = get_homogeneous_matrix(0, -0.195, 0.028, 90.0, 0.0, 0.0) # Screwdriver
    A_ee_t = A_ee_t1 @ A_t1_t
    set_body_pose(model, data, tool_tip_body_id, A_ee_t[:3, 3], rotm_to_quaternion(A_ee_t[:3, :3])) # Update tool tip

    # End-effector with respect to wrist3
    t_wl3_ee = np.array([0, 0.1, 0])
    R_wl3_e = R.from_euler('XYZ', [np.radians(-90), 0, 0], degrees=False).as_matrix()
    A_wl3_ee = np.eye(4)
    A_wl3_ee[:3, 3] = t_wl3_ee
    A_wl3_ee[:3, :3] = R_wl3_e

    #! ikflow inference  
    with mujoco.viewer.launch_passive(model, data) as viewer:
        input("Press Enter to start visualizing IK-flow solutions…")

        # Piece in the world frame 
        t_w_p = A_w_p[:3, 3]
        R_w_p = A_w_p[:3, :3]
        theta_w_p_x_0, theta_w_p_y_0, theta_w_p_z_0 = rotm2euler(R_w_p, degrees=False)

        # Loop through the discrete configurations
        sols_ok, fk_ok = [], []
        for i in range(ik_params.N_disc): 
            R_w_p_rotated = R.from_euler('XYZ', [theta_w_p_x_0, theta_w_p_y_0, theta_w_p_z_0 + i * 2 * np.pi / ik_params.N_disc], degrees=False).as_matrix()
            A_w_p_rotated = np.eye(4)
            A_w_p_rotated[:3, 3] = t_w_p
            A_w_p_rotated[:3, :3] = R_w_p_rotated
            A_b_wl3 = np.linalg.inv(A_w_b) @ A_w_p_rotated @ np.linalg.inv(A_ee_t)@ np.linalg.inv(A_wl3_ee)
            quat_pose = rotm_to_quaternion(A_b_wl3[:3, :3])
            target = np.array([
                A_b_wl3[0, 3], A_b_wl3[1, 3], A_b_wl3[2, 3], 
                quat_pose[0], quat_pose[1], quat_pose[2], quat_pose[3] 
            ], dtype=np.float64)
            tgt_tensor = torch.from_numpy(target.astype(np.float32))
            sols_disc, fk_disc = solve_ik_fast(tgt_tensor, N = ik_params.N_samples, fast_solver=fast_ik_solver) # Find N solutions for this target
            sols_ok.append(sols_disc)
            fk_ok.append(fk_disc)

        # bring solutions back to host for numpy()
        sols_ok = torch.cat(sols_ok, dim=0)
        fk_ok = torch.cat(fk_ok,   dim=0)
        sols_np = sols_ok.cpu().numpy()
        fk_np = fk_ok.cpu().numpy()

        # Update the pose of the Cartesian target
        quat_frame = rotm_to_quaternion(A_w_p[:3, :3])
        set_body_pose(model, data, piece_body_id,
                    t_w_p.tolist(),
                    [quat_frame[0], quat_frame[1], quat_frame[2], quat_frame[3]])
        mujoco.mj_forward(model, data)

        # loop over each valid IK solution
        best_cost = 1e12
        for i, (q, x) in enumerate(zip(sols_np, fk_np), 1):

            print(f"{fonts.green}Joints (rad): {np.round(q, 3)}{fonts.reset}")
            print(f"{fonts.green}Joints (deg): {np.round(np.degrees(q), 3)}{fonts.reset}")
            print(f"{fonts.blue}Cartesian pose: {np.round(x, 3)}{fonts.reset}")

            # apply joint solution
            data.qpos[:rob_params.nu] = q.tolist()
            mujoco.mj_forward(model, data)

            viewer.sync()
            n_cols = get_collisions(model, data, False)
            sigma_manip = inverse_manipulability(q, model, data, tool_tip_site_id)
            time.sleep(ik_params.show_pose_duration)

            # Save the configuration with best inverse manipulability
            if (sigma_manip < best_cost) and (n_cols == 0):
                print(f"{fonts.yellow}New best solution found with cost {sigma_manip:.3f}!{fonts.reset}")
                best_cost = sigma_manip
                best_q = q

        print(f"The best configuration is: {np.round(best_q, 3)} with cost {best_cost:.3f}")
        input("Press Enter to apply the best configuration…")

        # Optimization is over => apply the best configuration
        data.qpos[:rob_params.nu] = best_q.tolist()
        mujoco.mj_forward(model, data)
        viewer.sync()      
        input("Press Enter to close the viewer…")

if __name__ == "__main__":
    main()