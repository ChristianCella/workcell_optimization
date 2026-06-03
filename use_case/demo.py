#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
import time
import sys
import os
import torch
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
from mujoco_utils import set_body_pose, get_collisions, solve_ik_dls, joint_displacement
from ikflow_inference import FastIKFlowSolver, solve_ik_fast
fast_ik_solver = None
if ik_solver_to_use == "ikflow": fast_ik_solver = FastIKFlowSolver()

def main():

    # Path setup 
    robot_folder = rob_folder
    robot_name = rob_name
    tool_filename = tool_name
    robot_and_tool_file_name = f"temp_{robot_to_use}_with_tool.xml"
    piece_name = pie_name
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
    merged_scene_path = inject_robot_tool_into_scene(
        robot_tool_filename=robot_and_tool_file_name, 
        output_scene_filename=output_scene_filename, 
        base_dir=base_dir,
        robot_folder=robot_folder
    )

    # Import the piece with the trajectory
    add_instance(
        base_scene_path=merged_scene_path,
        instance_path=os.path.join(base_dir, "ur5e_utils_mujoco/pieces", piece_name),
        output_path=merged_scene_path,
        mesh_source_dir=os.path.join(base_dir, "ur5e_utils_mujoco/pieces"),
        mesh_target_dir=os.path.join(base_dir, f"ur5e_utils_mujoco/{robot_folder}/assets")
    )
    
    # Add one reference frame
    model_path = os.path.join(base_dir, "ur5e_utils_mujoco", output_scene_filename)

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Get body/site IDs
    piece_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    base_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    tool_base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_base")
    tool_tip_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")
    tool_tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tool_site')

    # Set robot base (matrix A^w_b)
    _, _, A_w_b = get_homogeneous_matrix(0.0, 0.0, 0.25, 0.0, 0.0, 180.0) 
    set_body_pose(model, data, base_body_id, A_w_b[:3, 3], rotm_to_quaternion(A_w_b[:3, :3]))
    data.qpos[:rob_params.nu] = rob_params.home_configuration.tolist()

    # Set the piece in the environment out of the way (matrix A^w_p)
    _, _, A_w_p = get_homogeneous_matrix(0.0, -0.75, 0.0, 0.0, 0.0, 0.0) 
    set_body_pose(model, data, piece_body_id, A_w_p[:3, 3], rotm_to_quaternion(A_w_p[:3, :3]))

    # Set the tool
    if tool_to_use == "welding_gun":
        _, _, A_ee_t1 = get_homogeneous_matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) # Welding gun
        set_body_pose(model, data, tool_base_body_id, A_ee_t1[:3, 3], rotm_to_quaternion(A_ee_t1[:3, :3])) # Update tool base
        _, _, A_t1_t = get_homogeneous_matrix(0.0, -0.083033, 0.31549, 45.0, 0.0, 0.0) # Welding gun
    elif tool_to_use == "screwdriver":
        _, _, A_ee_t1 = get_homogeneous_matrix(0.0, 0.0, 0.0, 0.0, 0.0, -45.0) # Screwdriver
        set_body_pose(model, data, tool_base_body_id, A_ee_t1[:3, 3], rotm_to_quaternion(A_ee_t1[:3, :3])) # Update tool base
        _, _, A_t1_t = get_homogeneous_matrix(0, -0.195, 0.028, 90.0, 0.0, 0.0) # Screwdriver
    else:
        raise ValueError(f"Unknown tool type: {tool_to_use}")

    # Compute the final tool tip pose
    A_ee_t = A_ee_t1 @ A_t1_t
    set_body_pose(model, data, tool_tip_body_id, A_ee_t[:3, 3], rotm_to_quaternion(A_ee_t[:3, :3])) # Update tool tip

    #! Fixed matrix A_wl3_ee
    _, _, A_wl3_ee = get_homogeneous_matrix(0.0, 0.1, 0.0, -90.0, 0.0, 0.0)

    # Get the Cartesian path stitched to the piece
    cartesian_frames = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and name.startswith("point_") and name.endswith("_traj"):
            cartesian_frames.append(i)

    cartesian_path = []
    for frame_id in cartesian_frames:
        pos = data.body(frame_id).xpos
        rot = data.body(frame_id).xmat.reshape(3, 3)
        euler_angles = rotm2euler(rot, degrees=False)
        cartesian_path.append((pos, euler_angles))
  
    #! Solve IK on the trajectory
    with mujoco.viewer.launch_passive(model, data) as viewer:
        input("Press Enter to start visualizing IK-flow solutions…")

        #* Loop throug all points of the path
        q_path = []
        start_time = time.time()
        for j in range(len(cartesian_path)):
            q_old = data.qpos[:rob_params.nu].copy()
            t_w_p = cartesian_path[j][0]
            theta_w_p_x_0, theta_w_p_y_0, theta_w_p_z_0 = cartesian_path[j][1]           

            #! Time-consuming solver
            if ik_solver_to_use == "ikflow":
                sols_ok, fk_ok = [], []
                #* Retrieve N ik solutions
                for i in range(ik_params.N_disc):
                    R_w_p_rotated = R.from_euler('XYZ', [theta_w_p_x_0, theta_w_p_y_0, theta_w_p_z_0 + i * 2 * np.pi / ik_params.N_disc], degrees=False).as_matrix()
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
                    sols_disc, fk_disc = solve_ik_fast(tgt_tensor, N = ik_params.N_samples, fast_solver=fast_ik_solver) # Find N solutions for this target
                    sols_ok.append(sols_disc)
                    fk_ok.append(fk_disc)

                # bring solutions back to host for numpy()
                sols_ok = torch.cat(sols_ok, dim=0)
                fk_ok = torch.cat(fk_ok,   dim=0)
                sols_np = sols_ok.cpu().numpy()
                fk_np = fk_ok.cpu().numpy()

                #* Rank each candidate based on smallest joint displacement from the previous configuration
                best_cost = 1e12
                best_q = np.zeros(rob_params.nu)
                for i, (q, x) in enumerate(zip(sols_np, fk_np), 1):

                    # apply joint solution
                    data.qpos[:6] = q.tolist()
                    mujoco.mj_forward(model, data)
                    n_cols = get_collisions(model, data, False)

                    # Smallest joint displacement
                    displacement = joint_displacement(q, q_old)
                    if (displacement < best_cost) and (n_cols == 0):
                        best_cost = displacement
                        best_q = q

                # Found optimal config
                q_path.append(best_q)
                data.qpos[:6] = best_q.tolist()
                mujoco.mj_forward(model, data)

            #! Damped-least squares
            elif ik_solver_to_use == "dls":
                target_rot = R.from_euler('XYZ', [theta_w_p_x_0, theta_w_p_y_0, theta_w_p_z_0], degrees=False).as_matrix()
                best_q = solve_ik_dls(model, data, tool_tip_site_id, t_w_p, target_rot, q_init=q_old)
                data.qpos[:6] = best_q
                mujoco.mj_forward(model, data)
                q_path.append(best_q)
            else:
                raise ValueError(f"Unknown IK solver type: {ik_solver_to_use}")
      
        # Path found
        end_time = time.time()
        print(f"{fonts.green}ik optimization completed in {end_time - start_time:.2f} seconds!{fonts.reset}")
        
        # Save path to CSV
        q_path_np = np.array(q_path)  
        csv_path = os.path.join(base_dir, "workcell_optimization/results", f"q_path_{robot_to_use}_{ik_solver_to_use}.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        np.savetxt(csv_path, q_path_np, delimiter=",",
                   header="q1,q2,q3,q4,q5,q6", comments="")
        print(f"{fonts.green}Path saved to {csv_path}{fonts.reset}")
        
        input("Press Enter to visualize the best path found")
        data.qpos[:6] = rob_params.home_configuration.tolist()
        mujoco.mj_forward(model, data)
        viewer.sync()
        for q in q_path:
            data.qpos[:6] = q.tolist()
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.5)
        input("Press Enter to finish the demo…")

if __name__ == "__main__":
    main()