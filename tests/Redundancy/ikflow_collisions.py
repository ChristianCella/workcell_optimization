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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scene_manager')))
from parameters import TestIkFlow
params = TestIkFlow()
from create_scene import create_scene

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils'))
sys.path.append(base_dir)
import fonts
from transformations import rotm_to_quaternion, get_homogeneous_matrix
from mujoco_utils import set_body_pose, get_collisions, inverse_manipulability
from ikflow_inference import FastIKFlowSolver, solve_ik_fast

def main():

    # Path setup 
    tool_filename = "driller.xml"
    robot_and_tool_file_name = "temp_kuka_with_tool.xml"
    output_scene_filename = "final_scene.xml"
    piece_name1 = "aluminium_plate.xml" 
    piece_name2 = "Linear_guide.xml"
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))

    # Create the scene

    model_path = create_scene(tool_name=tool_filename, robot_and_tool_file_name=robot_and_tool_file_name,
                              output_scene_filename=output_scene_filename, piece_name1=piece_name1, piece_name2=piece_name2, base_dir=base_dir)

    #base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    #model_path = os.path.join(base_dir, "kuka_iiwa_14_mujoco_utils/kuka/iiwa14.xml")

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Get body/site IDs
    piece_body_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "aluminium_plate")
    base_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    tool_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")
    tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tool_site')
    screwdriver_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_top")
    wrist_3_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_frame_visual_only") # wrist_3

    # Set robot base (matrix A^w_b)
    t_w_b = np.array([0.0, 0.0, 0.15])
    #t_w_b = np.array([0.0, 0.0, 0.0])
    R_w_b = R.from_euler('XYZ', [np.radians(0), np.radians(0), np.radians(90)], degrees=False).as_matrix()
    #R_w_b = R.from_euler('XYZ', [np.radians(0), np.radians(0), np.radians(0)], degrees=False).as_matrix()
    A_w_b = np.eye(4)
    A_w_b[:3, 3] = t_w_b
    A_w_b[:3, :3] = R_w_b
    set_body_pose(model, data, base_body_id, A_w_b[:3, 3], rotm_to_quaternion(A_w_b[:3, :3]))

    # Set the frame 'screw_top to a new pose wrt flange' and move the screwdriver there
    t_ee_t1 = np.array([0.1, 0.0, 0.0]) # 0, 0.15, 0
    R_ee_t1 = R.from_euler('XYZ', [np.radians(0), np.radians(0), np.radians(0)], degrees=False).as_matrix() # 30, 0, 0
    A_ee_t1 = np.eye(4)
    A_ee_t1[:3, 3] = t_ee_t1
    A_ee_t1[:3, :3] = R_ee_t1
    set_body_pose(model, data, screwdriver_body_id, A_ee_t1[:3, 3], rotm_to_quaternion(A_ee_t1[:3, :3]))

    # Fixed transformation 'tool top (t1) => tool tip (t)'
    t_t1_t = np.array([0, 0.0, 0.26])
    R_t1_t = R.from_euler('XYZ', [np.radians(0), np.radians(0), np.radians(0)], degrees=False).as_matrix()
    A_t1_t = np.eye(4)
    A_t1_t[:3, 3] = t_t1_t
    A_t1_t[:3, :3] = R_t1_t

    # Update the position of the tool tip (Just for visualization purposes)
    A_ee_t = A_ee_t1 @ A_t1_t  # combine the two transformations
    set_body_pose(model, data, tool_body_id, A_ee_t[:3, 3], rotm_to_quaternion(A_ee_t[:3, :3]))

    # Set the piece in the environment (matrix A^w_p)
    _, _, A_w_p = get_homogeneous_matrix(0.6, 0, 0.2, 0, 0, 180)
    set_body_pose(model, data, piece_body_id, A_w_p[:3, 3], rotm_to_quaternion(A_w_p[:3, :3]))

    # Get the frame of the hole
    ref_body_ids = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and name.startswith("hole_") and name.endswith("_frame_body"):
            ref_body_ids.append(i)
    #tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")
    #screwdriver_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_top")
    

    #! Make inference on the nornmalizing flow (ikflow)   
    fast_ik_solver = FastIKFlowSolver()       
    counter_start_inference = time.time()

    #Get the pose of the target 
    posit = data.xpos[ref_body_ids[1]]
    rotm = data.xmat[ref_body_ids[1]].reshape(3, 3)
    theta_x_0, theta_y_0, theta_z_0 = R.from_matrix(rotm).as_euler('XYZ', degrees=True)

    #print(f"The pose of the target is: pos={np.round(posit,3)}, angles={np.round([theta_x_0, theta_y_0, theta_z_0],3)}")

    # Loop through the discrete configurations
    sols_ok, fk_ok = [], []
    for i in range(params.N_disc): # 0, 1, 2, ... N_disc-1

        _, _, A_w_p_rotated = get_homogeneous_matrix(posit[0], posit[1], posit[2], theta_x_0, theta_y_0, theta_z_0 + i * 360 / params.N_disc)
        A_b_ee = np.linalg.inv(A_w_b) @ A_w_p_rotated @ np.linalg.inv(A_ee_t)

        # Create the target pose for the IK solver (from robot base to wrist_link_3)
        quat_pose = rotm_to_quaternion(A_b_ee[:3, :3])
        target = np.array([
            A_b_ee[0, 3], A_b_ee[1, 3], A_b_ee[2, 3],   # position
            quat_pose[0], quat_pose[1], quat_pose[2], quat_pose[3]  # quaternion
        ], dtype=np.float64)
        tgt_tensor = torch.from_numpy(target.astype(np.float32))

        # Solve the IK problem for the discretized pose
        sols_disc, fk_disc = solve_ik_fast(tgt_tensor, N=params.N_samples, fast_solver=fast_ik_solver) # Find N solutions for this target
        sols_ok.append(sols_disc)
        fk_ok.append(fk_disc)

    counter_end_inference = time.time()
    print(f"All the retrieved joint poses are: {sols_ok}")
    print(f"--- Inference took {counter_end_inference - counter_start_inference:.2f} seconds for {params.N_samples} samples ---")

    # bring solutions back to host for numpy()
    sols_ok = torch.cat(sols_ok, dim=0)  # -> (ΣKi, 7)
    fk_ok   = torch.cat(fk_ok,   dim=0)  # -> (ΣKi, 7)
    sols_np = sols_ok.cpu().numpy()
    fk_np   = fk_ok.cpu().numpy()


    with mujoco.viewer.launch_passive(model, data) as viewer:
        input("Press Enter to start visualizing IK-flow solutions…")      

        # loop over each valid IK solution
        best_cost = 1e12
        best_q = np.zeros(7)
        start_inference = time.time()

        
        if params.use_ikflow:

            for i, (q, x) in enumerate(zip(sols_np, fk_np), 1):
                if params.verbose:
                    print(f"[OK] sol {i:2d}: q={np.round(q,3)}  →  x={np.round(x,3)}")

                # apply joint solution
                data.qpos[:7] = q.tolist()
                mujoco.mj_forward(model, data)

                viewer.sync()
                n_cols = get_collisions(model, data, params.verbose)
                sigma_manip = inverse_manipulability(q, model, data, tool_site_id)
                print(f"{fonts.yellow}The manipulability is {sigma_manip}{fonts.reset}")
                time.sleep(params.show_pose_duration)
                #print(f"Number of collisions detected: {n_cols}; inverse manipulability: {sigma_manip:.3f}")

                # Save the configuration with best inverse manipulability
                if (sigma_manip < best_cost) and (n_cols == 0):
                    print(f"feasible configuration!")
                    best_cost = sigma_manip
                    best_q = q
                    print(f"{fonts.green}The current best config is: {best_q}{fonts.reset}")

            print(f"Evaluating collisions and Jacobian on {len(sols_np)} samples lasted {time.time() - start_inference:.2f} seconds")
            print(f"The best configuration is: {np.round(best_q, 3)} with cost {best_cost:.3f}")
            input("Press Enter to apply the best configuration…")
            # Optimization is over => apply the best configuration
            data.qpos[:7] = best_q.tolist()
            mujoco.mj_forward(model, data)
            viewer.sync()

            input("Press Enter to close the viewer…")

        else:

            # hard-coded joint configuration for testing
            q = np.array([1.7382,  0.8907, -2.2000, -1.5925,  0.4470, -1.9824, -1.1051])
            data.qpos[:7] = q.tolist()
            mujoco.mj_forward(model, data)

            # Print collisions
            n_cols = get_collisions(model, data, params.verbose)
            print(f"Collisions detected: {n_cols}")
            viewer.sync()

            # Get the forward kinematics
            pos = data.xpos[wrist_3_id]  # shape: (3,)
            rot = data.xmat[wrist_3_id].reshape(3, 3)  # shape: (3, 3)
            quat = rotm_to_quaternion(rot)
            print(f"FK wrist 3: pos={np.round(pos, 3)}, quat={np.round(quat, 3)}")

            # Compute torques to compensate gravity
            gravity_comp = data.qfrc_bias[:7]
            
            if params.verbose: print(f"Gravity compensation torques: {np.round(gravity_comp, 3)}")
            input("Press Enter to close the viewer…")

if __name__ == "__main__":
    main()
