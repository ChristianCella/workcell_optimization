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
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as ker
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process.kernels import WhiteKernel, Matern


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scene_manager')))
from parameters import TestIkFlow
params = TestIkFlow()
from create_scene import create_reference_frames,  merge_robot_and_tool, inject_robot_tool_into_scene, add_instance

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils'))
sys.path.append(base_dir)
import fonts
from transformations import get_world_wrench, rotm_to_quaternion
from mujoco_utils import compute_jacobian, set_body_pose, get_collisions, inverse_manipulability
from ikflow_inference import FastIKFlowSolver, solve_ik_fast

def main():

    # Path setup 
    tool_filename = "screwdriver.xml"
    robot_and_tool_file_name = "temp_ur5e_with_tool.xml"
    output_scene_filename = "final_scene.xml"
    obstacle_name = "screwing_plate.xml" 
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))

    # Create the robot + tool model
    merged_robot_path = merge_robot_and_tool(tool_filename=tool_filename, base_dir=base_dir, output_robot_tool_filename=robot_and_tool_file_name)
    
    # Add the robot + tool to the scene
    merged_scene_path = inject_robot_tool_into_scene(robot_tool_filename=robot_and_tool_file_name, 
                                                     output_scene_filename=output_scene_filename, 
                                                     base_dir=base_dir)
    
    # Add a piece for screwing
    obstacle_path = os.path.join(base_dir, "ur5e_utils_mujoco/screwing_pieces", obstacle_name)
    add_instance(
        merged_scene_path,
        obstacle_path,
        merged_scene_path,
        mesh_source_dir=os.path.join(base_dir, "ur5e_utils_mujoco/screwing_pieces"),
        mesh_target_dir=os.path.join(base_dir, "ur5e_utils_mujoco/ur5e/assets")
    )
    
    #inserimento disco zona di ricerca
    # path al disco
    disk_path = os.path.join(
      base_dir,
      "ur5e_utils_mujoco/screwing_pieces",
      "Search_area.xml"
    )

    add_instance(
     merged_scene_path,          # file di destinazione (la scena)
     disk_path,                  # file da inserire
     merged_scene_path,          # di nuovo output = scena
     mesh_source_dir="",         # nessuna mesh extra
     mesh_target_dir=""
    )



    # Create the reference frames
    temp_xml_name = create_reference_frames(base_dir, "ur5e_utils_mujoco/" + output_scene_filename, 1)
    model_path = os.path.join(base_dir, "ur5e_utils_mujoco", temp_xml_name)

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data  = mujoco.MjData(model)
    ''''
    ----------------------------DEFINITIONNS FOR BAYESIAN OPTIMIZATION----------------------------

    '''
    xmin = -0.8
    xmax = 0.8
    x_inf = np.array([xmin, xmin])
    x_sup = np.array([xmax, xmax])
    training_samples = 50
    verbose = True
    need_training = True
    local_wrenches=[0,0,1,0,0,1]
    piece_body_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "reference_target_1")
    base_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    tool_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")
    tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tool_site')
    screwdriver_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_top")
    fast_ik_solver = FastIKFlowSolver()
    
    if need_training:
        dataset_csv_path = os.path.join(base_dir, "datasets", f"training_dataset_{training_samples}.csv") # Unique dataset name
        time_start = time.time()
        X_dataset = np.random.uniform(x_inf, x_sup, (training_samples, 2))
        Y_dataset=[]
        viewer = mujoco.viewer.launch_passive(model, data)
        try:
         input("Viewer avviato. Premi Invio per iniziare la visualizzazione delle valutazioni…")

         for j in range(X_dataset.shape[0]):
             mujoco.mj_resetData(model, data)
            
                
   

             # Set robot base (matrix A^w_b)
             t_w_b = np.array([X_dataset[j,0], X_dataset[j,1], 0.1])
             R_w_b = R.from_euler('XYZ', [np.radians(0), np.radians(0), np.radians(0)], degrees=False).as_matrix()
             A_w_b = np.eye(4)
             A_w_b[:3, 3] = t_w_b
             A_w_b[:3, :3] = R_w_b
             set_body_pose(model, data, base_body_id, A_w_b[:3, 3], rotm_to_quaternion(A_w_b[:3, :3]))

             # Set the frame 'screw_top to a new pose wrt flange' and move the screwdriver there
             t_ee_t1 = np.array([0, 0.15, 0]) # 0, 0.15, 0
             R_ee_t1 = R.from_euler('XYZ', [np.radians(30), np.radians(0), np.radians(0)], degrees=False).as_matrix() # 30, 0, 0
             A_ee_t1 = np.eye(4)
             A_ee_t1[:3, 3] = t_ee_t1
             A_ee_t1[:3, :3] = R_ee_t1
             set_body_pose(model, data, screwdriver_body_id, A_ee_t1[:3, 3], rotm_to_quaternion(A_ee_t1[:3, :3]))

             # Fixed transformation 'tool top (t1) => tool tip (t)'
             t_t1_t = np.array([0, 0.0, 0.31])
             R_t1_t = R.from_euler('XYZ', [np.radians(0), np.radians(0), np.radians(0)], degrees=False).as_matrix()
             A_t1_t = np.eye(4)
             A_t1_t[:3, 3] = t_t1_t
             A_t1_t[:3, :3] = R_t1_t

             # Update the position of the tool tip (Just for visualization purposes)
             A_ee_t = A_ee_t1 @ A_t1_t  # combine the two transformations
             set_body_pose(model, data, tool_body_id, A_ee_t[:3, 3], rotm_to_quaternion(A_ee_t[:3, :3]))

             # Piece in the world (define A^w_p) => this is also used to put the frame in space  
             theta_w_p_x_0 = np.radians(180)
             theta_w_p_y_0 = np.radians(0)
             theta_w_p_z_0 = np.radians(45)
             t_w_p = np.array([0.4, 0.4, 0.4]) # [0.2, 0.2, 0.2]
             R_w_p = R.from_euler('XYZ', [theta_w_p_x_0, theta_w_p_y_0, theta_w_p_z_0], degrees=False).as_matrix()
             A_w_p = np.eye(4)
             A_w_p[:3, 3] = t_w_p
             A_w_p[:3, :3] = R_w_p

             # End-effector with respect to wrist3
             t_wl3_ee = np.array([0, 0.1, 0])
             R_wl3_e = R.from_euler('XYZ', [np.radians(-90), 0, 0], degrees=False).as_matrix()
             A_wl3_ee = np.eye(4)
             A_wl3_ee[:3, 3] = t_wl3_ee
             A_wl3_ee[:3, :3] = R_wl3_e

              #! Make inference on the nornmalizing flow (ikflow)   
             #fast_ik_solver = FastIKFlowSolver()       
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

             

                     # first update the marker in the new pose (NOTE: in terms of world coordinates)
             quat_frame = rotm_to_quaternion(A_w_p[:3, :3])
             set_body_pose(model, data, piece_body_id,
                      t_w_p.tolist(),
                      [quat_frame[0], quat_frame[1], quat_frame[2], quat_frame[3]])
             mujoco.mj_forward(model, data)

        # loop over each valid IK solution
             cost = 1e12
             best_cost = 1e12
             best_q = np.zeros(6) 
             start_inference = time.time()
             if params.use_ikflow:
                     for i, (q, x) in enumerate(zip(sols_np, fk_np), 1):
                         if params.verbose:
                            print(f"[OK] sol {i:2d}: q={np.round(q,3)}  →  x={np.round(x,3)}")

                            # apply joint solution
                         data.qpos[:6] = q.tolist()
                         mujoco.mj_forward(model, data)

                         viewer.sync()
                         n_cols = get_collisions(model, data, params.verbose)
                         sigma_manip = inverse_manipulability(q, model, data, tool_site_id)
                         time.sleep(params.show_pose_duration)
                            #print(f"Number of collisions detected: {n_cols}; inverse manipulability: {sigma_manip:.3f}")

                            # Compute the metric for the evaluation
                         if n_cols > 0:
                             cost = 1e12
                         else:
                                cost = sigma_manip

                              # Save the configuration with best inverse manipulability
                         if cost < best_cost:
                             best_cost = cost
                             best_q = q
                         print(f"Sample {j}, solution {i}: cost={cost:.3f}, best_cost={best_cost:.3f}, q={np.round(q, 3)}, n_cols={n_cols}, sigma_manip={sigma_manip:.3f}")

                     print(f"Evaluating collisions and Jacobian on {len(sols_np)} samples lasted {time.time() - start_inference:.2f} seconds")
                     print(f"Best joint configuration for sample {j}: {np.round(best_q, 3)}")
                     # Optimization is over => apply the best configuration
                     data.qpos[:6] = best_q.tolist()
                     mujoco.mj_forward(model, data)
                     viewer.sync()
                     time.sleep(max(1.0, params.show_pose_duration * 3))  # show ~3× più a lungo
                     J = compute_jacobian(model, data, tool_site_id)
                     tau_g = data.qfrc_bias[:6]
                     R_tool_to_world = data.site_xmat[tool_site_id].reshape(3, 3) # Rotation tool => world
                     R_world_to_tool = R_tool_to_world.T # Rotation world => tool
                     world_wrench = get_world_wrench(R_world_to_tool, local_wrenches) #! Wrench in world frame
                     tau_ext = J.T @ world_wrench
                     tau_tot = tau_g + tau_ext
                     if not np.array_equal(best_q, np.zeros(6)):
                        norms=np.linalg.norm(tau_tot) # || tau_tot ||_2 = sqrt(tau1^2 + tau2^2 + ...)
                     else:
                        norms=1e12
                     fitness=float(norms)
                     Y_dataset.append(fitness)
                     
             else:

                      # hard-coded joint configuration for testing
                     q = np.radians([100, -94.96, 101.82, -95.72, -96.35, 180])
                     data.qpos[:6] = q.tolist()
                     mujoco.mj_forward(model, data)

                     # Print collisions
                     n_cols = get_collisions(model, data, params.verbose)
                     print(f"Collisions detected: {n_cols}")
                     viewer.sync()

                    # Compute torques to compensate gravity
                     gravity_comp = data.qfrc_bias[:6]
            
                     if params.verbose: print(f"Gravity compensation torques: {np.round(gravity_comp, 3)}")
        finally:
           viewer.close()

        dataset = np.hstack((X_dataset, Y_dataset))
        df = pd.DataFrame(dataset, columns=["x1", "x2", "y"])
        df.to_csv(dataset_csv_path, index=False)
        time_end = time.time()
        if verbose: print(f"Time taken to generate the initial dataset: {(time_end - time_start)/60} minutes")

    else:
    # Load dataset from a fixed or most recent CSV file
        dataset_csv_path = os.path.join(base_dir, "datasets/training_dataset_300.csv")
    
        if verbose: print(f"Loading dataset from: {dataset_csv_path}")
    
        df = pd.read_csv(dataset_csv_path)
        X_dataset = df[["x1", "x2"]].values
        Y_dataset = df[["y"]].values

    # creation and training of the initial GPR using the dataset above
    kernel = 1.5 * ker.Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=1.0)
    GP = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
    GP.fit(X_dataset, Y_dataset)

if __name__ == "__main__":
    main()



   