#!/usr/bin/env python3
import os
import sys
import time
import logging
import pandas as pd

import torch

#from workcell_optimization.tests.Redundancy.bio_ik2_custom import A_t1_t
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' #! Avoid diplaying useless warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #! Avoid diplaying useless warnings

import numpy as np
from scipy.spatial.transform import Rotation as R

import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
tf.random.set_seed(444)

import cma
import mujoco
import mujoco.viewer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scene_manager')))
from parameters import TxtUseCase
parameters = TxtUseCase()
from create_scene import create_reference_frames,  merge_robot_and_tool, inject_robot_tool_into_scene, add_instance

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(base_dir)
import fonts
from transformations import rotm_to_quaternion, get_world_wrench, get_homogeneous_matrix
from mujoco_utils import set_body_pose, get_collisions, inverse_manipulability, compute_jacobian
from ikflow_inference import FastIKFlowSolver, solve_ik_fast

# ! Wrapper for the simulation
def make_simulator(local_wrenches):

    # Path setup 
    tool_filename = "screwdriver.xml"
    robot_and_tool_file_name = "temp_ur5e_with_tool.xml"
    output_scene_filename = "final_scene.xml"
    obstacle_name = "screwing_plate.xml" 
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

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

    # Create the reference frames
    model_path = os.path.join(base_dir, "ur5e_utils_mujoco", output_scene_filename)
    
    model = mujoco.MjModel.from_xml_path(model_path)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Get body/site IDs
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")
    piece_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "screw_plate")
    ref_body_ids = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and name.startswith("hole_") and name.endswith("_frame_body"):
            ref_body_ids.append(i)
    tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")
    screwdriver_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_top")

    #! This method is run for every individual of a certain generation, for all generations
    def run_simulation(params: np.ndarray) -> float:

        mujoco.mj_resetData(model, data) #! Reset the simulation data

        # Set the new robot base (matrix A^w_b)
        _, _, A_w_b = get_homogeneous_matrix(float(params[0]), float(params[1]), 0, 0, 0, 0)
        set_body_pose(model, data, base_body_id, A_w_b[:3, 3], rotm_to_quaternion(A_w_b[:3, :3]))

        # Set the piece in the environment (matrix A^w_p)
        _, _, A_w_p = get_homogeneous_matrix(-0.15, -0.15, 0, 0, 0, 0)
        set_body_pose(model, data, piece_body_id, A_w_p[:3, 3], rotm_to_quaternion(A_w_p[:3, :3]))

        # Set the frame 'screw_top to a new pose wrt flange' and move the screwdriver there
        _, _, A_ee_t1 = get_homogeneous_matrix(0, 0.15, 0, 30, 0, 0)
        set_body_pose(model, data, screwdriver_body_id, A_ee_t1[:3, 3], rotm_to_quaternion(A_ee_t1[:3, :3]))

        # Fixed transformation 'tool top (t1) => tool tip (t)' (NOTE: the rotation around z is not important)
        _, _, A_t1_t = get_homogeneous_matrix(0, 0, 0.33, 0, 0, 0)

        # Update the position of the tool tip (Just for visualization purposes)
        A_ee_t = A_ee_t1 @ A_t1_t  # combine the two transformations
        set_body_pose(model, data, tool_body_id, A_ee_t[:3, 3], rotm_to_quaternion(A_ee_t[:3, :3]))

        # End-effector with respect to wrist3 (NOTE: this is always fixed)
        _, _, A_wl3_ee = get_homogeneous_matrix(0, 0.1, 0, -90, 0, 0)

        # Set the robot to a configuration called q0
        q0 = np.radians([100, -94.96, 101.82, -95.72, -96.35, 180])
        data.qpos[:6] = q0.tolist()
        mujoco.mj_forward(model, data)
        if parameters.activate_gui: viewer.sync()

        #! Optimization of redundancy
        norms = []
        best_configs = []
        best_gravity_torques = [] 
        best_external_torques = [] 
        for j in range(len(ref_body_ids)): # ! For each piece to be screwed
            if parameters.verbose: print(f"Solving IK for target frame {j}")

            #Get the pose of the target 
            posit = data.xpos[ref_body_ids[j]]  # shape: (3,)
            rotm = data.xmat[ref_body_ids[j]].reshape(3, 3)
            theta_x_0, theta_y_0, theta_z_0 = R.from_matrix(rotm).as_euler('XYZ', degrees=True)

            #! Solve IK for the speficic piece with ikflow
            fast_ik_solver = FastIKFlowSolver() 
            sols_ok, fk_ok = [], []
            for i in range(parameters.N_disc): # 0, 1, 2, ... N_disc-1
                
                _, _, A_w_p_rotated = get_homogeneous_matrix(posit[0], posit[1], posit[2], theta_x_0, theta_y_0, theta_z_0 + i * 360 / parameters.N_disc)
                A_b_wl3 = np.linalg.inv(A_w_b) @ A_w_p_rotated @ np.linalg.inv(A_ee_t) @ np.linalg.inv(A_wl3_ee)

                # Create the target pose for the IK solver (from robot base to wrist_link_3)
                quat_pose = rotm_to_quaternion(A_b_wl3[:3, :3])
                target = np.array([
                    A_b_wl3[0, 3], A_b_wl3[1, 3], A_b_wl3[2, 3],   # position
                    quat_pose[0], quat_pose[1], quat_pose[2], quat_pose[3]  # quaternion
                ], dtype=np.float64)
                tgt_tensor = torch.from_numpy(target.astype(np.float32))

                # Solve the IK problem for the discretized pose
                sols_disc, fk_disc = solve_ik_fast(tgt_tensor, N = parameters.N_samples, fast_solver=fast_ik_solver) # Find N solutions for this target
                sols_ok.append(sols_disc)
                fk_ok.append(fk_disc)

            # ! Inference for the specific piece is over: determine the best configuration
            sols_ok = torch.cat(sols_ok, dim=0)
            fk_ok = torch.cat(fk_ok, dim=0)
            sols_np = sols_ok.cpu().numpy()
            fk_np = fk_ok.cpu().numpy()
            cost = 1e12
            best_cost = 1e12

            # Maybe, no IK solution is available (i.e., the piece is unreachable since outside the workspace)
            best_q = np.zeros(6) # This variable will be overwritten
            if len(sols_np) > 0: #! There are IK solutions available
            
                for i, (q, x) in enumerate(zip(sols_np, fk_np), 1):
                    if parameters.verbose:
                        print(f"[OK] sol {i:2d}: q={np.round(q,3)}  →  x={np.round(x,3)}")

                    # apply joint solution, but do not display it
                    data.qpos[:6] = q.tolist()
                    mujoco.mj_forward(model, data)
                    #viewer.sync()
                    #time.sleep(parameters.show_pose_duration)

                    # Collisions and manipulability
                    n_cols = get_collisions(model, data, parameters.verbose)
                    sigma_manip = inverse_manipulability(q, model, data, tool_site_id)

                    # Compute the metric for the evaluation
                    if n_cols > 0:
                        cost = 1e12
                    else:
                        cost = sigma_manip

                    # Save the configuration with best inverse manipulability
                    if cost < best_cost:
                        best_cost = cost
                        best_q = q

            else: #! No IK solution found, set the best configuration to the default one (all joints at 0)               
                best_q = np.zeros(6)
            
            # Udate the viewer with the best configuration found
            data.qpos[:6] = best_q.tolist()
            data.qvel[:] = 0  # clear velocities
            data.qacc[:] = 0  # clear accelerations
            data.ctrl[:] = 0  # (if using actuators, may help avoid torque pollution)
            mujoco.mj_forward(model, data)
            if parameters.activate_gui: viewer.sync()
            if parameters.activate_gui: time.sleep(1.0) # If this is not present, you will never have time to see also the 'optimal' config. for the final piece

            # ! Compute the torques for the best configuration
            J = compute_jacobian(model, data, tool_site_id)
            tau_g = data.qfrc_bias[:6]
            R_tool_to_world = data.site_xmat[tool_site_id].reshape(3, 3) # Rotation tool => world
            R_world_to_tool = R_tool_to_world.T # Rotation world => tool
            world_wrench = get_world_wrench(R_world_to_tool, local_wrenches[j]) #! Wrench in world frame
            tau_ext = J.T @ world_wrench
            tau_tot = tau_g + tau_ext
            if not np.array_equal(best_q, np.zeros(6)):
                norms.append(np.linalg.norm(tau_tot)) # || tau_tot ||_2 = sqrt(tau1^2 + tau2^2 + ...)
            else:
                norms.append(1e12) #! Ik not feasible => Drive the algorithm away from this configuration

            # Append the best configuration for this piece
            best_configs.append(best_q)
            best_gravity_torques.append(tau_g)
            best_external_torques.append(tau_ext)
            if parameters.verbose: print(f"Best configuration for piece {j}: {np.round(best_q, 3)} with cost {best_cost:.3f}")

        # All the pieces to be screwed have been processed
        fitness = float(np.mean(norms)) # sum(|| tau_tot ||_2) / N_pieces
        if parameters.verbose: print(f"For generation {gen} the best configurations are: {best_configs}")
        return fitness, best_configs, best_gravity_torques, best_external_torques

    return run_simulation, model, data, base_body_id


#! Main
if __name__ == "__main__":

    # Directory to save data
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    #! This will become a query to a database
    local_wrenches = [
        (np.array([0, 0, -30, 0, 0, -10])),
        (np.array([0, 0, -20, 0, 0, -5])),
        (np.array([0, 0, -30, 0, 0, -10])),
    ]

    run_sim, model, data, base_body_id = make_simulator(local_wrenches)

    # Parameters of the genetic algorithm
    x0 = parameters.x0
    sigma0 = parameters.sigma0
    popsize = parameters.popsize
    n_iter = parameters.n_iter
    opts = {
        "popsize": popsize,
        "bounds": [[-0.3, -0.4], [0.0,  0.4]],
        "verb_disp": 0,
    }
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    # ! Start the optimization loop
    viewer = None

    if parameters.activate_gui: # Activate GUI for visualization
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(model, data)
        input("Press Enter to start optimization…")
    else: # The user does not want to see the GUI
        print("Running in headless mode (no GUI).")

    # Lists containing the trend of the best fitness
    best_fitness_trend = []
    best_configs_trend = []
    best_gravity_trend = []
    best_external_trend = []
    best_solutions = []
    starting_fitness = 1e12
    
    try:
        for gen in range(n_iter): #! Loop until the maximum number of generations is reached
            print(f"Generation of chromosomes number: {gen}")
            sols = es.ask() # Generate a new population of chromosomes

            # Lists that will contain the results for this generation
            fitnesses = []
            best_configs = []
            best_gravity_torques_gen = []
            best_external_torques_gen = []
            for idx, sol in enumerate(sols): #! For each chromosome in the current generation
                if parameters.verbose: print(f"    • chromosome {idx}: {sol}")
                fit, best_config, best_gravity, best_external = run_sim(sol)
                fitnesses.append(fit)
                best_configs.append(best_config)
                best_gravity_torques_gen.append(best_gravity)
                best_external_torques_gen.append(best_external)

            if parameters.verbose: print(f"For generation {gen} the fitnesses are: {fitnesses}, and the best fitness is: {min(fitnesses)}")
            if parameters.verbose: print(f"For generation {gen} the complete set of best configurations is: {best_configs}")

            # Update the distribution mean, step size and covariance matrix
            es.tell(sols, fitnesses)
            es.logger.add()
            es.disp()

            # Update robot base to best solution
            i_best = int(np.argmin(fitnesses))
            x_b, y_b  = sols[i_best][:2]
            final_best_configs = best_configs[i_best] # Best joint configs for the current generation
            if parameters.verbose: print(f"The best configurations for generation {gen} is: {final_best_configs}")

            # Set the robot base to the best solution
            set_body_pose(model, data, base_body_id, [x_b, y_b, 0], rotm_to_quaternion(np.eye(3))) # ! NOTE: this is not always identity!
            mujoco.mj_forward(model, data)
            if viewer: viewer.sync()

            # For each screw/piece, apply the best configuration found
            for idx in range(len(local_wrenches)):
                data.qpos[:6] = final_best_configs[idx][:6].tolist()
                mujoco.mj_forward(model, data)
                if viewer: viewer.sync()

            # Custom implementation of 'keep track of the trend'
            if min(fitnesses) < starting_fitness:
                starting_fitness = min(fitnesses)
                best_fitness_trend.append(starting_fitness)
                best_solutions.append((x_b, y_b))
                best_configs_trend.append(final_best_configs)
                best_gravity_trend.append(best_gravity_torques_gen[i_best])
                best_external_trend.append(best_external_torques_gen[i_best])
            else:
                if best_fitness_trend:  # Only proceed if the list is non-empty
                    best_fitness_trend.append(best_fitness_trend[-1])
                    best_solutions.append(best_solutions[-1])
                    best_configs_trend.append(best_configs_trend[-1])
                    best_gravity_trend.append(best_gravity_trend[-1])
                    best_external_trend.append(best_external_trend[-1])
                else:
                    print(f"The lists were empty. Appending None to proceed.")
                    best_fitness_trend.append(starting_fitness)
                    best_solutions.append(None)
                    best_configs_trend.append(None)
                    best_gravity_trend.append(None)
                    best_external_trend.append(None)

            #! In case that: the fitness does not improve for too long, the step size is too small, or others, stop the optimization
            if es.stop():
                break

        # Get the best across all generations
        res = es.result

        # Save all the data
        df_fit = pd.DataFrame(best_fitness_trend, columns=["fitness"])
        df_fit.to_csv(os.path.join(save_dir, "results/data", f"best_fitness.csv"), index=False)

        df_x = pd.DataFrame(best_solutions, columns=["x", "y"])
        df_x.to_csv(os.path.join(save_dir, "results/data", f"best_solutions.csv"), index=False)

        print("\nOptimization terminated:")
        #if parameters.verbose:
        print(f"f_min cma-es: {res.fbest:.6f}; f_min hand computed: {best_fitness_trend[-1]:.6f}")
        print(f"x_best cma-es: {res.xbest}; x_best hand computed {best_solutions[-1]}")
        print(f"q_best for x_best: {best_configs_trend[-1]}")
        print(f"tau_g at x_best: {best_gravity_trend[-1]}")
        print(f"tau_ext at x_best: {best_external_trend[-1]}")
                   
        # Display the resulting configuration
        if viewer:
            input("Press enter to visualize the result ...")
            mujoco.mj_resetData(model, data)
            set_body_pose(model, data, base_body_id, [best_solutions[-1][0], best_solutions[-1][1], 0], rotm_to_quaternion(np.eye(3))) # ! NOTE: this is not always identity!
            mujoco.mj_forward(model, data)

            norms = []
            for idx in range(len(local_wrenches)):
                print(f"Layout for piece {idx+1}")
                data.qpos[:6] = best_configs_trend[-1][idx][:6].tolist()
                mujoco.mj_forward(model, data)
                viewer.sync()

                # ! Compute the torques for the best configuration
                tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")
                J = compute_jacobian(model, data, tool_site_id)

                tau_g = data.qfrc_bias[:6]
                R_tool_to_world = data.site_xmat[tool_site_id].reshape(3, 3) # Rotation tool => world
                R_world_to_tool = R_tool_to_world.T # Rotation world => tool
                world_wrench = get_world_wrench(R_world_to_tool, local_wrenches[idx]) #! Wrench in world frame
                tau_ext = J.T @ world_wrench
                tau_tot = tau_g + tau_ext
                norms.append(np.linalg.norm(tau_tot)) # || tau_tot ||_2 = sqrt(tau1^2 + tau2^2 + ...)
                input(f"Press Enter to see the next piece configuration (piece {idx+1})…")
            print(f"f obtained testing the layout: {float(np.mean(norms))}")
                

    finally:
        if viewer:
            viewer.close()

