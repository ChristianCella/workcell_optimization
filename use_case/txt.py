#!/usr/bin/env python3
import os
import sys
import time
import logging

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

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(base_dir)
import fonts
from transformations import rotm_to_quaternion
from mujoco_utils import set_body_pose, get_collisions, inverse_manipulability, setup_target_frames
from ikflow_inference import FastIKFlowSolver, solve_ik_fast

# ! Wrapper for the simulation

def make_simulator(xml_path, pieces_target_poses, wrench_world):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Get body/site IDs
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")
    ref_body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"reference_target_1"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"reference_target_2")
    ]
    tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")
    screwdriver_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "screw_top")

    #! This method is run for every chromosome of a certain generation, for all generations
    def run_simulation(params: np.ndarray) -> float:

        # Set the new robot base (matrix A^w_b)
        t_w_b = np.array([float(params[0]), float(params[1]), 0])
        R_w_b = R.from_euler('XYZ', [np.radians(0), np.radians(0), np.radians(0)], degrees=False).as_matrix()
        A_w_b = np.eye(4)
        A_w_b[:3, 3] = t_w_b
        A_w_b[:3, :3] = R_w_b
        set_body_pose(model, data, base_body_id, A_w_b[:3, 3], rotm_to_quaternion(A_w_b[:3, :3]))

        # Set the frame 'screw_top to a new pose wrt flange' and move the screwdriver there
        #! Fixed, for the moment
        t_ee_t1 = np.array([0, 0.15, 0])
        R_ee_t1 = R.from_euler('XYZ', [np.radians(30), np.radians(0), np.radians(0)], degrees=False).as_matrix()
        A_ee_t1 = np.eye(4)
        A_ee_t1[:3, 3] = t_ee_t1
        A_ee_t1[:3, :3] = R_ee_t1
        set_body_pose(model, data, screwdriver_body_id, A_ee_t1[:3, 3], rotm_to_quaternion(A_ee_t1[:3, :3]))

        # Fixed transformation 'tool top (t1) => tool tip (t)' (NOTE: the rotation around z is not important)
        t_t1_t = np.array([0, 0.0, 0.26])
        R_t1_t = R.from_euler('XYZ', [np.radians(0), np.radians(0), np.radians(0)], degrees=False).as_matrix()
        A_t1_t = np.eye(4)
        A_t1_t[:3, 3] = t_t1_t
        A_t1_t[:3, :3] = R_t1_t

        # Update the position of the tool tip (Just for visualization purposes)
        A_ee_t = A_ee_t1 @ A_t1_t  # combine the two transformations
        set_body_pose(model, data, tool_body_id, A_ee_t[:3, 3], rotm_to_quaternion(A_ee_t[:3, :3]))

        # End-effector with respect to wrist3 (NOTE: this is always fixed)
        t_wl3_ee = np.array([0, 0.1, 0])
        R_wl3_e = R.from_euler('XYZ', [np.radians(-90), 0, 0], degrees=False).as_matrix()
        A_wl3_ee = np.eye(4)
        A_wl3_ee[:3, 3] = t_wl3_ee
        A_wl3_ee[:3, :3] = R_wl3_e

        # Pieces in the world (define A^w_pi) => this is also used to put the frame in space 
        #! Fixed, for the moment (NOTE: query to the database in the future) 
        #theta_w_p1_x_0 = np.radians(180)
        #theta_w_p1_y_0 = np.radians(0)
        #theta_w_p1_z_0 = np.radians(45)
        t_w_p1 = pieces_target_poses[0][0]
        q_w_p1 = pieces_target_poses[0][1]
        R_w_p1 = R.from_quat(q_w_p1).as_matrix()
        A_w_p1 = np.eye(4)
        A_w_p1[:3, 3] = t_w_p1 
        A_w_p1[:3, :3] = R_w_p1

        #theta_w_p2_x_0 = np.radians(180)
        #theta_w_p2_y_0 = np.radians(0)
        #theta_w_p2_z_0 = np.radians(45)
        t_w_p2 = pieces_target_poses[1][0]
        q_w_p2 = pieces_target_poses[1][1]
        R_w_p2 = R.from_quat(q_w_p2).as_matrix()
        A_w_p2 = np.eye(4)
        A_w_p2[:3, 3] = t_w_p2
        A_w_p2[:3, :3] = R_w_p2
        setup_target_frames(model, data, ref_body_ids, pieces_target_poses)

        # Set the robot to a configuration called q0
        q0 = np.radians([100, -94.96, 101.82, -95.72, -96.35, 180])
        data.qpos[:6] = q0.tolist()
        mujoco.mj_forward(model, data)
        viewer.sync() # ! refresh the scene to see the new chromosome encoding the layout

        #! Optimization of redundancy
        norms = []
        best_configs = []
        for j in range(len(pieces_target_poses)): # ! For each piece to be screwed
            if parameters.verbose: print(f"Solving IK for target frame {j}")

            #! Solve IK for the speficic piece with ikflow
            fast_ik_solver = FastIKFlowSolver() 
            sols_ok, fk_ok = [], []
            for i in range(parameters.N_disc): # 0, 1, 2, ... N_disc-1
                theta_x_0, theta_y_0, theta_z_0 = R.from_quat(pieces_target_poses[j][1]).as_euler('XYZ', degrees=False)
                R_w_p_rotated = R.from_euler('XYZ', [theta_x_0, theta_y_0, theta_z_0 + i * 2 * np.pi / parameters.N_disc], degrees=False).as_matrix()
                A_w_p_rotated = np.eye(4)
                A_w_p_rotated[:3, 3] = pieces_target_poses[j][0]
                A_w_p_rotated[:3, :3] = R_w_p_rotated
                A_b_wl3 = np.linalg.inv(A_w_b) @ A_w_p_rotated @ np.linalg.inv(A_ee_t)@ np.linalg.inv(A_wl3_ee)

                # Craete the target pose for the IK solver (from robot base to wrist_link_3)
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

            # ! Ineference for the specific piece is over: determine the best configuration
            sols_ok = torch.cat(sols_ok, dim=0)
            fk_ok = torch.cat(fk_ok,   dim=0)
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
            #input(f"All the IK solutions for piece {j} have been computed. Press Enter to continue…")               
            # Udate the viewer with the best configuration found
            data.qpos[:6] = best_q.tolist()
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(3.0) # If this is not present, you will never have time to see also the 'optimal' config. for the final piece

            # ! Compute the torques for the best configuration
            Jp = np.zeros((3, model.nv))
            Jr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, Jp, Jr, tool_site_id)
            J6 = np.vstack([Jp, Jr])[:, :6]

            tau_g = data.qfrc_bias[:6]
            tau_ext = J6.T @ wrench_world
            tau_tot = tau_g + tau_ext
            if not np.array_equal(best_q, np.zeros(6)):
                norms.append(np.linalg.norm(tau_tot)) # || tau_tot ||_2 = sqrt(tau1^2 + tau2^2 + ...)
            else:
                norms.append(1e12) #! Ik not feasible => Drive the algorithm away from this configuration

            # Append the best configuration for this piece
            best_configs.append(best_q)
            print(f"Best configuration for piece {j}: {np.round(best_q, 3)} with cost {best_cost:.3f}")

        # All the pieces to be screwed have been processed
        fitness = float(np.mean(norms)) # sum(|| tau_tot ||_2) / N_pieces
        #print(f"Fitness function for a chromosome: {fitness:.6f}")
        print(f"For generation {gen} the best configurations are: {best_configs}")
        return fitness, best_configs

    return run_simulation, model, data, base_body_id

# ───── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    sys.path.append(base_dir)
    xml_path = os.path.join(base_dir, "ur5e_utils_mujoco/scene.xml")

    #! Watch out: either rotations are in 'fixed' frame, or this method IS NOT correct
    pieces_target_poses = [
        (np.array([0.2, 0.2, 0.2]), R.from_euler('XYZ',[180, 0, 0],True).as_quat()),
        (np.array([-0.3, -0.2, 0.3]), R.from_euler('XYZ',[180, 0, 0],True).as_quat()),
    ]
    f = np.array([0.0, 0.0, 10.0]) # ! Only a force in the z direction
    m = np.zeros(3)
    wrench_world = np.hstack([f, m])

    run_sim, model, data, base_body_id = make_simulator(
        xml_path, pieces_target_poses, wrench_world
    )

    # Parameters of the genetic algorithm
    dim = 2 # Number of dimensions of the search space (x_b, y_b)
    x0 = np.zeros(dim)
    sigma0 = 0.5
    popsize = 3 # Number of chromosomes in the population
    max_gens = 4
    opts = {
        "popsize": popsize,
        "bounds": [[-0.3, -0.3], [ 0.3,  0.3]],
        "verb_disp": 0,
    }
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    # ! Start the optimization loop
    with mujoco.viewer.launch_passive(model, data) as viewer:
        input("Press Enter to start optimization…")

        for gen in range(max_gens):
            print(f"Generation of chromosomes number: {gen}")
            sols = es.ask()

            fitnesses = []
            best_configs = []
            for idx, sol in enumerate(sols): # sol contains the x_b, y_b coordinates of the robot base for a chromosome of the batch
                print(f"    • chromosome {idx}: {sol}")
                fit, best_config = run_sim(sol)
                fitnesses.append(fit)
                best_configs.append(best_config)

            print(f"For generation {gen} the fitnesses are: {fitnesses}, and the best fitness is: {min(fitnesses)}")
            print(f"For generation {gen} the complete set of best configurations is: {best_configs}")

            es.tell(sols, fitnesses)
            es.logger.add()
            es.disp()

            # ! Put the robot base in the best position found so far
            i_best = int(np.argmin(fitnesses))
            x_b, y_b  = sols[i_best][:2]
            final_best_configs = best_configs[i_best]
            print(f"The best configurations for generation {gen} is: {final_best_configs}")
            set_body_pose(model, data, base_body_id, [x_b, y_b, 0], rotm_to_quaternion(np.eye(3))) #! To be changed
            mujoco.mj_forward(model, data)
            viewer.sync()

            # 2) **for each** target frame, re-run IK & display
            for idx, (pos, quat) in enumerate(pieces_target_poses):
                data.qpos[:6] = final_best_configs[idx][:6].tolist()
                mujoco.mj_forward(model, data)
                viewer.sync()
                #input(f"Press enter to see the next optimal robot configuration")

            if es.stop():
                break

        res = es.result
        print("\nOptimization terminated:")
        print("  best solution:", res.xbest)
        print(f"  best fitness : {res.fbest:.6f}")

        input("Done — press Enter to exit…")
