import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import mpl_toolkits.mplot3d as tool
import warnings
import sys
import time
import os
import pandas as pd
from datetime import datetime
import matplotlib    
import matplotlib.pyplot as plt
import torch     
import logging

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
warnings.filterwarnings('ignore')

from scipy.spatial.transform import Rotation as R
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
tf.random.set_seed(444)

import cma
import mujoco
import mujoco.viewer

#* ur5e directory
ur5e_utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ur5e_utils_mujoco'))

#* Results directory 
save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results'))

#* Database directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../database')))
from query_db import complete_query

#* Utils directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from constant_parameters import OptimizationParameters, Ur5eRobot, Tools
import fonts
from transformations import rotm_to_quaternion, get_world_wrench, get_homogeneous_matrix, quaternion_to_euler, get_cartesian_pose
from mujoco_utils import set_body_pose, get_collisions, inverse_manipulability, compute_jacobian, scene_manager
from ikflow_inference import FastIKFlowSolver, solve_ik_fast

#* TuRBO directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../TuRBO')))
from turbo.turbo_m import TurboM

#* Path planner directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../tests')))
from rrt_connect_planner import clamp_to_limits, prune_near_duplicates, resample_path_by_count, workspace_length_simple, create_rrt_planner

#* Instances of ikflow model and parameters
global_fast_ik_solver = FastIKFlowSolver()
opt_par = OptimizationParameters()
rob_par = Ur5eRobot()
tool_par = Tools()

#! Wrapper to use mujoco APIs during the optimization
def make_simulator():

    # Query the database
    n_targets, targets_poses, world_wrenches, tool_ids = complete_query()
    world_wrenches = [np.array(w) if w is not None else np.zeros(6) for w in world_wrenches]

    # Path setup 
    model_path = scene_manager("full", n_targets, ur5e_utils_dir, "bringup_ur5e.xml", "extension.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    #* Instance of the rrt planner
    revolute_mask = np.array([model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE for j in np.arange(rob_par.nu, dtype=int)], dtype=bool)
    planner, cc, jnt_range, _ = create_rrt_planner(model, data, n_joints=rob_par.nu, verbose=False, weights=opt_par.weights_rrt, revolute_mask=revolute_mask)

    # Get the ids of all the target locations
    target_body_ids = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and name.startswith("reference_target_"):
            target_body_ids.append(i)

    # Place the static targets in the scene
    for i, body_id in enumerate(target_body_ids):
        t_w_p = np.array([targets_poses[i][0], targets_poses[i][1], targets_poses[i][2]])
        q_frame = [targets_poses[i][6], targets_poses[i][3], targets_poses[i][4], targets_poses[i][5]] 
        theta_w_p_x_0, theta_w_p_y_0, theta_w_p_z_0 = quaternion_to_euler(q_frame, degrees=False)
        R_w_p = R.from_euler('XYZ', [theta_w_p_x_0, theta_w_p_y_0, theta_w_p_z_0], degrees=False).as_matrix()
        A_w_p = np.eye(4)
        A_w_p[:3, 3] = t_w_p
        A_w_p[:3, :3] = R_w_p
        set_body_pose(model, data, body_id, A_w_p[:3, 3], rotm_to_quaternion(A_w_p[:3, :3]))

    # Get body/site IDs
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    tool_base_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_base") #* Base of the tool
    tool_tip_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_tip") #* 'movable' frame in the tool
    tool_tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tool_tip_site')

    ext_base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ext_base") #* Base of the extension tool

    #! This method is run for every individual of a certain generation, for all generations
    def run_simulation(params: np.ndarray) -> float:

        mujoco.mj_resetData(model, data) #* NOTE => you must reset the data at each call

        # Set the new robot base (matrix A^w_b)
        _, _, A_w_b = get_homogeneous_matrix(float(params[0]), float(params[1]), float(params[2]), 0, 0, 0)
        set_body_pose(model, data, base_body_id, A_w_b[:3, 3], rotm_to_quaternion(A_w_b[:3, :3]))

        # Set the hande base with respect to the flange
        _, _, A_ee_t1 = get_homogeneous_matrix(0, 0, 0, 0, 0, 0)
        set_body_pose(model, data, tool_base_body_id, A_ee_t1[:3, 3], rotm_to_quaternion(A_ee_t1[:3, :3]))

        # End-effector with respect to wrist3 (NOTE: this is always fixed)
        _, _, A_wl3_ee = get_homogeneous_matrix(0, 0.1, 0, -90, 0, 0)

        # Constant matrices for the extension tool
        _, _, A_eb_et = get_homogeneous_matrix(0, 0, tool_par.extension_offset, 0, 0, 0)

        # Set the new robot configuration
        q0 = np.radians([-90, -90, -90, -90, 90, 0])
        data.qpos[:6] = q0.tolist()
        mujoco.mj_forward(model, data)
        if opt_par.activate_gui: viewer.sync()

        # Matrix S
        H_mat = np.diag(rob_par.gear_ratios) # Diagonal matrix for gear ratios
        Gamma_mat = np.diag(rob_par.max_torques) # Diagonal matrix for max torques
        S = np.linalg.inv(H_mat.T) @ np.linalg.inv(Gamma_mat.T) @ np.linalg.inv(Gamma_mat) @ np.linalg.inv(H_mat) #! S = H^-T * Gamma^-T * Gamma^-1 * H^-1

        # Start the optimization for the individual
        norms = []
        best_configs = []

        #! First check: collisions of the initial layout (Soft constraint for layout feasibility)
        n_cols_initial = get_collisions(model, data, opt_par.verbose)

        if n_cols_initial > 0:
            if opt_par.verbose: print(f"Initial layout has {n_cols_initial} collisions. Skipping this individual.")

            # Append values that you can associate to this failure (bad initial layout)
            for j in range(len(target_body_ids)):
                norms.append(1e2) 
                best_configs.append(np.zeros(6)) 

            # The fitness will be 'infinite' in this case
            fit_lead_prim = float(np.mean(norms))
            fit_lead_fol = 1e2 / (2 * np.pi * rob_par.robot_reach)
            return fit_lead_prim, fit_lead_fol, best_configs

        else:
            if opt_par.verbose: print(f"Initial layout has no collisions. Proceeding with the optimization.")

            # Counter for the optimization
            counter_pieces_without_cols = 0
            counter_pieces_ik_aval = 0

            for j in range(len(target_body_ids)): # ! For each target location
                if opt_par.verbose: print(f"Solving IK for target frame {j}")

                # Get the pose of the target 
                posit = data.xpos[target_body_ids[j]]
                rotm = data.xmat[target_body_ids[j]].reshape(3, 3)
                theta_x_0, theta_y_0, theta_z_0 = R.from_matrix(rotm).as_euler('XYZ', degrees=True)

                # Decide if the j-th target needs the Finger Tool
                tool_id = tool_ids[j]
                if tool_id == "gripper_hande":
                    gripper_length = tool_par.hande_offset
                elif tool_id == "FingerTool":   
                    gripper_length = tool_par.extension_offset + tool_par.hande_offset

                # Compute matrices
                _, _, A_t1_t = get_homogeneous_matrix(0, 0, gripper_length, 0, 0, 0)
                set_body_pose(model, data, tool_tip_body_id, A_t1_t[:3, 3], rotm_to_quaternion(A_t1_t[:3, :3])) #* Tool tip update
                A_ee_t = A_ee_t1 @ A_t1_t

                # Update the pose of the extension
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
                if opt_par.activate_gui: viewer.sync()

                #! Solve IK for the speficic piece with ikflow
                sols_ok, fk_ok = [], []
                for i in range(opt_par.Nd): 
                    
                    _, _, A_w_p_rotated = get_homogeneous_matrix(posit[0], posit[1], posit[2], theta_x_0, theta_y_0, theta_z_0 + i * 360 / opt_par.Nd)
                    A_b_wl3 = np.linalg.inv(A_w_b) @ A_w_p_rotated @ np.linalg.inv(A_ee_t) @ np.linalg.inv(A_wl3_ee)

                    # Create the target pose for the IK solver (from robot base to wrist_link_3)
                    quat_pose = rotm_to_quaternion(A_b_wl3[:3, :3])
                    target = np.array([
                        A_b_wl3[0, 3], A_b_wl3[1, 3], A_b_wl3[2, 3],   # position
                        quat_pose[0], quat_pose[1], quat_pose[2], quat_pose[3]  # quaternion
                    ], dtype=np.float64)
                    tgt_tensor = torch.from_numpy(target.astype(np.float32))

                    # Solve the IK problem for the discretized pose
                    sols_disc, fk_disc = solve_ik_fast(tgt_tensor, N=opt_par.Ns, fast_solver=global_fast_ik_solver) # Find Ns solutions for this target
                    sols_ok.append(sols_disc)
                    fk_ok.append(fk_disc)

                # ! Inference for the specific piece is over: determine the best configuration
                sols_ok = torch.cat(sols_ok, dim=0)
                fk_ok = torch.cat(fk_ok, dim=0)
                sols_np = sols_ok.cpu().numpy()
                fk_np = fk_ok.cpu().numpy()
                cost_fol = 1e6
                best_cost_fol = 1e6

                # Maybe, no IK solution is available (i.e., the piece is unreachable since outside the workspace)
                best_q = np.zeros(rob_par.nu)
                if len(sols_np) > 0: #! There are IK solutions available

                    counter_pieces_ik_aval += 1 # Increase the counter

                    for i, (q, x) in enumerate(zip(sols_np, fk_np), 1): #! Test each joint configuration
                        if opt_par.verbose: print(f"[OK] sol {i:2d}: q={np.round(q,3)}  →  x={np.round(x,3)}")

                        # apply joint solution
                        data.qpos[:6] = q.tolist()
                        mujoco.mj_forward(model, data)
                        #viewer.sync()
                        #time.sleep(parameters.show_pose_duration)

                        #* Collisions, 'inverse' manipulability and secondary objective
                        n_cols = get_collisions(model, data, opt_par.verbose)

                        # Follower primary objective
                        f_delta_j = inverse_manipulability(q.copy(), model, data, tool_tip_site_id)

                        # Follower secondary objective
                        m  = 0.5 * (rob_par.lb + rob_par.ub)
                        s  = 0.5 * (rob_par.ub - rob_par.lb)
                        diff = q.copy() - m
                        f_q = float(np.sum(opt_par.centering_weights * (diff / s)**2)) 

                        # Total cost for the j-th follower
                        cost_fol = opt_par.weights_follower[0] * f_delta_j + opt_par.weights_follower[1] * f_q

                        # Check if better than the current
                        if (cost_fol < best_cost_fol) and (n_cols == 0):
                            best_cost_fol = cost_fol
                            best_q = q

                    # ! If best cost is not equal to infinite
                    if best_cost_fol < 1e6:
                        counter_pieces_without_cols += 1 # Increase the counter
        
                else: #! No IK solution found, set the best configuration to the default one (all joints at 0)  
                    best_q = np.zeros(6)
                
                # Udate the viewer with the best configuration found
                data.qpos[:6] = best_q.tolist()
                data.qvel[:] = 0  # clear velocities
                data.qacc[:] = 0  # clear accelerations
                data.ctrl[:] = 0  # (if using actuators, may help avoid torque pollution)
                mujoco.mj_forward(model, data)
                if opt_par.activate_gui: viewer.sync()
                if opt_par.activate_gui: time.sleep(1.0) # If this is not present, you will never have time to see also the 'optimal' config. for the final piece

                # ! Compute the torques for the best configuration
                J = compute_jacobian(model, data, tool_tip_site_id)
                tau_g = data.qfrc_bias[:6]
                tau_ext = J.T @ world_wrenches[j][:]
                tau_tot = (tau_ext + tau_g) / (rob_par.gear_ratios * rob_par.max_torques)

                # Check on feasibility: if q = np.zeros(6) => IK failed
                if not np.array_equal(best_q, np.zeros(6)):
                    norms.append(np.linalg.norm(tau_tot))
                else:
                    norms.append(1e2) #! Ik not feasible => Drive the algorithm away from this configuration

                # Append the best configuration for this piece
                best_configs.append(best_q.copy())
                if opt_par.verbose: print(f"Best configuration for piece {j}: {np.round(best_q, 3)} with cost {best_cost_fol:.3f}")

            # ! All the pieces to be screwed have been processed 
            if (counter_pieces_without_cols == len(target_body_ids)) and (counter_pieces_ik_aval == len(target_body_ids)):
                total_length = 0
                q_list_proxy = [q0.copy()] + best_configs.copy()
                for p in range(len(target_body_ids) + 1): 
                    h = p+1
                    if p == len(target_body_ids): 
                        h = 0
                    # Define start and goal
                    q_start = clamp_to_limits(q_list_proxy[p].copy(), jnt_range)
                    q_goal  = clamp_to_limits(q_list_proxy[h].copy(), jnt_range)

                    # Set the start joint config
                    data.qpos[:6] = q_start.tolist()
                    mujoco.mj_forward(model, data)

                    # Get the path
                    try:
                        path, _ = planner.plan(q_start, q_goal, time_budget_s=5.0)
                    except RuntimeError as e:
                        # Typical planner errors (e.g., goal in collision).
                        if opt_par.verbose:
                            print(f"[Planner] {e} — treating as no-path for this segment.")
                        path = None
                    except Exception as e:
                        # Any other unexpected planner issue: degrade gracefully
                        if opt_par.verbose:
                            print(f"[Planner] Unexpected error: {e} — treating as no-path.")
                        path = None

                    if path is not None: #! A path has been found => compute its length

                        # Your existing post-processing
                        path_pruned = prune_near_duplicates(path, min_step=1e-3,
                                                            weights=opt_par.weights_rrt, revolute_mask=revolute_mask)
                        path_uniform = resample_path_by_count(path_pruned, target_points=60,
                                                            weights=opt_par.weights_rrt, revolute_mask=revolute_mask)        

                        # Compute the path length
                        path_length = workspace_length_simple(cc, path_uniform, site_id = tool_tip_site_id)

                    else: #! No path found: probably it did not exist
                        path_length = 5 # 5 meters is surely a lot for the robot (penalization)

                    # Update the total length
                    total_length += path_length

            # Impose to infinite the secondary objective
            else:
                total_length = 1e2

            #! Return the priamry and secondary objectives for the leader
            fit_lead_fol = total_length / (2 * np.pi * 0.85)
            fit_lead_prim = float(np.mean(norms)) 

            return fit_lead_prim, fit_lead_fol, best_configs
    return run_simulation, model, data

'''
Optimization of the workcell layout. 
'''
if __name__ == "__main__":

    #! Create the simulation
    run_sim, model, data = make_simulator()

    #* MuJoCO viewer
    viewer = None
    if opt_par.activate_gui: # Activate GUI 
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(model, data)
        input("Press Enter to start optimization…")
    else: # No GUI needed
        print("Running in headless mode (no GUI).")

    #! Black-box objective function minimized by TuRBO

    def decode(z, center, scale):  return center + scale * z
    def objective_single(x_np_1d_scaled: np.ndarray) -> float:

        #* Function evaluation through the simulator
        center = (opt_par.ub_real + opt_par.lb_real) / 2.0
        scale  = (opt_par.ub_real - opt_par.lb_real) / 2.0
        x_np_1d = decode(x_np_1d_scaled, center, scale)
        f_tau, f_path, *_ = run_sim(x_np_1d)

        #* Evaluate the overall fitness
        fit = f_tau * opt_par.weights_leader[0] + f_path * opt_par.weights_leader[1]

        return float(fit)
  
    #! Optimization
    max_evals = opt_par.init_rand_points + opt_par.n_desired_iterations * opt_par.batch_size
    turbo = TurboM(
        f = objective_single,
        lb = np.ones(opt_par.d) * -1.0,
        ub = np.ones(opt_par.d) * 1.0,
        n_init = opt_par.init_rand_points,
        max_evals = max_evals,
        batch_size = opt_par.batch_size,
        verbose = False,
        use_ard = False,
        device = 'cuda',
        n_training_steps = opt_par.n_training_steps,
        n_trust_regions = opt_par.n_trust_regions
    )

    turbo.optimize() #* Run the optimization

    print("Optimization completed.")






 