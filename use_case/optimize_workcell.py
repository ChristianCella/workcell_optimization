import numpy as np
import warnings
import sys
import time
import os
import torch     
from scipy.spatial.transform import Rotation as R
import mujoco, mujoco.viewer
import random
import pandas as pd

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
warnings.filterwarnings('ignore')

#* ur5e 
ur5e_utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ur5e_utils_mujoco'))

#* Results  
save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results'))

#* Database 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../database')))
from query_db import complete_query

#* Utils 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from constant_parameters import OptimizationParameters, Ur5eRobot, Tools
from transformations import rotm_to_quaternion, get_homogeneous_matrix, quaternion_to_euler
from mujoco_utils import set_body_pose, get_collisions, inverse_manipulability, compute_jacobian, scene_manager, get_cartesian_pose
from ikflow_inference import FastIKFlowSolver, solve_ik_fast
import fonts

#* TuRBO 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../TuRBO')))
from turbo.turbo_m import TurboM

#* Instances of ikflow model and parameters
global_fast_ik_solver = FastIKFlowSolver()
opt_par = OptimizationParameters()
rob_par = Ur5eRobot()
tool_par = Tools()

#* Constant matrices
_, _, A_wl3_ee = get_homogeneous_matrix(0, 0.1, 0, -90, 0, 0)
_, _, A_eb_et = get_homogeneous_matrix(0, 0, tool_par.extension_offset, 0, 0, 0)

'''
Functions for the optimization.
'''

#! Domain scaling for TuRBO
def decode(z, center, scale):  return center + scale * z

#! Wrapper to use mujoco APIs during the optimization
def make_simulator(n_targets, targets_poses, world_wrenches, tool_ids):

    # Path setup (always use "full")
    model_path = scene_manager("full", n_targets, ur5e_utils_dir, "bringup_ur5e.xml", "extension.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Get the ids of all the target locations
    target_body_ids = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and name.startswith("reference_target_"):
            target_body_ids.append(i)

    # Place the static targets in the scene
    for i, body_id in enumerate(target_body_ids): 
        tetax, tetay, tetaz = quaternion_to_euler([targets_poses[i][6], targets_poses[i][3], targets_poses[i][4], targets_poses[i][5]], degrees=False)
        A_w_p = np.eye(4)
        A_w_p[:3, 3] = np.array([targets_poses[i][0], targets_poses[i][1], targets_poses[i][2]])
        A_w_p[:3, :3] = R.from_euler('XYZ', [tetax, tetay, tetaz], degrees=False).as_matrix()
        set_body_pose(model, data, body_id, A_w_p[:3, 3], rotm_to_quaternion(A_w_p[:3, :3]))

    # Get body & site IDs
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    tool_base_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_base") 
    tool_tip_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_tip")
    tool_tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tool_tip_site')
    ext_base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ext_base") 

    #! This method is run for every individual of a certain generation, for all generations
    def run_simulation(params: np.ndarray) -> float:
        mujoco.mj_resetData(model, data) 

        # Set robot base wrt world
        _, _, A_w_b = get_homogeneous_matrix(float(params[0]), float(params[1]), 0, 0, 0, 0)
        set_body_pose(model, data, base_body_id, A_w_b[:3, 3], rotm_to_quaternion(A_w_b[:3, :3]))

        # Set gripper wrt flange
        _, _, A_ee_t1 = get_homogeneous_matrix(0, 0, 0, 0, 0, 0)
        set_body_pose(model, data, tool_base_body_id, A_ee_t1[:3, 3], rotm_to_quaternion(A_ee_t1[:3, :3]))

        # Home configuration
        q0 = rob_par.home_configuration.copy()
        data.qpos[:rob_par.nu] = q0.tolist()
        mujoco.mj_forward(model, data)
      
        # Optimization for one individual of the batch
        tau_hat_abs = [] 
        q_star = [] 

        #! First check: collisions of the initial layout (Soft constraint for layout feasibility)
        n_cols_initial = get_collisions(model, data, opt_par.verbose)

        if n_cols_initial > 0:
            if opt_par.verbose: print(f"{fonts.red}Initial layout has {n_cols_initial} collisions. Skipping this individual.{fonts.reset}")

            # Append values that you can associate to this failure (bad initial layout)
            for j in range(len(target_body_ids)):
                tau_hat_abs.append(1e2) 
                q_star.append(np.zeros(rob_par.nu)) 

            # The fitness will be 'infinite' in this case
            fit_lead_prim = float(np.mean(tau_hat_abs))
            fit_lead_sec = 1e2 
            return fit_lead_prim, fit_lead_sec, q_star

        else:
            if opt_par.verbose: print(f"{fonts.green}Initial layout has no collisions. Proceeding with the optimization.{fonts.reset}")

            # Counters 
            counter_pieces_without_cols = 0
            counter_pieces_ik_aval = 0

            for j in range(len(target_body_ids)): # ! For each target location
                #print(f"{fonts.yellow}Target frame {j}{fonts.reset}")

                # Check needed tool
                tool_id = tool_ids[j]
                if tool_id == "gripper_hande":
                    gripper_length = tool_par.hande_offset
                    task_redundancy = 2 #* Only 2 possible ways of grasping with the hande

                    # In case of gripper alone, move the extension away
                    _, _, A_w_et = get_homogeneous_matrix(tool_par.detachment_pose[0], tool_par.detachment_pose[1], tool_par.detachment_pose[2], tool_par.detachment_pose[3], tool_par.detachment_pose[4], tool_par.detachment_pose[5])
                    A_w_eb = A_w_et @ np.linalg.inv(A_eb_et)
                    set_body_pose(model, data, ext_base_body_id, A_w_eb[:3, 3], rotm_to_quaternion(A_w_eb[:3, :3]))
                    mujoco.mj_forward(model, data)
                elif tool_id == "FingerTool":   
                    gripper_length = tool_par.extension_offset + tool_par.hande_offset
                    task_redundancy = opt_par.Nd #* Many different ways of using the Finger extension

                # Set tip frame
                _, _, A_t1_t = get_homogeneous_matrix(0, 0, gripper_length, 0, 0, 0)
                set_body_pose(model, data, tool_tip_body_id, A_t1_t[:3, 3], rotm_to_quaternion(A_t1_t[:3, :3])) 
                A_ee_t = A_ee_t1 @ A_t1_t
                if opt_par.activate_gui: viewer.sync()

                #! Solve IK for the speficic piece with ikflow
                sols_ok, fk_ok = [], []
                for i in range(task_redundancy): 
                    tetax, tetay, tetaz = quaternion_to_euler([targets_poses[j][6], targets_poses[j][3], targets_poses[j][4], targets_poses[j][5]], degrees=True)                 
                    _, _, A_w_p_rotated = get_homogeneous_matrix(targets_poses[j][0], targets_poses[j][1], targets_poses[j][2], tetax, tetay, tetaz + i * 360 / task_redundancy)
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

                #* Reachability check
                best_q = np.zeros(rob_par.nu)
                if len(sols_np) > 0: #! There are IK solutions available

                    counter_pieces_ik_aval += 1 # Increase the counter

                    for i, (q, x) in enumerate(zip(sols_np, fk_np), 1): #! Test each joint configuration

                        # apply joint solution
                        data.qpos[:rob_par.nu] = q.tolist()
                        mujoco.mj_forward(model, data)

                        # Update the pose of the extension, if needed
                        if tool_id == "FingerTool":
                            pos, eul = get_cartesian_pose(tool_tip_body_id, data, 'euler')
                            _, _, A_w_et = get_homogeneous_matrix(pos[0], pos[1], pos[2], np.degrees(eul[0]), np.degrees(eul[1]), np.degrees(eul[2]))
                            A_w_eb = A_w_et @ np.linalg.inv(A_eb_et)
                            set_body_pose(model, data, ext_base_body_id, A_w_eb[:3, 3], rotm_to_quaternion(A_w_eb[:3, :3]))
                            mujoco.mj_forward(model, data)

                        #viewer.sync()
                        #input(f"{fonts.cyan}Enter to evaluate next q{fonts.reset}")
                        #time.sleep(parameters.show_pose_duration)

                        # Collisions, 'inverse' manipulability and secondary objective
                        n_cols = get_collisions(model, data, opt_par.verbose)

                        # Follower primary objective
                        fit_fol_prim = inverse_manipulability(q.copy(), model, data, tool_tip_site_id)

                        # Follower secondary objective
                        m  = 0.5 * (rob_par.lb + rob_par.ub)
                        s  = 0.5 * (rob_par.ub - rob_par.lb)
                        diff = q.copy() - m
                        fit_fol_sec = float(np.sum(opt_par.centering_weights * (diff / s)**2)) 

                        # Total cost for the j-th follower
                        cost_fol = opt_par.weights_follower[0] * fit_fol_prim + opt_par.weights_follower[1] * fit_fol_sec

                        # Check if better than the current
                        if (cost_fol < best_cost_fol) and (n_cols == 0):
                            best_cost_fol = cost_fol
                            best_q = q

                    # ! If best cost is not equal to infinite
                    if best_cost_fol < 1e6:
                        counter_pieces_without_cols += 1 
        
                else: #! No IK solution found  
                    best_q = np.zeros(rob_par.nu)
                
                # Update the viewer with the best configuration found
                data.qpos[:rob_par.nu] = best_q.tolist()
                data.qvel[:] = 0
                data.qacc[:] = 0    
                data.ctrl[:] = 0
                mujoco.mj_forward(model, data)

                # Update the pose of the extension, if needed
                if tool_id == "FingerTool":
                    pos, eul = get_cartesian_pose(tool_tip_body_id, data, 'euler')
                    _, _, A_w_et = get_homogeneous_matrix(pos[0], pos[1], pos[2], np.degrees(eul[0]), np.degrees(eul[1]), np.degrees(eul[2]))
                    A_w_eb = A_w_et @ np.linalg.inv(A_eb_et)
                    set_body_pose(model, data, ext_base_body_id, A_w_eb[:3, 3], rotm_to_quaternion(A_w_eb[:3, :3]))
                    mujoco.mj_forward(model, data)

                if opt_par.activate_gui: viewer.sync()
                if opt_par.activate_gui: time.sleep(1.0) # Pause to show the best configuration found

                # ! Compute the torques for the best configuration
                J = compute_jacobian(model, data, tool_tip_site_id)
                tau_g = data.qfrc_bias[:rob_par.nu]
                #* R_tool_to_world = data.site_xmat[tool_tip_site_id].reshape(3, 3) # If the local wrench was given instead
                #* world_wrench = get_world_wrench(R_tool_to_world, local_wrenches[j][:])
                #* tau_ext = J.T @ world_wrenches
                tau_ext = J.T @ world_wrenches[j][:]
                tau_tot = (tau_ext + tau_g) / (rob_par.gear_ratios * rob_par.max_torques)

                # Check on feasibility: if q = np.zeros() => IK failed
                if not np.array_equal(best_q, np.zeros(rob_par.nu)):
                    tau_hat_abs.append(np.linalg.norm(tau_tot))
                else:
                    tau_hat_abs.append(1e2) 

                # Append the best configuration for this piece
                q_star.append(best_q.copy())
                if opt_par.verbose: print(f"Best configuration for piece {j}: {np.round(best_q, 3)} with cost {best_cost_fol:.3f}")

            #! Compute metrics for the leader
            fit_lead_prim = float(np.mean(tau_hat_abs)) 
            fit_lead_sec = 1 - (counter_pieces_without_cols / n_targets)
            
            #* Results for a specific individual of the batch
            return fit_lead_prim, fit_lead_sec, q_star
        
    return run_simulation, model, data

'''
Optimization of the workcell layout. 
'''
if __name__ == "__main__":

    #* Query the database
    n_targets, targets_poses, world_wrenches, tool_ids, clusters = complete_query()
    world_wrenches = [np.array(w) if w is not None else np.zeros(6) for w in world_wrenches]

    if opt_par.theta == 1: #! One single optimization
        list_n_targets       = [n_targets]
        list_target_poses    = [targets_poses]
        list_world_wrenches  = [world_wrenches]
        list_tool_ids       = [tool_ids]
        print(f"The optimziation lasts for {len(list_n_targets)} iterations")

    elif opt_par.theta == 0: #! Optimization for clusters of targets
        #* Organize targets by clusters
        unique_clusters = sorted(set(clusters)) 
        cluster_idx = {c: i for i, c in enumerate(unique_clusters)}

        list_n_targets       = [0] * len(unique_clusters)  
        list_target_poses    = [[] for _ in unique_clusters]
        list_world_wrenches  = [[] for _ in unique_clusters]
        list_tool_ids       = [[] for _ in unique_clusters]

        for pose, wrench, cl, id in zip(targets_poses, world_wrenches, clusters, tool_ids):
            idx = cluster_idx[cl]         
            list_n_targets[idx] += 1
            list_target_poses[idx].append(pose)
            list_world_wrenches[idx].append(wrench)
            list_tool_ids[idx].append(id)
        print(f"The optimziation lasts for {len(list_n_targets)} iterations")
    else:
        raise ValueError("opt_par.theta must be either 0 or 1.")
     
    total_iterations = len(list_n_targets)

    #! Complete optimization
    for iter_opt in range(total_iterations):
        
        print(f"{fonts.purple}Current global iteration:{iter_opt+1}/{total_iterations}{fonts.reset}")

        #* Set the wrapper to valuate one layout 
        run_sim, model, data = make_simulator(n_targets=list_n_targets[iter_opt], targets_poses=list_target_poses[iter_opt], world_wrenches=list_world_wrenches[iter_opt], tool_ids=list_tool_ids[iter_opt])

        #* MuJoCO viewer
        viewer = None
        if opt_par.activate_gui: # Activate GUI 
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(model, data)
            input("Press Enter to start optimization…")
        else: # No GUI needed
            print("Running in headless mode (no GUI).")

        #! Black-box objective function minimized by TuRBO
        initialization_counter = 0
        individual_counter = 0
        iteration_counter = 0
        initial_fitness = 1e2
        fit_batch = []  
        configuration_batch = [] 
        layout_batch = []
        fit_trend = []
        configurations_trend = [] 
        layout_trend = []
        best_so_far_fit_trend = []
        best_so_far_configurations_trend = []
        best_so_far_layout_trend = []

        def objective_single(adim_layout: np.ndarray) -> float:

            #* Define varibales as global to keep their values across calls
            global initialization_counter, individual_counter, iteration_counter, initial_fitness
            global fit_batch, configuration_batch, layout_batch, fit_trend, configurations_trend, layout_trend
            global best_so_far_fit_trend, best_so_far_configurations_trend, best_so_far_layout_trend

            if initialization_counter < opt_par.init_rand_points:
                print(f"{fonts.green}Random initialization phase. Starting evaluation: {initialization_counter + 1}/{opt_par.init_rand_points}{fonts.reset}")
                initialization_counter += 1
            else:
                if opt_par.verbose: print(f"{fonts.red}Iteration: {iteration_counter}/{opt_par.n_desired_iterations}{fonts.reset}")
                if opt_par.verbose: print(f"{fonts.cyan}Individual:{individual_counter + 1}/{opt_par.batch_size}{fonts.reset}")
                individual_counter += 1

            #* Simulate this layout (individual) for all the targets
            layout = decode(adim_layout, opt_par.center, opt_par.scale)
            if opt_par.verbose: print(f"{fonts.yellow}The layout to be tested is: {layout}{fonts.reset}")

            #* Fitness for this individual
            if opt_par.mode == "debugging":
                fit = random.random()  # Placeholder for testing
                q_star = np.zeros((len(complete_query()[1]), 6))  # Placeholder for testing
            else:
                f_tau, f_reach, q_star = run_sim(layout)
                fit = f_tau * opt_par.weights_leader[0] + f_reach * opt_par.weights_leader[1]

            #* Augment datasets for the batch
            fit_batch.append(fit)
            configuration_batch.append(q_star)
            layout_batch.append(layout)

            #* Check if a batch has been filled
            if (initialization_counter == opt_par.init_rand_points) and (individual_counter % opt_par.batch_size == 0):  

                #* if the procedure has already finished initialization       
                if iteration_counter != 0:
                    if opt_par.verbose: print(f"{fonts.blue}Update the iteration counter.{fonts.reset}")
                    best_idx = np.argmin(fit_batch)
                    fit_trend.append(fit_batch[best_idx])
                    configurations_trend.append(configuration_batch[best_idx])
                    layout_trend.append(layout_batch[best_idx])

                    #! Best-so-far trend:
                    if fit_batch[best_idx] < initial_fitness: #* Case 1: improvement
                        initial_fitness = fit_batch[best_idx]
                        best_so_far_fit_trend.append(initial_fitness)
                        best_so_far_configurations_trend.append(configuration_batch[best_idx])
                        best_so_far_layout_trend.append(layout_batch[best_idx])
                    else: #* Case 2: no improvement
                        best_so_far_fit_trend.append(best_so_far_fit_trend[-1])
                        best_so_far_configurations_trend.append(best_so_far_configurations_trend[-1])
                        best_so_far_layout_trend.append(best_so_far_layout_trend[-1])

                    #* Display the status
                    print(f"{fonts.green}Iteration: {iteration_counter}; Best so far: {best_so_far_fit_trend[-1]}{fonts.reset}")
                
                #* Reset counters and batches
                individual_counter = 0
                iteration_counter += 1
                fit_batch = []
                configuration_batch = []
                layout_batch = []

            return float(fit)
    
        #! Optimization
        turbo = TurboM(
            f = objective_single,
            lb = np.ones(opt_par.d) * -1.0,
            ub = np.ones(opt_par.d) * 1.0,
            n_init = opt_par.init_rand_points,
            max_evals = opt_par.max_evals,
            batch_size = opt_par.batch_size,
            verbose = False,
            use_ard = True,
            device = 'cuda',
            n_training_steps = opt_par.n_training_steps,
            n_trust_regions = opt_par.n_trust_regions
        )
        start_time = time.time()
        turbo.optimize() 
        elapsed_time = time.time() - start_time
        print(f"{fonts.green_light}Optimization completed in: {elapsed_time:.2f} seconds{fonts.reset}")

        if opt_par.verbose:
            print(f"{fonts.cyan}Best per iteration: {fit_trend}{fonts.reset}")
            print(f"{fonts.red}Best so far: {best_so_far_fit_trend}{fonts.reset}")

        #! Save data

        # Fitness trend
        df_fit = pd.DataFrame(best_so_far_fit_trend, columns=["fitness"])
        df_fit.to_csv(os.path.join(save_dir, opt_par.csv_directory, f"fitness_cluster_{iter_opt+1}.csv"), index=False)

        # Best joint configurations trend
        configs = np.array(best_so_far_configurations_trend)  # shape: (n_iters, n_targets, n_joints)
        n_iters, n_targets, n_joints = configs.shape

        # Flatten each (n_targets, n_joints) into a 1D vector (length = n_targets * n_joints)
        configs_flat = configs.reshape(n_iters, n_targets * n_joints)

        # Build meaningful column names: target_0_joint_0, target_0_joint_1, ...
        columns = [
            f"t{t}_j{j+1}"
            for t in range(n_targets)
            for j in range(n_joints)
        ]

        df_configs = pd.DataFrame(configs_flat, columns=columns)
        df_configs.to_csv(os.path.join(save_dir, opt_par.csv_directory, f"best_joints_configs_cluster_{iter_opt+1}.csv"), index=False)

        # Best layout trend
        df_layout = pd.DataFrame(best_so_far_layout_trend, columns=["xb", "yb"])
        df_layout.to_csv(os.path.join(save_dir, opt_par.csv_directory, f"best_layout_cluster_{iter_opt+1}.csv"), index=False)

        # NOTE: close viewer before next iteration
        if viewer is not None:
            viewer.close()
            viewer = None










 