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

#* Base directory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

#* Scene manager
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scene_manager')))
from create_scene import create_scene
from parameters import UseCaseData, Ur5eRobot, Tools

#* Results  
save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results'))

#* Utils 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from transformations import rotm_to_quaternion, get_homogeneous_matrix
from mujoco_utils import set_body_pose, get_collisions, inverse_manipulability, compute_jacobian
from ikflow_inference import FastIKFlowSolver, solve_ik_fast
import fonts

#* TuRBO 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../TuRBO')))
from turbo.turbo_m import TurboM

#* Instances of ikflow model and parameters
global_fast_ik_solver = FastIKFlowSolver()
opt_par = UseCaseData()
rob_par = Ur5eRobot()
tool_par = Tools()

#* Constant matrices
_, _, A_wl3_ee = get_homogeneous_matrix(0, 0.1, 0, -90, 0, 0)
_, _, A_ee_t1 = get_homogeneous_matrix(-0.06, 0.0, tool_par.fixed_radius, 0.0, -90.0, 90.0)
_, _, A_t2_t = get_homogeneous_matrix(0, -0.195, 0.028, 90.0, 0.0, 0.0)

#* Half-ranges and mid-ranges of the robot
m  = 0.5 * (rob_par.lb + rob_par.ub)
s  = 0.5 * (rob_par.ub - rob_par.lb)

'''
Functions for the optimization.
'''

#! Domain scaling for TuRBO
def decode(z, center, scale):  return center + scale * z

#! Wrapper to use mujoco APIs during the optimization
def make_simulator(world_wrenches):

    # Path setup 
    tool_filename = "screwdriver_marco.xml"
    robot_and_tool_file_name = "temp_ur5e_with_tool.xml"
    output_scene_filename = "final_scene.xml"
    piece_name = "plate.xml" 

    # Create the scene
    model_path = create_scene(tool_name=tool_filename, robot_and_tool_file_name=robot_and_tool_file_name,
                              output_scene_filename=output_scene_filename, piece_name=piece_name, base_dir=base_dir)

    # Load the newly created model
    model = mujoco.MjModel.from_xml_path(model_path)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Get the ids of the screws in the xml file
    target_body_ids = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and name.startswith("hole_") and name.endswith("_frame_body"):
            target_body_ids.append(i)

    # Get body & site IDs
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")
    piece_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
    tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")
    screwdriver_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_top")


    #! This method is run for every individual of a certain generation, for all generations
    def run_simulation(params: np.ndarray) -> float:
        mujoco.mj_resetData(model, data) 

        # Set robot base wrt world (fixed for this use case)
        _, _, A_w_b = get_homogeneous_matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        set_body_pose(model, data, base_body_id, A_w_b[:3, 3], rotm_to_quaternion(A_w_b[:3, :3]))

        # Set the piece in the environment (matrix A^w_p)
        _, _, A_w_p = get_homogeneous_matrix(float(params[0]), float(params[1]), -0.02, 0.0, 0.0, 0.0) #! Take the correct z value (fixed)
        set_body_pose(model, data, piece_body_id, A_w_p[:3, 3], rotm_to_quaternion(A_w_p[:3, :3]))

        # Rotated frame
        theta = float(params[2])
        _, _, A_t1_t2 = get_homogeneous_matrix(0.0, tool_par.fixed_radius - (tool_par.fixed_radius * np.cos(theta)), -tool_par.fixed_radius * np.sin(theta), np.degrees(theta), 0.0, 0.0)
        A_ee_t2 = A_ee_t1 @ A_t1_t2
        set_body_pose(model, data, screwdriver_body_id, A_ee_t2[:3, 3], rotm_to_quaternion(A_ee_t2[:3, :3]))

        # Set the tip frame
        A_ee_t = A_ee_t1 @ A_t1_t2 @ A_t2_t
        set_body_pose(model, data, tool_body_id, A_ee_t[:3, 3], rotm_to_quaternion(A_ee_t[:3, :3]))

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
            return fit_lead_prim, q_star

        else: #! No initial collision
            if opt_par.verbose: print(f"{fonts.green}Initial layout has no collisions. Proceeding with the optimization.{fonts.reset}")

            # Counters 
            counter_pieces_without_cols = 0
            counter_pieces_ik_aval = 0

            for j in range(len(target_body_ids)): # ! For each target location

                #! Solve IK for the speficic piece with ikflow
                posit = data.xpos[target_body_ids[j]]
                rotm = data.xmat[target_body_ids[j]].reshape(3, 3)
                theta_x_0, theta_y_0, theta_z_0 = R.from_matrix(rotm).as_euler('XYZ', degrees=True)

                sols_ok, fk_ok = [], []
                for i in range(opt_par.Nd):     

                    # Compute target rotated pose around z axis            
                    _, _, A_w_p_rotated = get_homogeneous_matrix(posit[0], posit[1], posit[2], theta_x_0, theta_y_0, theta_z_0 + i * 360 / opt_par.Nd)
                    A_b_wl3 = np.linalg.inv(A_w_b) @ A_w_p_rotated @ np.linalg.inv(A_ee_t) @ np.linalg.inv(A_wl3_ee)

                    # Create the target pose for the IK solver (from robot base to wrist_link_3)
                    quat_pose = rotm_to_quaternion(A_b_wl3[:3, :3])
                    target = np.array([A_b_wl3[0, 3], A_b_wl3[1, 3], A_b_wl3[2, 3], quat_pose[0], quat_pose[1], quat_pose[2], quat_pose[3]], dtype=np.float64)
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

                        # Collisions, 'inverse' manipulability and secondary objective
                        n_cols = get_collisions(model, data, opt_par.verbose)

                        # Follower primary objective
                        fit_fol_prim = inverse_manipulability(q.copy(), model, data, tool_site_id)

                        # Follower secondary objective
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

                if opt_par.activate_gui: viewer.sync()
                if opt_par.activate_gui: time.sleep(1.0) # Pause to show the best configuration found

                # ! Compute the torques for the best configuration
                J = compute_jacobian(model, data, tool_site_id)
                tau_g = data.qfrc_bias[:rob_par.nu]
                tau_ext = J.T @ world_wrenches[j][:]
                tau_tot = (tau_ext + tau_g) / (rob_par.gear_ratios * rob_par.max_torques)

                # Check on feasibility: if q = np.zeros() => IK failed
                if not np.array_equal(best_q, np.zeros(rob_par.nu)):
                    #tau_hat_abs.append(np.linalg.norm(tau_tot)) # Norm 2
                    tau_hat_abs.append(np.max(np.abs(tau_tot[-3:]))) # Norm infinity on the last 3 joints
                else:
                    tau_hat_abs.append(1e2) 

                # Append the best configuration for this piece
                q_star.append(best_q.copy())
                if opt_par.verbose: print(f"Best configuration for piece {j}: {np.round(best_q, 3)} with cost {best_cost_fol:.3f}")

            #! Compute leader metric
            fit_lead_prim = float(np.mean(tau_hat_abs)) 
            
            #* Results for a specific individual of the batch
            return fit_lead_prim, q_star
        
    return run_simulation, model, data

'''
Optimization of the workcell layout. 
'''
if __name__ == "__main__":

    world_wrenches = [(np.array([0.0, 0.0, 0.0, 0.0, 0.0, -2.0]))]

    #* Set the wrapper to valuate one layout 
    run_sim, model, data = make_simulator(world_wrenches)

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
            q_star = np.zeros((1, rob_par.nu))  # Placeholder for testing
        else:
            fit, q_star = run_sim(layout)

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
                    best_so_far_fit_trend.append(best_so_far_fit_trend[-1] if best_so_far_fit_trend else initial_fitness)
                    best_so_far_configurations_trend.append(best_so_far_configurations_trend[-1] if best_so_far_configurations_trend else configuration_batch[best_idx])
                    best_so_far_layout_trend.append(best_so_far_layout_trend[-1] if best_so_far_layout_trend else layout_batch[best_idx])

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

    # Leader fitness trend
    df_fit = pd.DataFrame(best_so_far_fit_trend, columns=["fitness"])
    df_fit.to_csv(os.path.join(save_dir, opt_par.csv_directory, "last_3_joints", f"fitness_trend.csv"), index=False)

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
    df_configs.to_csv(os.path.join(save_dir, opt_par.csv_directory, "last_3_joints", f"best_joints_configs.csv"), index=False)

    # Best layout trend
    df_layout = pd.DataFrame(best_so_far_layout_trend, columns=["xp", "yp", "theta"])
    df_layout.to_csv(os.path.join(save_dir, opt_par.csv_directory, "last_3_joints", f"best_layout.csv"), index=False)

    # NOTE: close viewer before next iteration
    if viewer is not None:
        viewer.close()
        viewer = None










 