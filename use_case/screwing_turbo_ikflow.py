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
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' #! Avoid diplaying useless warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #! Avoid diplaying useless warnings
warnings.filterwarnings('ignore')

from scipy.spatial.transform import Rotation as R
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
tf.random.set_seed(444)

import cma
import mujoco
import mujoco.viewer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scene_manager')))
from parameters import ScrewingTurbo, KukaIiwa14
parameters = ScrewingTurbo()
robot_parameters = KukaIiwa14()
from create_scene import create_scene

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(base_dir)
import fonts
from transformations import rotm_to_quaternion, get_world_wrench, get_homogeneous_matrix,euler_to_quaternion
from mujoco_utils import set_body_pose, get_collisions, inverse_manipulability, compute_jacobian
from ikflow_inference import FastIKFlowSolver, solve_ik_fast

# Import TuRBO
turbo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../TuRBO'))
sys.path.append(turbo_dir)
from turbo.turbo_m import TurboM

# Load a global model
global_fast_ik_solver = FastIKFlowSolver()

# Wrapper
def make_simulator(local_wrenches):

    # Path setup 
    tool_filename = "driller.xml"
    robot_and_tool_file_name = "temp_kuka_with_tool.xml"
    output_scene_filename = "final_scene.xml"
    piece_name1 = "aluminium_plate.xml" 
    piece_name2 = "Linear_guide.xml"
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    # Create the scene

    model_path = create_scene(tool_name=tool_filename, robot_and_tool_file_name=robot_and_tool_file_name,
                              output_scene_filename=output_scene_filename, piece_name1=piece_name1, piece_name2=piece_name2, base_dir=base_dir)

    # Load the newly created model
    model = mujoco.MjModel.from_xml_path(model_path)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Define the parameters for the secondary metric of the followers
    plan_joint_ids = np.arange(robot_parameters.nu, dtype=int)
    jnt_range = model.jnt_range[plan_joint_ids].copy()
    lb = jnt_range[:, 0] # Lower bounds
    ub = jnt_range[:, 1] # Upper bounds
    w_diff = np.ones_like(lb)
    m  = 0.5 * (lb + ub) # Midpoints
    s  = 0.5 * (ub - lb) 
    
    # Get body/site IDs
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")
    piece_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "aluminium_plate")
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
        _, _, A_w_b = get_homogeneous_matrix(0.0, float(params[0]), 0.15, 0.0, 0.0, 90.0)
        set_body_pose(model, data, base_body_id, A_w_b[:3, 3], rotm_to_quaternion(A_w_b[:3, :3]))

        # Set the piece in the environment (matrix A^w_p)
        _, _, A_w_p = get_homogeneous_matrix(0.6, 0.0, 0.8, 0, np.degrees(float(params[2])), 180)
        set_body_pose(model, data, piece_body_id, A_w_p[:3, 3], rotm_to_quaternion(A_w_p[:3, :3]))

        # Set the frame 'screw_top to a new pose wrt flange' and move the screwdriver there
        _, _, A_ee_t1 = get_homogeneous_matrix(0.1, 0.0, float(params[1]), 0.0, 0.0, 0.0)
        set_body_pose(model, data, screwdriver_body_id, A_ee_t1[:3, 3], rotm_to_quaternion(A_ee_t1[:3, :3]))

        # Fixed transformation 'tool top (t1) => tool tip (t)' (NOTE: the rotation around z is not important)
        _, _, A_t1_t = get_homogeneous_matrix(0, 0, 0.26, 0, 0, 0)

        # Update the position of the tool tip (Just for visualization purposes)
        A_ee_t = A_ee_t1 @ A_t1_t
        set_body_pose(model, data, tool_body_id, A_ee_t[:3, 3], rotm_to_quaternion(A_ee_t[:3, :3]))

        # End-effector with respect to wrist3 (NOTE: this is always fixed.)
        # NOTE: for the kuka, ikflow already provides the solution at ee, not wrist
        _, _, A_wl3_ee = get_homogeneous_matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # NOTE: Now q0 is fixed 
        q0 = np.array([2.014, 0.201, -0.098, 1.735, -0.018, -1.607, 0.346])
        data.qpos[:robot_parameters.nu] = q0.tolist()
        mujoco.mj_forward(model, data)
        if parameters.activate_gui: viewer.sync()

        # Matrix S
        gear_ratios = robot_parameters.gear_ratios
        max_torques = robot_parameters.max_torques
        H_mat = np.diag(gear_ratios) # Diagonal matrix for gear artios
        Gamma_mat = np.diag(max_torques) # Diagonal matrix for max torques
        S = np.linalg.inv(H_mat.T) @ np.linalg.inv(Gamma_mat.T) @ np.linalg.inv(Gamma_mat) @ np.linalg.inv(H_mat) #! S = H^-T * Gamma^-T * Gamma^-1 * H^-1

        # Start the optimization for the individual
        norms = []
        best_configs = []
        best_primary_followers = []
        best_secondary_followers = []
        best_followers = []
        best_gravity_torques = [] 
        best_external_torques = [] 
        best_alpha = []
        best_beta = []
        best_gamma = []
        individual_status = []

        #! First check: collisions of the initial layout (Soft constraint for layout feasibility)
        n_cols_initial = get_collisions(model, data, parameters.verbose)

        if n_cols_initial > 0:
            if parameters.verbose: print(f"Initial layout has {n_cols_initial} collisions. Skipping this individual.")

            # Append values that you can associate to this failure (bad initial layout)
            for j in range(len(ref_body_ids)):
                norms.append(1e2) #era 1e12 per il discorso di non corrompere in gpr l'ho messo a 200 ora a 20
                best_configs.append(np.zeros(robot_parameters.nu)) 
                best_gravity_torques.append(1e2 * np.ones(robot_parameters.nu))
                best_external_torques.append(1e2 * np.ones(robot_parameters.nu))
                best_alpha.append(1e2)
                best_beta.append(1e2)
                best_gamma.append(1e2)
                best_followers.append(1e6)
                best_primary_followers.append(1e6)
                best_secondary_followers.append(1e6)

            # The fitness will be infinite in this case
            fit_tau = float(np.mean(norms))
            fit_path = 1e2 / parameters.line_length
            individual_status.append(1) # 1 = layout problem 
            individual_status.append(1000) # Placeholder for impossibility to compute IK
            individual_status.append(1000) # Placeholder for impossibility to verify if IK has collisions
            return fit_tau, fit_path, best_configs, best_followers, best_primary_followers, best_secondary_followers, best_gravity_torques, best_external_torques, individual_status, best_alpha, best_beta, best_gamma

        else:
            if parameters.verbose: print(f"Initial layout has no collisions. Proceeding with the optimization.")
            individual_status.append(0) # 0 = fine, no layout problem

            # Counter to keep track of the status
            counter_pieces_ik_aval = 0
            counter_pieces_without_cols = 0

            for j in range(len(ref_body_ids)): # ! For each piece to be screwed
                if parameters.verbose: print(f"Solving IK for target frame {j}")

                #Get the pose of the target 
                posit = data.xpos[ref_body_ids[j]]
                rotm = data.xmat[ref_body_ids[j]].reshape(3, 3)
                theta_x_0, theta_y_0, theta_z_0 = R.from_matrix(rotm).as_euler('XYZ', degrees=True)

                #! Solve IK for the speficic piece with ikflow
                #fast_ik_solver = FastIKFlowSolver() 
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
                    sols_disc, fk_disc = solve_ik_fast(tgt_tensor, N = parameters.N_samples, fast_solver=global_fast_ik_solver) # Find N solutions for this target
                    sols_ok.append(sols_disc)
                    fk_ok.append(fk_disc)

                # ! Inference for the specific piece is over: determine the best configuration
                sols_ok = torch.cat(sols_ok, dim=0)
                fk_ok = torch.cat(fk_ok, dim=0)
                sols_np = sols_ok.cpu().numpy()
                fk_np = fk_ok.cpu().numpy()
                cost = 1e6
                best_cost = 1e6
                best_man = 1e6
                best_q_diff = 1e6

                # Maybe, no IK solution is available (i.e., the piece is unreachable since outside the workspace)
                best_q = np.zeros(robot_parameters.nu) # This variable will be overwritten
                if len(sols_np) > 0: #! There are IK solutions available

                    counter_pieces_ik_aval += 1 # Increase the counter

                    for i, (q, x) in enumerate(zip(sols_np, fk_np), 1):
                        if parameters.verbose: print(f"[OK] sol {i:2d}: q={np.round(q,3)}  →  x={np.round(x,3)}")

                        # apply joint solution, but do not display it
                        data.qpos[:robot_parameters.nu] = q.tolist()
                        mujoco.mj_forward(model, data)
                        #viewer.sync()
                        #time.sleep(parameters.show_pose_duration)

                        # ! Collisions, 'inverse' manipulability and secondary objective
                        n_cols = get_collisions(model, data, parameters.verbose)

                        # Compute the primary objective
                        f_delta_j = inverse_manipulability(q.copy(), model, data, tool_site_id)

                        # ! Compute the secondary objective
                        diff = q.copy() - m
                        f_q = float(np.sum(w_diff * (diff / s)**2))    # w shape (n,)

                        # Total cost for the j-th follower
                        cost = 10 * f_delta_j + 0.5 * f_q

                        # Check if better than the current
                        if (cost < best_cost) and (n_cols == 0):
                            best_cost = cost
                            best_man = f_delta_j
                            best_q_diff = f_q
                            best_q = q

                    # ! If best cost is not equal to infinite
                    if best_cost < 1e6:
                        counter_pieces_without_cols += 1 # Increase the counter
        
                else: #! No IK solution found, set the best configuration to the default one (all joints at 0)  
                    best_q = np.zeros(robot_parameters.nu) 
                
                # Udate the viewer with the best configuration found
                data.qpos[:robot_parameters.nu] = best_q.tolist()
                data.qvel[:] = 0  # clear velocities
                data.qacc[:] = 0  # clear accelerations
                data.ctrl[:] = 0  # (if using actuators, may help avoid torque pollution)
                mujoco.mj_forward(model, data)
                if parameters.activate_gui: viewer.sync()
                if parameters.activate_gui: time.sleep(1.0) # If this is not present, you will never have time to see also the 'optimal' config. for the final piece

                # ! Compute the torques for the best configuration
                J = compute_jacobian(model, data, tool_site_id)
                tau_g = data.qfrc_bias[:robot_parameters.nu]
                R_tool_to_world = data.site_xmat[tool_site_id].reshape(3, 3) # Rotation tool => world
                R_world_to_tool = R_tool_to_world.T # Rotation world => tool
                world_wrench = get_world_wrench(R_world_to_tool, local_wrenches[j]) #! Wrench in world frame
                tau_ext = J.T @ world_wrench
                #tau_tot = tau_g + tau_ext

                # Compute alpha, beta and gammma
                alpha = world_wrench.T @ J @ S @ J.T @ world_wrench
                beta = 2 * world_wrench.T @ J @ S @ tau_g
                gamma = tau_g.T @ S @ tau_g

                # Check on feasibility: if q = np.zeros(6) => IK failed
                if not np.array_equal(best_q, np.zeros(robot_parameters.nu)):
                    norms.append(np.sqrt(alpha + beta + gamma))
                else:
                    norms.append(1e2) #stesso discorso#! Ik not feasible => Drive the algorithm away from this configuration

                # Append the best configuration for this piece
                best_configs.append(best_q.copy())
                best_followers.append(best_cost)
                best_primary_followers.append(best_man)
                best_secondary_followers.append(best_q_diff)
                best_gravity_torques.append(tau_g.copy())
                best_external_torques.append(tau_ext.copy())

                # Append the values of alpha, beta and gamma
                best_alpha.append(alpha)
                best_beta.append(beta)
                best_gamma.append(gamma)
                if parameters.verbose: print(f"Best configuration for piece {j}: {np.round(best_q, 3)} with cost {best_cost:.3f}")

            # Update the status for the individual 
            individual_status.append(counter_pieces_ik_aval) 
            individual_status.append(counter_pieces_without_cols) 

            # ! All the pieces to be screwed have been processed 
            # If all pieces have valid IK solutions
            if (counter_pieces_without_cols == len(ref_body_ids)) and (counter_pieces_ik_aval == len(ref_body_ids)):
                total_length = (np.abs(params[0] - parameters.starting_position))

            # Impose to infinite the secondary objective
            else:
                total_length = 1e2

            # ! The complete fitness is the sum of f_tau and f_path
            f_path = total_length / (2 * np.pi * 0.85)
            f_tau = float(np.mean(norms)) 
            if parameters.verbose:
                print(f"The primary objective is {f_tau:.4f}")
                print(f"The secondary objective is {f_path:.4f}")
            return f_tau, f_path, best_configs, best_followers, best_primary_followers, best_secondary_followers, best_gravity_torques, best_external_torques, individual_status, best_alpha, best_beta, best_gamma

    return run_simulation, model, data, base_body_id, screwdriver_body_id, piece_body_id, tool_site_id

'''
---------------------------------------PRELIMINARY FUNCTION FOR BAYESIAN OPTIMIZATION---------------------------------------'''
matplotlib.rcParams['mathtext.fontset'] = 'cm' # 'cm' or 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rc('xtick', labelsize = 15)
matplotlib.rc('ytick', labelsize = 15)
matplotlib.rcParams['mathtext.fontset'] = 'cm' # 'cm' or 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rc('xtick', labelsize = 15)
matplotlib.rc('ytick', labelsize = 15)

if __name__ == "__main__":
 
    # Directory to save data
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
  
    local_wrenches = [
        (np.array([0, 0, -150, 0, 0, -10])),
    ]
    
    run_sim, model, data, base_body_id, screwdriver_body_id, piece_body_id, tool_site_id = make_simulator(local_wrenches)

    # Parameters of the genetic algorithm
    lb_real = np.array([-0.75, 0.0, -70 * np.pi/180]) 
    ub_real = np.array([0.75, 0.15, -90 * np.pi/180])

    center = (ub_real + lb_real) / 2.0
    scale  = (ub_real - lb_real) / 2.0

    def encode(x):  return (x - center) / scale
    def decode(z):  return center + scale * z

    ub = np.array([1, 1, 1])
    lb = np.array([-1, -1, -1])
    '''
     ----------------------------Parameters for TuRBO-m--------------------------------
    '''
    
    batch_size = 3
    n_desired_iterations = 3
    n_trust_regions = 5 
    n_training_steps = 50 
    n_init = 5
    max_evals = 30   
    use_ard = False #! Automatic relevance determination 
    verbose_turbo = True
    
    # -----------------------------------------------------------------------------------
    #pesi tella fitness:
    w_tau=10
    w_path=0.5

    viewer = None

    if parameters.activate_gui: # Activate GUI for visualization
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(model, data)
        input("Press Enter to start optimization…")
    else: # The user does not want to see the GUI
        print("Running in headless mode (no GUI).")

    # Lists containing the trend of the best fitness
    eval_artifacts=[]
    best_fitness_trend = []
    best_fitnesses_tau_trend = []
    best_fitnesses_path_trend = []
    best_configs_trend = []
    best_manip_trend = []
    best_gravity_trend = []
    best_external_trend = []
    best_solutions = []
    simulation_status = []
    best_alpha_trend = []
    best_beta_trend = []
    best_gamma_trend = []
    complete_alpha_trend = []
    complete_beta_trend = []
    complete_gamma_trend = []  
    best_individual_idx = [] 
    best_secondary_fit_trend = []
    best_range_trend = []
    starting_fitness = 1e2

    # --- Per-batch accumulators (TuRBO batch bookkeeping) ---
    eval_counter = 0
    current_batch_points = []      # list of (x_np_1d, fit)
    batch_f_tau = []               # list of scalars for the current batch
    batch_f_path = []              # list of scalars for the current batch
    batch_artifacts = []           # list of dicts aligned with current_batch_points
    # Collect simulation status for ALL individuals in the current batch
    batch_sim_status = []  # will be appended to `simulation_status` at end of each batch
    # Complete trends per batch (batch → individuals → pieces)
    batch_complete_alpha = []
    batch_complete_beta  = []
    batch_complete_gamma = []

    # Human-readable names for the 24 decision variables (keep order aligned with params)
    variable_names = [
        "yb", "z_t", "theta_y_p",
    ]

    '''------------------------DEFINIZIONE FUNZIONE DI COSTO---------------------'''
    def objective_single(x_np_1d_scaled: np.ndarray) -> float:
        global eval_counter, current_batch_points, batch_f_tau, batch_f_path, batch_artifacts
        global best_fitness_trend, best_solutions, batch_size, variable_names
        global best_fitnesses_tau_trend, best_fitnesses_path_trend
        global best_configs_trend, best_secondary_fit_trend, best_manip_trend, best_range_trend
        global best_gravity_trend, best_external_trend, simulation_status
        global best_alpha_trend, best_beta_trend, best_gamma_trend
        global batch_sim_status
        global batch_complete_alpha, batch_complete_beta, batch_complete_gamma
        global complete_alpha_trend, complete_beta_trend, complete_gamma_trend
        global starting_fitness
        # Evaluate the simulator
        x_np_1d=decode(x_np_1d_scaled)
        f_tau, f_path, best_configs, best_followers, best_primary_followers, best_secondary_followers, best_gravity_torques, best_external_torques, individual_status, best_alpha, best_beta, best_gamma = run_sim(x_np_1d)

        # Build artifacts for this evaluation
        art = {
            "f_tau": float(f_tau),
            "f_path": float(f_path),
            "best_configs": best_configs,
            "best_followers": best_followers,
            "best_primary_followers": best_primary_followers,
            "best_secondary_followers": best_secondary_followers,
            "best_gravity_torques": best_gravity_torques,
            "best_external_torques": best_external_torques,
            "individual_status": individual_status,
            "best_alpha": best_alpha,
            "best_beta": best_beta,
            "best_gamma": best_gamma,
        }
        # Collect complete (per-individual) alpha/beta/gamma for this batch
        batch_complete_alpha.append(list(art["best_alpha"]))
        batch_complete_beta.append(list(art["best_beta"]))
        batch_complete_gamma.append(list(art["best_gamma"]))
        # Collect per-individual simulation status for this batch
        batch_sim_status.append(art["individual_status"])  # list of ints per individual

        # Scalar fitness used by TuRBO
        fit = f_tau * w_tau + f_path * w_path

        # Accumulate this point into the current batch
        current_batch_points.append((x_np_1d, float(fit)))
        batch_f_tau.append(float(f_tau))
        batch_f_path.append(float(f_path))
        batch_artifacts.append(art)

        # Increase evaluation counter
        eval_counter += 1

        # If we've completed a batch (one TuRBO iteration), log and store the best
        if eval_counter % batch_size == 0:
            # pick the best within the current batch
            best_idx = int(np.argmin([pt[1] for pt in current_batch_points]))
            batch_best_x, batch_best_fit = current_batch_points[best_idx]
            art = batch_artifacts[best_idx]

            # --- BEST-SO-FAR  ---
            if batch_best_fit < starting_fitness or len(best_fitness_trend) == 0:
                starting_fitness = batch_best_fit
                best_fitness_trend.append(starting_fitness)
                best_solutions.append(np.array(batch_best_x, dtype=float))
                best_individual_idx.append(best_idx)

                best_fitnesses_tau_trend.append(art["f_tau"])               # scalar
                best_fitnesses_path_trend.append(art["f_path"])             # scalar
                best_configs_trend.append(art["best_configs"])              # list per piece
                best_secondary_fit_trend.append(art["best_followers"])      # followers cost per piece
                best_manip_trend.append(art["best_primary_followers"])      # manipulability per piece
                best_range_trend.append(art["best_secondary_followers"])    # range-centering per piece
                best_gravity_trend.append(art["best_gravity_torques"])      # list of arrays
                best_external_trend.append(art["best_external_torques"])    # list of arrays
                best_alpha_trend.append(art["best_alpha"])                   # per piece
                best_beta_trend.append(art["best_beta"])                     # per piece
                best_gamma_trend.append(art["best_gamma"])                   # per piece
            else:
                # Repeat previous values to keep a piecewise-constant best-so-far curve
                best_fitness_trend.append(best_fitness_trend[-1])
                best_solutions.append(best_solutions[-1])
                best_individual_idx.append(best_individual_idx[-1])

                best_fitnesses_tau_trend.append(best_fitnesses_tau_trend[-1])
                best_fitnesses_path_trend.append(best_fitnesses_path_trend[-1])
                best_configs_trend.append(best_configs_trend[-1])
                best_secondary_fit_trend.append(best_secondary_fit_trend[-1])
                best_manip_trend.append(best_manip_trend[-1])
                best_range_trend.append(best_range_trend[-1])
                best_gravity_trend.append(best_gravity_trend[-1])
                best_external_trend.append(best_external_trend[-1])
                best_alpha_trend.append(best_alpha_trend[-1])
                best_beta_trend.append(best_beta_trend[-1])
                best_gamma_trend.append(best_gamma_trend[-1])

            # Persist ALL individuals' statuses for this batch (always append, like christian_so)
            simulation_status.append(batch_sim_status)
            batch_sim_status = []

            # Persist complete trends for this batch (batch → individuals → pieces)
            complete_alpha_trend.append(batch_complete_alpha)
            complete_beta_trend.append(batch_complete_beta)
            complete_gamma_trend.append(batch_complete_gamma)

            # Reset batch complete collectors
            batch_complete_alpha = []
            batch_complete_beta = []
            batch_complete_gamma = []

            y_b, z_t, theta_y_p = batch_best_x
            iter_idx = len(best_fitness_trend)
            print(f"[Iter {iter_idx:03d}] best_fitness_so_far = {best_fitness_trend[-1]:.6f} (batch best = {batch_best_fit:.6f})")
            for idx, (name, val) in enumerate(zip(variable_names, batch_best_x)):
                try:
                    print(f"  {idx:02d} {name:>10s} = {float(val): .6f}")
                except Exception:
                    print(f"  {idx:02d} {name:>10s} = {val}")

            # Set the robot base to the best solution
            set_body_pose(model, data, base_body_id, [0.0, y_b, 0.15], euler_to_quaternion(0, 0, np.pi/2)) 
            mujoco.mj_forward(model, data)

            # Set the piece to the best position
            set_body_pose(model, data, piece_body_id, [0.6, 0.0, 0.8], euler_to_quaternion(0, theta_y_p, np.pi)) 
            mujoco.mj_forward(model, data)

            # Set the tool to the best position
            set_body_pose(model, data, screwdriver_body_id, [0.1, 0.0, z_t], euler_to_quaternion(0, 0, 0)) 
            mujoco.mj_forward(model, data)
            if viewer: viewer.sync()

            # Use the per-piece best configurations returned by the simulator
            final_best_configs = art["best_configs"]  # list/array of shape (n_pieces, 6)

            # For each screw/piece, apply the best configuration found
            n_pieces_to_show = min(len(local_wrenches), len(final_best_configs))
            for idx in range(n_pieces_to_show):
                q_best_piece = np.asarray(final_best_configs[idx], dtype=float).ravel()[:robot_parameters.nu]
                data.qpos[:robot_parameters.nu] = q_best_piece.tolist()
                mujoco.mj_forward(model, data)
                if viewer: viewer.sync()

            # reset for next batch
            current_batch_points = []
            batch_f_tau = []
            batch_f_path = []
            batch_artifacts = []

        return float(fit)
    '''---------------------------------------------------------------------------------'''
    '''
    -----------------------------INITIALIZATION OF THE OPTIMIZER--------------------------------
    '''
    turbo = TurboM(
        f=objective_single,
        lb=lb,
        ub=ub,
        n_init=n_init,
        max_evals=max_evals,
        batch_size=batch_size,
        verbose=verbose_turbo,
        use_ard=use_ard,
        device='cuda',
        n_training_steps=n_training_steps,
        n_trust_regions=n_trust_regions
    )

    # ---------------- Timing & output bookkeeping ----------------
    opt_start_time = time.time()
    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    '''
    -----------------------------OPTIMIZATION--------------------------------
    '''
    # ---------------- Run optimization ----------------
    turbo.optimize()

    # ---------------- Finalize logs (handle incomplete last batch) ----------------
    if len(current_batch_points) > 0:
        best_idx = int(np.argmin([pt[1] for pt in current_batch_points]))
        batch_best_x, batch_best_fit = current_batch_points[best_idx]
        art = batch_artifacts[best_idx]

        if batch_best_fit < starting_fitness or len(best_fitness_trend) == 0:
            starting_fitness = batch_best_fit
            best_fitness_trend.append(starting_fitness)
            best_solutions.append(np.array(batch_best_x, dtype=float))
            best_individual_idx.append(best_idx)

            best_fitnesses_tau_trend.append(art["f_tau"])               # scalar
            best_fitnesses_path_trend.append(art["f_path"])             # scalar
            best_configs_trend.append(art["best_configs"])              # list per piece
            best_secondary_fit_trend.append(art["best_followers"])      # followers cost per piece
            best_manip_trend.append(art["best_primary_followers"])      # manipulability per piece
            best_range_trend.append(art["best_secondary_followers"])    # range-centering per piece
            best_gravity_trend.append(art["best_gravity_torques"])      # list of arrays
            best_external_trend.append(art["best_external_torques"])    # list of arrays
            best_alpha_trend.append(art["best_alpha"])                   # per piece
            best_beta_trend.append(art["best_beta"])                     # per piece
            best_gamma_trend.append(art["best_gamma"])                   # per piece
        else:
            best_fitness_trend.append(best_fitness_trend[-1])
            best_solutions.append(best_solutions[-1])
            best_individual_idx.append(best_individual_idx[-1])

            best_fitnesses_tau_trend.append(best_fitnesses_tau_trend[-1])
            best_fitnesses_path_trend.append(best_fitnesses_path_trend[-1])
            best_configs_trend.append(best_configs_trend[-1])
            best_secondary_fit_trend.append(best_secondary_fit_trend[-1])
            best_manip_trend.append(best_manip_trend[-1])
            best_range_trend.append(best_range_trend[-1])
            best_gravity_trend.append(best_gravity_trend[-1])
            best_external_trend.append(best_external_trend[-1])
            best_alpha_trend.append(best_alpha_trend[-1])
            best_beta_trend.append(best_beta_trend[-1])
            best_gamma_trend.append(best_gamma_trend[-1])

        # Persist remaining per-individual statuses for the last (partial) batch
        if len(batch_sim_status) > 0:
            simulation_status.append(batch_sim_status)
            batch_sim_status = []

        # Persist remaining complete trends for the last (partial) batch
        if len(batch_complete_alpha) > 0 or len(batch_complete_beta) > 0 or len(batch_complete_gamma) > 0:
            complete_alpha_trend.append(batch_complete_alpha)
            complete_beta_trend.append(batch_complete_beta)
            complete_gamma_trend.append(batch_complete_gamma)
            batch_complete_alpha = []
            batch_complete_beta = []
            batch_complete_gamma = []

        iter_idx = len(best_fitness_trend)
        print(f"[Iter {iter_idx:03d}] best_fitness_so_far = {best_fitness_trend[-1]:.6f} (last batch best = {batch_best_fit:.6f})")

        # reset batch accumulators for cleanliness
        current_batch_points = []
        batch_f_tau = []
        batch_f_path = []
        batch_artifacts = []
    #------------------OPTIMIZATION COMPLETED-----------------
    elapsed_s = time.time() - opt_start_time
    print(f"Ottimizzazione completata in {elapsed_s:.2f} s (\u2248 {elapsed_s/60.0:.2f} min).")

    # ---------------- Persist results to CSV ----------------
    df_fit = pd.DataFrame(best_fitness_trend, columns=["fitness"])
    df_fit.to_csv(os.path.join(save_dir, "results/data", f"{parameters.csv_directory}", f"fitness_fL.csv"), index=False)

    df_fit = pd.DataFrame(best_fitnesses_tau_trend, columns=["fitness"])
    df_fit.to_csv(os.path.join(save_dir, "results/data", f"{parameters.csv_directory}", f"Primary_leader.csv"), index=False)

    df_fit = pd.DataFrame(best_fitnesses_path_trend, columns=["fitness"])
    df_fit.to_csv(os.path.join(save_dir, "results/data", f"{parameters.csv_directory}", f"Secondary_leader.csv"), index=False)

        # Follower problems scalarized trend
    n_pieces = len(best_secondary_fit_trend[0])
    cols = [f"piece{p+1}" for p in range(n_pieces)]

    df = pd.DataFrame(best_secondary_fit_trend, columns=cols)
    df.insert(0, "generation", range(len(best_secondary_fit_trend)))  # optional but handy

    out_dir = os.path.join(save_dir, "results", "data")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{parameters.csv_directory}", f"fitness_followers.csv"), index=False)

        # Primary indicator follower (manipulability)
    n_pieces = len(best_manip_trend[0])
    cols = [f"piece{p+1}" for p in range(n_pieces)]

    df = pd.DataFrame(best_manip_trend, columns=cols)
    df.insert(0, "generation", range(len(best_manip_trend)))  # optional but handy

    out_dir = os.path.join(save_dir, "results", "data")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{parameters.csv_directory}", f"best_primary_followers.csv"), index=False)

        # Secondary indicator follower (center in the range)
    n_pieces = len(best_range_trend[0])
    cols = [f"piece{p+1}" for p in range(n_pieces)]

    df = pd.DataFrame(best_range_trend, columns=cols)
    df.insert(0, "generation", range(len(best_range_trend)))  # optional but handy

    out_dir = os.path.join(save_dir, "results", "data")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{parameters.csv_directory}", f"best_secondary_followers.csv"), index=False)

        # Save the complete trends of alpha beta and gamma
    def save_trend_wide(trend_data, filename):
            if not trend_data:
                print(f"No data to save for {filename} — skipping.")
                return
            flat_rows = []
            for pieces in trend_data:
                row = []
                for joints in pieces:
                    row.extend(joints)
                flat_rows.append(row)

            n_pieces = len(trend_data[0])
            n_joints = len(trend_data[0][0])
            columns = [f"piece{p+1}_joint{j+1}" for p in range(n_pieces) for j in range(n_joints)]

            df = pd.DataFrame(flat_rows, columns=columns)
            df.to_csv(os.path.join(save_dir, "results/data", f"{parameters.csv_directory}", filename), index=False)

    save_trend_wide(complete_alpha_trend, f"complete_alpha_trend_wide.csv")
    save_trend_wide(complete_beta_trend, f"complete_beta_trend_wide.csv")
    save_trend_wide(complete_gamma_trend, f"complete_gamma_trend_wide.csv")

        # Save the list of indices
    df_best = pd.DataFrame({"generation": range(len(best_individual_idx)),
                        "best_individual": best_individual_idx})
    df_best.to_csv(os.path.join(save_dir, "results/data", f"{parameters.csv_directory}", f"best_individuals_indices.csv"), index=False)


        # Flatten the best_gravity_trend into rows: [generation, individual, tau1, ..., tau6]
    flat_gravity = []
    for generation in best_gravity_trend:
            # Flatten all piece torques into a single row for that generation
            row = []
            for tau in generation:  # tau: array of 6 values
                row.extend(tau.tolist())
            flat_gravity.append(row)

        # Create column names like tau_g_piece0_1, ..., tau_g_piece3_6
    n_pieces = len(best_gravity_trend[0])  # Should match best_external_trend[0]
    n_joints = robot_parameters.nu 

        # ---- Gravity Torques ----
    flat_gravity = []
    for generation in best_gravity_trend:
            row = []
            for tau in generation:  # Each tau is a NumPy array of 6 gravity torques
                row.extend(tau.tolist())
            flat_gravity.append(row)

        # Column names: tau_g_piece0_1 ... tau_g_piece3_6
    columns_grav = [f"tau_g_piece{p}_{j+1}" for p in range(n_pieces) for j in range(n_joints)]

    df_gravity = pd.DataFrame(flat_gravity, columns=columns_grav)
    df_gravity.to_csv(os.path.join(save_dir, "results/data", f"{parameters.csv_directory}", f"best_gravity_torques.csv"), index=False)

        # ---- External Torques ----
    flat_external = []
    for generation in best_external_trend:
            row = []
            for tau in generation:  # Each tau is a NumPy array of 6 external torques
                row.extend(tau.tolist())
            flat_external.append(row)

    columns_ext = [f"tau_ext_piece{p}_{j+1}" for p in range(n_pieces) for j in range(n_joints)]

    df_external = pd.DataFrame(flat_external, columns=columns_ext)
    df_external.to_csv(os.path.join(save_dir, "results/data", f"{parameters.csv_directory}", f"best_external_torques.csv"), index=False)

        # Best configuration trend
    df_x = pd.DataFrame(best_solutions, columns=["y_b", "z_t", "theta_y_p"])
    df_x.to_csv(os.path.join(save_dir, "results/data", f"{parameters.csv_directory}", f"best_solutions.csv"), index=False)

        # Status of each individual in each generation
    stringified_status = [
            [str(individual) for individual in generation]
            for generation in simulation_status
        ]
    popsize_string = len(stringified_status[0])
    df_status = pd.DataFrame(stringified_status, columns=[f"ind_{i}" for i in range(popsize_string)])
    df_status.to_csv(os.path.join(save_dir, "results/data", f"{parameters.csv_directory}", f"simulation_status.csv"), index=False)

    # Save the best configurations (one row per batch, one column per piece)
    if best_configs_trend:
     df_configs = pd.DataFrame(best_configs_trend,
                              columns=[f"config_{i}" for i in range(len(best_configs_trend[0]))])
     df_configs.to_csv(os.path.join(save_dir, "results/data", f"{parameters.csv_directory}",
                                   f"best_configs.csv"),
                      index=False)
    else:
     print("No best_configs_trend to save — skipping best_configs CSV.")

# Save the total optimization time (in seconds)
    df_time = pd.DataFrame([[elapsed_s]], columns=["total_time"])
    df_time.to_csv(os.path.join(save_dir, "results/data", f"{parameters.csv_directory}",
                            f"total_time.csv"),
               index=False)




    print("\nOptimization terminated:")
    if parameters.verbose:
            print(f"f_min hand computed: {best_fitness_trend[-1]:.6f}")
            print(f"x_best hand computed {best_solutions[-1]}")
            print(f"q_best for x_best: {best_configs_trend[-1]}")
            print(f"tau_g at x_best: {best_gravity_trend[-1]}")
            print(f"tau_ext at x_best: {best_external_trend[-1]}")
                   
        # Display the resulting configuration
    if viewer:
            input("Press enter to visualize the result ...")
            mujoco.mj_resetData(model, data)

            # Set robot base
            set_body_pose(model, data, base_body_id, [0.0, float(best_solutions[-1][0]), 0.15], euler_to_quaternion(0, 0, np.pi/2)) 
            
            # Set piece
            set_body_pose(model, data, piece_body_id, [0.6, 0.0, 0.8], euler_to_quaternion(0, float(best_solutions[-1][2]), np.pi))

            # Set tool
            set_body_pose(model, data, screwdriver_body_id, [0.1, 0.0, float(best_solutions[-1][1])], euler_to_quaternion(0, 0, 0))

            # NOTE: put tool frame in the correct position (otherwise the fitness you compute is wrong!!)
            _, _, A_ee_t1 = get_homogeneous_matrix(0.1, 0.0, float(best_solutions[-1][1]), 0, 0, 0)
            _, _, A_t1_t  = get_homogeneous_matrix(0, 0, 0.26, 0, 0, 0)
            A_ee_t = A_ee_t1 @ A_t1_t
            tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_frame")
            set_body_pose(model, data, tool_body_id, A_ee_t[:3, 3], rotm_to_quaternion(A_ee_t[:3, :3]))

            # Set robot joints
            q0_final = np.array([2.014, 0.201, -0.098, 1.735, -0.018, -1.607, 0.346])
            data.qpos[:robot_parameters.nu] = q0_final.tolist()
            mujoco.mj_forward(model, data)

            norms = []
            gear_ratios = robot_parameters.gear_ratios
            max_torques = robot_parameters.max_torques
            for idx in range(len(local_wrenches)):
                print(f"Layout for piece {idx+1}")
                data.qpos[:robot_parameters.nu] = best_configs_trend[-1][idx][:robot_parameters.nu].tolist()
                data.qvel[:] = 0
                data.qacc[:] = 0
                data.ctrl[:] = 0
                mujoco.mj_forward(model, data)
                viewer.sync()

                # ! Compute the torques for the best configuration
                tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")
                J = compute_jacobian(model, data, tool_site_id)

                tau_g = data.qfrc_bias[:robot_parameters.nu]
                R_tool_to_world = data.site_xmat[tool_site_id].reshape(3, 3) # Rotation tool => world
                R_world_to_tool = R_tool_to_world.T # Rotation world => tool
                world_wrench = get_world_wrench(R_world_to_tool, local_wrenches[idx]) #! Wrench in world frame
                tau_ext = J.T @ world_wrench
                tau_tot = tau_g + tau_ext

                norms.append(np.linalg.norm(tau_tot / (np.array(gear_ratios) * np.array(max_torques)))) # || tau_tot ||_2 = sqrt(tau1^2 + tau2^2 + ...)
                input(f"Press Enter to see the next piece configuration (piece {idx+1})…")
            print(f"f obtained testing the layout: {float(np.mean(norms))}")
                