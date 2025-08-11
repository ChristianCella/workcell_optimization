'''
This code implements the bayesian optimization for the XY position of the base of the robot.
'''

#!/usr/bin/env python3
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

from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as ker
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process.kernels import WhiteKernel, Matern
from sklearn.preprocessing import StandardScaler

os.environ["PYTORCH_MPS_DEVICE_DISABLED"] = "1"   # ← deve venire PRIMA di import torch
import torch
torch.set_default_device("cpu")       
import logging
#from workcell_optimization.tests.Redundancy.bio_ik2_custom import A_t1_t
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' #! Avoid diplaying useless warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #! Avoid diplaying useless warnings


warnings.filterwarnings('ignore')

from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as ker
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process.kernels import WhiteKernel, Matern

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

# Load a global model
global_fast_ik_solver = FastIKFlowSolver()


'''
---------------------------------------SIMULATION---------------------------------------
This part of the code setup the simulation environment for the UR5e robot with a screwdriver tool.
It returns:
- Avarage of the total torques
- The best configurations for each piece to be screwed
- The best gravity torques
- The best external torques
'''
def make_simulator(local_wrenches):

    # Path setup 
    tool_filename = "screwdriver.xml"
    robot_and_tool_file_name = "temp_ur5e_with_tool.xml"
    output_scene_filename = "final_scene.xml"
    obstacle_name = "table_grip.xml" 
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    # Create the robot + tool model
    _ = merge_robot_and_tool(tool_filename=tool_filename, base_dir=base_dir, output_robot_tool_filename=robot_and_tool_file_name)
    
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
    piece_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table_grip")
    ref_body_ids = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and name.startswith("hole_") and name.endswith("_frame_body"):
            ref_body_ids.append(i)
    print(f"The frames are: {ref_body_ids}")
    tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")
    screwdriver_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_top")

    #! This method is run for every individual of a certain generation, for all generations
    def run_simulation(params: np.ndarray) -> float:

        mujoco.mj_resetData(model, data) #! Reset the simulation data

        # Set the new robot base (matrix A^w_b)
        _, _, A_w_b = get_homogeneous_matrix(float(params[0]), float(params[1]), 0.1, 0, 0, 0)
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
            cost = 1e12
            best_cost = 1e12

            # Maybe, no IK solution is available (i.e., the piece is unreachable since outside the workspace)
            best_q = np.zeros(6) # This variable will be overwritten##############
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
            ###BEST q è la migliore configurazione trovata per il pezzo j--> bisogna calcolare tau 
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
                norms.append(200) #! Ik not feasible => Drive the algorithm away from this configuration

            # Append the best configuration for this piece
            best_configs.append(best_q)
            best_gravity_torques.append(tau_g)
            best_external_torques.append(tau_ext)
            if parameters.verbose: print(f"Best configuration for piece {j}: {np.round(best_q, 3)} with cost {best_cost:.3f}")

        # All the pieces to be screwed have been processed
        fitness = float(np.mean(norms)) # sum(|| tau_tot ||_2) / N_pieces
        if parameters.verbose: print(f"For generation  the best configurations are: {best_configs}")
        return fitness, best_configs, best_gravity_torques, best_external_torques

    return run_simulation, model, data, base_body_id
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

''' The xy have to be inside a circle of radius R '''
'''
def constraint_func1(x):
    return - (x[0]**2 + x[1]**2) + radius**2
'''
def kappa_var(iteration, kappa, it_center, num_iters):
    cen = it_center * num_iters
    return kappa / (1 + np.exp(0.1 * (iteration - cen)))

def constraint_scaled(x_s):
    # x_s è il punto in coordinate scalate (StandardScaler)
    x_orig = scaler_X.inverse_transform([x_s])[0]
    return - (x_orig[0]**2 + x_orig[1]**2) + radius**2

''' Acquisition function for Bayesian Optimization: Upper Confidence Bound (UCB) '''
def UCB(X, GPR_model, kappa,it_center, num_iters,iteration): 

    if len(X.shape) == 1 :

        X = np.expand_dims(X, axis = 0)

    mean, std = GPR_model.predict(X, return_std = True) # this is actually implementing the Surrogate Function

    # adjust the dimensions of the vectors
    mean = mean.flatten()
    std = std.flatten()
    ucb = mean + kappa_var(iteration,kappa,it_center,num_iters) * std
    #ucb=mean + kappa*std

    return ucb

'''Acquisition function for Bayesian Optimization: Expected Improvement (EI)'''
def EI(X, GPR_model, best_y):
    
    if len(X.shape) == 1 :

        X = np.expand_dims(X, axis = 0)
    
    mu, sigma = GPR_model.predict(X, return_std=True)
    mu, sigma = mu.ravel(), sigma.ravel()
    sigma = np.maximum(sigma, 1e-12)     # evita σ=0

    # Obiettivo di MINIMIZZAZIONE
    improvement = mu - best_y - 0.01
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    return ei

def optimize_acquisition(GPR_model, n, anchor_number, best_evaluation, x_inf, x_sup, constraint, kappa,iteration, it_center, num_iters):

    # creation of the random points (n = 100 in the main)
    random_points = np.random.uniform(x_inf, x_sup, (n,2)) # I create a matrix (2) of random numbers from -10 to 10
    #acquisition_values = UCB(random_points, GPR_model, kappa) # I apply the UCB acquisition function to these points
    acquisition_values = UCB(random_points, GPR_model, kappa,it_center,num_iters,iteration) # I apply the EI acquisition function to these points

    # keep the best N = "anchor_number" points
    best_predictions = np.argsort(acquisition_values)[0 : anchor_number] # find their positions
    selected_anchors = random_points[best_predictions] # get the anchor points 
    optimized_points = []
    
    for anchor in selected_anchors :

        # in "acq" store the acquisition function (UCB) evaluated at the i-th anchor point        
        #acq = lambda anchor, GPR_model: UCB(anchor, GPR_model, kappa)
        acq = lambda anchor, GPR_model: UCB(anchor, GPR_model, kappa,it_center, num_iters,iteration)

        """
        Real minimization procedure: the constraints DO NOT work on "Nelder-Mead" method, but, for example, 
        they work with SLSQP
        """      
        result = minimize(acq, anchor, GPR_model, method = 'SLSQP', bounds = ((x_inf[0], x_sup[0]), (x_inf[1], x_sup[1])),constraints=constraint)
        optimized_points.append(result.x)

    optimized_points = np.array(optimized_points)
    #optimized_acquisition_values = UCB(optimized_points, GPR_model, kappa) # get K_RBF of all the opt. points
    optimized_acquisition_values = UCB(optimized_points, GPR_model, kappa, it_center, num_iters, iteration) # get K_RBF of all the opt. points
    best = np.argsort(optimized_acquisition_values)[0]
    
    # The "x_next" is the tentative point taht will respect the constraints   
    x_next = optimized_points[best]

    return np.expand_dims(x_next, axis = 0),optimized_acquisition_values[best] # return the best point and its acquisition value

'''
---------------------------------------MAIN FUNCTION---------------------------------------
'''


if __name__ == "__main__":
 
    '''
    ----------------------INIZIALIZATION-------------------------------------------------
    '''
    # Directory to save data
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    #! This will become a query to a database
    local_wrenches = [
        (np.array([0, 0, -30, 0, 0, -10])),
        (np.array([0, 0, -20, 0, 0, -5])),
        (np.array([0, 0, -30, 0, 0, -10])),
        (np.array([0, 0, -30, 0, 0, -10])),
    ]

    run_sim, model, data, base_body_id = make_simulator(local_wrenches)

    #Params for the Bayesian Optimization
    xmin = -0.8
    xmax = 0.8
    x_inf = np.array([xmin, xmin])
    x_sup = np.array([xmax, xmax])
    training_samples = 300
    kappa = 0.1 # UCB parameter
    n = 500 
    anchor_number = 100
    num_iters = 201 
    step_plot = 0.5
    radius = 0.8
    verbose = True
    need_training = False
    it_center=0.7



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
    '''
    ---------------------------------------TRAINING GPR---------------------------------------
'''
    if need_training:
        dataset_csv_path = os.path.join(base_dir, "datasets", f"training_dataset_{training_samples}.csv") # Unique dataset name
        time_start = time.time()
        X_dataset = np.random.uniform(xmin, xmax, (training_samples, 2))
        Y_dataset=[]
        for i in range(X_dataset.shape[0]):
            #! Run the simulation for each point in the dataset
            fitness, best_configs, best_gravity_torques, best_external_torques = run_sim(X_dataset[i])
            Y_dataset.append(fitness)
            print(f"Fitness for point {i+1}/{X_dataset.shape[0]}: {fitness:.3f}")
        Y_dataset = np.array(Y_dataset).reshape(-1, 1)  
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
   # kernel = 1.5 * ker.Matern(length_scale=1, nu=2.5) # + WhiteKernel(noise_level=1.0)
    kernel = 1.0 * Matern(length_scale=[1.0, 1.0],        
                      length_scale_bounds=(1e-2, 1e2), 
                      nu=2.5) \
         + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))
    
    GP = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_dataset)   
    GP.fit(X_scaled, Y_dataset)


    y_history = [] 
    best_sofar_hist = [] 
    UCB_Story=[] # List to store the UCB values for each iteration
    '''
-----------------------------INIZIO OPTIMIZZAZIONE BAYESIANA------------------------------------
''' 
    x_inf_s = scaler_X.transform([x_inf])[0]   # scala -0.5,0.5  →  ~(-1,1)
    x_sup_s = scaler_X.transform([x_sup])[0]

    for i in range(num_iters): # 0, 1, ..., num_iters-1
        x_inf_s = scaler_X.transform([x_inf])[0]   # scala -0.5,0.5  →  ~(-1,1)
        x_sup_s = scaler_X.transform([x_sup])[0]
        constraint = [{'type': 'ineq', 'fun': constraint_scaled}]
   
        # Get the new "tentative" point  
        best_evaluation = np.min(Y_dataset)
        #scala dei limiti
        
        x_next_s,UCB_S = optimize_acquisition(GP, n, anchor_number, best_evaluation, x_inf_s, x_sup_s, constraint, kappa,i,it_center, num_iters)
        UCB_Story.append(UCB_S) # Store the UCB value for this iteration
        x_next = scaler_X.inverse_transform(x_next_s).flatten() #lo riprto alla scala originale (-0.5, 0.5)
        # Evaluate the new candidate (Perform a new simulation) 
        eval_x_next, best_configs, best_gravity_torques, best_external_torques = run_sim(x_next)
        
        y_history.append(float(eval_x_next)) 

        # If best_sofar_hist is empty, initialize it with the first evaluation, otherwise make a comparison
        current_best = float(eval_x_next) if not best_sofar_hist else min(best_sofar_hist[-1], float(eval_x_next))
        best_sofar_hist.append(current_best)

        if verbose:
         print(f"Tested candidate at iteration {i + 1}: {x_next}")
         print(f"Evaluation associated to the candidate: {eval_x_next}")

        # Augment the dataset
        X_dataset = np.append(X_dataset, x_next.reshape(1,-1), axis = 0)
        Y_new = np.array([[float(eval_x_next)]])
        Y_dataset = np.append(Y_dataset, Y_new, axis = 0)
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_dataset)
        #! Re-train the dataset
        GP = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
        GP.fit(X_scaled, Y_dataset)
        '''
------------------------------SALVATAGGIO DEI RISULTATI------------------------------------      '''


    # Save the dataset in a csv file
    dataset_csv_path = os.path.join(base_dir, "datasets", f"final_dataset_{num_iters}_UCB_KVAR.csv")
    df = pd.DataFrame(np.hstack((X_dataset, Y_dataset)), columns=["x1", "x2", "y"])
    df.to_csv(dataset_csv_path, index=False)
    if verbose:
        print(f"Final dataset saved to: {dataset_csv_path}")

    # Save the history of evaluations
    history_csv_path = os.path.join(base_dir, "datasets", f"history_{num_iters}_UCB_KVAR.csv")
    df_history = pd.DataFrame({"iteration": range(1, len(y_history) + 1), "y": y_history, "best_so_far": best_sofar_hist})
    df_history.to_csv(history_csv_path, index=False)
    if verbose:
        print(f"History of evaluations saved to: {history_csv_path}")
    
    UCB_csv_path = os.path.join(base_dir, "datasets", f"UCB_{num_iters}_UCB_KVAR.csv")
    df_history_ucb = pd.DataFrame({"iteration": range(1, len(UCB_Story) + 1), "UCB":   UCB_Story})
    df_history_ucb.to_csv(UCB_csv_path, index=False)
    if verbose:
        print(f"History of EI evaluations saved to: {UCB_csv_path}")