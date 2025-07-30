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

warnings.filterwarnings('ignore')

from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as ker
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from DE_XY import func
from sklearn.gaussian_process.kernels import WhiteKernel, Matern

matplotlib.rcParams['mathtext.fontset'] = 'cm' # 'cm' or 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rc('xtick', labelsize = 15)
matplotlib.rc('ytick', labelsize = 15)
matplotlib.rcParams['mathtext.fontset'] = 'cm' # 'cm' or 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rc('xtick', labelsize = 15)
matplotlib.rc('ytick', labelsize = 15)

# For relative imports
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_dir)

#! Constraint function: x^2 + y^2 < R^2
def constraint_func1(x): # la base del robot deve essere dentro un cerchio di raggio radius centrato in 0   
    return - (x[0]**2 + x[1]**2) + radius**2

#? Acquisition function => Upper Confidence Bound (UCB)
def UCB(X, GPR_model, kappa) :

    if len(X.shape) == 1 :

        X = np.expand_dims(X, axis = 0)

    mean, std = GPR_model.predict(X, return_std = True) # this is actually implementing the Surrogate Function

    # adjust the dimensions of the vectors
    mean = mean.flatten()
    std = std.flatten()
    ucb = mean + kappa * std

    return ucb

#! Wrapper for the optimization, relying on anchor points
def optimize_acquisition(GPR_model, n, anchor_number, x_inf, x_sup,constraint,kappa):

    # creation of the random points (n = 100 in the main)
    random_points = np.random.uniform(x_inf, x_sup, (n,2)) # I create a matrix (2) of random numbers from -10 to 10
    acquisition_values = UCB(random_points, GPR_model, kappa) # I apply the UCB acquisition function to these points

    # keep the best N = "anchor_number" points
    best_predictions = np.argsort(acquisition_values)[0 : anchor_number] # find their positions
    selected_anchors = random_points[best_predictions] # get the anchor points 
    optimized_points = []
    
    for anchor in selected_anchors :

        # in "acq" store the acquisition function (UCB) evaluated at the i-th anchor point        
        acq = lambda anchor, GPR_model: UCB(anchor, GPR_model, kappa)
        
        """
        Real minimization procedure: the constraints DO NOT work on "Nelder-Mead" method, but, for example, 
        they work with SLSQP
        """      
        result = minimize(acq, anchor, GPR_model, method = 'SLSQP', bounds = ((x_inf[0], x_sup[0]), (x_inf[1], x_sup[1])),constraints=constraint)
        optimized_points.append(result.x)

    optimized_points = np.array(optimized_points)
    optimized_acquisition_values = UCB(optimized_points, GPR_model, kappa) # get K_RBF of all the opt. points
    best = np.argsort(optimized_acquisition_values)[0]
    
    # The "x_next" is the tentative point taht will respect the constraints   
    x_next = optimized_points[best]

    return np.expand_dims(x_next, axis = 0)

#? 'Hidden' function used in the optimization
def hidden_f(X): #questa è per il caso x_next
 
    X = np.atleast_2d(X)                        # garantisce shape (N,2)
    return np.array([func(row[0], row[1]) for row in X])

#? 'hidden' function used for dataset creation
def hidden_f_test(X):
    Y = np.empty(X.shape[0])          
    for i in range(X.shape[0]):
        Y[i] = func(X[i, 0], X[i, 1])
        if verbose: print(f"Evaluating hidden_f at iteration {i}. The sample point is {X[i, 0]}, {X[i, 1]} => {Y[i]}")
    return Y

#! Variables
xmin = -0.5
xmax = 0.5
x_inf = np.array([xmin, xmin])
x_sup = np.array([xmax, xmax])
training_samples = 300
kappa = 0.1
n = 500 
anchor_number = 100
num_iters = 200 
step_plot = 0.5
radius = 0.5
verbose = True
need_training = False
 
#! If the dataset has not been generated yet => train the GP
if need_training:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_csv_path = os.path.join(base_dir, "datasets", f"training_dataset_{timestamp}.csv") # Unique dataset name
    time_start = time.time()
    X_dataset = np.random.uniform(xmin, xmax, (training_samples, 2))
    Y_dataset = hidden_f_test(X_dataset).reshape(-1, 1)
    dataset = np.hstack((X_dataset, Y_dataset))
    df = pd.DataFrame(dataset, columns=["x1", "x2", "y"])
    df.to_csv(dataset_csv_path, index=False)
    time_end = time.time()
    if verbose: print(f"Time taken to generate the initial dataset: {(time_end - time_start)/60} minutes")

else:
    # Load dataset from a fixed or most recent CSV file
    dataset_dir = os.path.join(base_dir, "datasets")
    dataset_files = [f for f in os.listdir(dataset_dir) if f.startswith("training_dataset_") and f.endswith(".csv")]
    
    if not dataset_files:
        raise FileNotFoundError("No training dataset found in the datasets directory.") 

    # Optional: sort to get the most recent one by timestamp in filename
    dataset_files.sort(reverse=True)
    dataset_csv_path = os.path.join(dataset_dir, dataset_files[0])
    
    if verbose: print(f"Loading dataset from: {dataset_csv_path}")
    
    df = pd.read_csv(dataset_csv_path)
    X_dataset = df[["x1", "x2"]].values
    Y_dataset = df[["y"]].values

# creation and training of the initial GPR using the dataset above
kernel = 1.5 * ker.Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=1.0)
GP = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
GP.fit(X_dataset, Y_dataset)

#! Real optimization procedure
y_history = [] 
best_sofar_hist = [] 

for i in range(num_iters): # 0, 1, ..., num_iters-1

    constraint = [{'type': 'ineq', 'fun': constraint_func1}]
   
   # Get the new "tentative" point  
    best_evaluation = np.min(Y_dataset)
    x_next = optimize_acquisition(GP, n, anchor_number, x_inf, x_sup, constraint, kappa)
    
    # Evaluate the new candidate (Perform a new simulation)   
    eval_x_next = hidden_f(x_next).reshape(-1, 1)
    y_history.append(float(eval_x_next)) 

    # If best_sofar_hist is empty, initialize it with the first evaluation, otherwise make a comparison
    current_best = float(eval_x_next) if not best_sofar_hist else min(best_sofar_hist[-1], float(eval_x_next))
    best_sofar_hist.append(current_best)

    if verbose:
        print(f"Tested candidate at iteration {i + 1}: {x_next.flatten()}")
        print(f"Evaluation associated to the candidate: {eval_x_next.flatten()}")

    # Augment the dataset
    X_dataset = np.append(X_dataset, x_next, axis = 0)
    Y_dataset = np.append(Y_dataset, eval_x_next, axis = 0)

    #! Re-train the dataset
    GP = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
    GP.fit(X_dataset, Y_dataset)

# Save the dataset in a csv file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
dataset_csv_path = os.path.join(base_dir, "datasets", f"final_dataset_{timestamp}.csv")
df = pd.DataFrame(np.hstack((X_dataset, Y_dataset)), columns=["x1", "x2", "y"])
df.to_csv(dataset_csv_path, index=False)
if verbose:
    print(f"Final dataset saved to: {dataset_csv_path}")

# Save the history of evaluations
history_csv_path = os.path.join(base_dir, "datasets", f"history_{timestamp}.csv")
df_history = pd.DataFrame({"iteration": range(1, len(y_history) + 1), "y": y_history, "best_so_far": best_sofar_hist})
df_history.to_csv(history_csv_path, index=False)
if verbose:
    print(f"History of evaluations saved to: {history_csv_path}")



