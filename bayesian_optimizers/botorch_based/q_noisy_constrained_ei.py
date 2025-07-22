"""
* In this code it's shown how to perform a closed-loop Bayesian optimization example;
* The alternative would be to use "Ax" as shown in the folder "Using BoTorch with Ax";
* We will show how to use the "Batch Expercted Improvement" (qEI) and its "Noisy" counterpart (qNEI);
* The test function is the Hartmann6 function over the domain [0, 1]^6;
* There is also one constraint: ||x||_1 - 3 <= 0 (remember: the norm 1 of a vector is the sum of its absolute values);
* Both objectives and constraints are considered to be noisy (meaning: some uncertainties in the constraints);
* BoTorch assumes a maximization problem: we are going to maximize -f(x)
* We are going to use "tensor", that isa 64-bit floating point, to store the data;
* Remember: 'outcome constraint' is different from 'parameter constraint' ==> look at the pdf in notability
    (BoTorch folder ==> Advanced Topics ==> Constraints)
* We use Monte Carlo sampling because we are in 'batch mode' and standard acquisition functions do not exist:
    they need to be approximated by their MC approximation (Noitability ==> BoTorch folder ==> Basic Concepts in BoTorch
    ==> Acquisition functions)
"""

"""
Import the libraries
"""

# Block 1

import os
import torch
from botorch.test_functions import Hartmann

# Block 2

from botorch.models import FixedNoiseGP, ModelListGP #! FIX this
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

# Block 3

from botorch.acquisition.objective import ConstrainedMCObjective

# Block 4

from botorch.optim import optimize_acqf

# Block 5

from botorch import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning

import time
import warnings

# Block 6

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# LaTeX commands

matplotlib.rcParams['mathtext.fontset'] = 'cm' # 'cm' or 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

matplotlib.rc('xtick', labelsize = 15)
matplotlib.rc('ytick', labelsize = 15)


"""
First checks
* Check "cuda" availability;
* Check "SMOKETEST" ==> type of software testing that determines whether the deployed build is stable or not
"""

# Check "cuda" availability

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
print(f'The selecetd device is: {device}')
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKETEST")

"""
Constants
"""

# Block 2

# NOISE_SE = 0.0
NOISE_SE = 0.5
n_training_points = 10

# Block 4

BATCH_SIZE = 12 if not SMOKE_TEST else 2
NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 32

# Block 5

N_TRIALS = 4 if not SMOKE_TEST else 2
N_BATCH = 30 if not SMOKE_TEST else 2  
MC_SAMPLES = 256 if not SMOKE_TEST else 32

verbose = False

"""
Block 1: Problem setup
"""

# import the test function (-f(x) to have a minimization problem)

neg_hartmann6 = Hartmann(negate = True)

# Define the constraints (they need to be specified as " <= 0")

def outcome_constraint(X):
    
    return X.sum(dim =  -1) - 3

# Define the "fesibility weighted objective"

def weighted_obj(X):
    
    """
    * This function compares "outcome_constraint" with 0:
        * If the constraint is satisfied, the function returns -f(x);
        * If the constraint is not satisfied, the function returns 0;
    * ".type_as" is used to ensure that the output has the same type as the input;

    """
    
    return neg_hartmann6(X) * (outcome_constraint(X) <= 0).type_as(X)

"""
Block 2: Initialize the model
* We use a "MultiOutputGP" to model the objective (output 0) and the constraint (output 1);
* hp: homoskedastic noise with sigma = 0.5 on both objective and constraint ("FixedNoiseGP");
"""

train_yvar = torch.tensor(NOISE_SE ** 2, device = device, dtype = dtype)
# print(train_yvar)

# Generate the initial training data

def generate_initial_data(n):
    
    train_x = torch.rand(n, 6, device = device, dtype = dtype) # 6 - dimensional random vector
    
    # Evaluate both the function and the constraint
    
    exact_obj = neg_hartmann6(train_x).unsqueeze(-1) # add output dimension 
    exact_con = outcome_constraint(train_x).unsqueeze(-1) # add output dimension 
    
    # Generate noisy observations (a.k.a. add noise to the function and the constraint)
    
    train_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
    train_con = exact_con + NOISE_SE * torch.randn_like(exact_con)
    
    # Get the maximum, accounting for the feasibility of the constraint ('.item()' is used to get a scalar value)
    
    best_observed_value = weighted_obj(train_x).max().item() # We get the maximum of -f(x) = min
    
    # Get the index of the element of the tensor 'train_x' that has the maximum value
   
    idx = weighted_obj(train_x).argmax().item()
  
    return train_x, train_obj, train_con, best_observed_value, idx

# Call the function

train_x, train_obj, train_con, best_observed_value, idx = generate_initial_data(n = n_training_points)

print("\n")
print(f"The random training dataset is : {train_x} and its size is {train_x.size()}")
print("\n")
print(f"The evaluations of the hartman function with noise are : {train_obj}")
print("\n")
print(f"The evaluations of the constraint with noise are : {train_con}")
print("\n")
print(f"The best observed value is : {best_observed_value}")
print("\n")
print(f"The index of the best observed value is : {idx}")
print("\n")
print(f"The tensor of the best observed value is : {train_x[idx]}")
print("\n")

# Initialize the model

def initialize_model(train_x, train_obj, train_con, state_dict = None):
    
    """
    Syntax:
    * train_X: A `batch_shape x n x d` tensor of training features.
    * train_Y: A `batch_shape x n x m` tensor of training observations.
    * train_Yvar: A `batch_shape x n x m` tensor of observed measurement noise (defined above as "train_yvar").
    * ".to(train_x)": move the model to the same device (cpu) as the training data;
    * "load_state_dict": is usually used in deep learnig frameworks like PyTorch to 
        load weghts and parameters of a pre-trained model; 
    """
    
    # Training
    
    model_obj = FixedNoiseGP(train_x, train_obj, train_yvar.expand_as(train_obj)).to(train_x)
    model_con = FixedNoiseGP(train_x, train_con, train_yvar.expand_as(train_con)).to(train_x)
    
    # Create a multi-ouptut GP model
    
    model = ModelListGP(model_obj, model_con)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    
    # If necessary, import the weights and parameters of a pre-trained model
    
    if state_dict is not None:
        
        model.load_state_dict(state_dict)
        
    return mll, model

"""
Block 3: Definition of a construct to extract the objective and the constraint from the model
* "..." is used to indicate that Z can have any number of dimensions;
*  [..., 0] is used to select the objective from the model, that is the first output;
*  [..., 1] is used to select the constraint from the model, that is the second output;
"""

def obj_callable(Z):
    
    return Z[..., 0]

def constraint_callable(Z):
    
    return Z[..., 1]

# Definition of a feasibility-weighted objective for optimization

constrained_obj = ConstrainedMCObjective(
    objective = obj_callable,
    constraints = [constraint_callable],
)

"""
Block 4: Definition of a helper function that performs the optimization steps
* The helper function below takes an acquisition function as an argument, optimizes it, and returns the batch {x1,x2,…xq}
along with the observed function values. 
* For this example, we'll use a small batch of q = 3
* The function optimize_acqf optimizes the q points jointly (if in 'optimize_acqf' I specify 'sequential = True' ==> 
    optimization is done sequentially and not jointly). 
* A simple initialization heuristic is used to select the 10 restart initial locations from a set of 50 random points
"""

bounds = torch.tensor([[0.0] * 6, [1.0] * 6], device = device, dtype = dtype)

# Function to optimize the acquisition function

def optimize_acqf_and_get_observation(acq_function):
    
    """
    * The function "optimize_acqf" is used to perform the optimization steps;
    * This function returns a new candidate and a noisy observation
    """
    
    # Generate initial conditions
    
    candidates, _ = optimize_acqf(
        acq_function = acq_function,
        bounds = bounds,
        q = BATCH_SIZE,
        num_restarts = NUM_RESTARTS,
        raw_samples = RAW_SAMPLES,
        options = {"batch_limit": 5, "maxiter": 200},
    )
    
    # Observe new values
    
    """
    * ".detach()" is used to detach the output from the current graph and prevent future gradient computation;
    * ".unsqueeze(-1)" allows you to add a new dimension to the tensor or array. 
        The argument passed to .unsqueeze() specifies the axis along which the new dimension should be added. 
        In this case, -1 is passed as the argument, which indicates the last axis.
    """
    
    new_x = candidates.detach()
    exact_obj = neg_hartmann6(new_x).unsqueeze(-1)
    exact_con = outcome_constraint(new_x).unsqueeze(-1)
    
    # Add noise
    
    new_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
    new_con = exact_con + NOISE_SE * torch.randn_like(exact_con)
    
    return new_x, new_obj, new_con

# Function to simulate a random policy

def update_random_observations(best_random):
    
    """
    * Simulates a random policy by taking the current list of best values observed randomly,
        drawing a new random point, observing its value, and updating the list.
    * "max(best_random[-1], next_random_best)" is used to update the list of best values observed randomly:
        * best_random[-1] is the last value of the list;
        * next_random_best is the new value to be added to the list;
        The idea is to append the maximum value between the last value of the list and the new value;
    """
    
    rand_x = torch.rand(BATCH_SIZE, 6)
    next_random_best = weighted_obj(rand_x).max().item()
    best_random.append(max(best_random[-1], next_random_best))
    return best_random

"""
Block 5: Main Bayesian optimization loop
For the case of "q-batches", the loop is divided into three parts:
* Given the surrogate model, select a batch of points to evaluate {x1, x2, …, xq}
* Observe the function at each xi in the batch (for me: parallel evaluations in TPS)
* Update the surrogate model
The final plot will be a comparison between the results obtained by:
* A random policy
* The "Batch Expected Improvement" (qEI)
* The "Batch Noisy Expected Improvement" (qNEI)
"""

warnings.filterwarnings('ignore', category = BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category = RuntimeWarning)
warnings.filterwarnings('ignore')

# Initialize some empty vectors that will be plot at the end

best_observed_all_ei, best_observed_all_nei, best_random_all, tensor_of_results = [], [], [], []

# Start looping through all the simulations in TPS

for trial in range(1, N_TRIALS + 1):
    
    print(f"\n...................Trial {trial:>2} of {N_TRIALS} ........................", end = "")
    
    t0_trial = time.monotonic()
    
    # Define empty vectors for each trial
    
    best_observed_ei, best_observed_nei, best_random, x_vector = [], [], [], []
    
    # Call the helper function to generate the initial training data (randomly generated points with noise)
       
    train_x_ei, train_obj_ei, train_con_ei, best_observed_value_ei, idx_trial = generate_initial_data(n = n_training_points)
    
    # Create a MultiOutputGP model and obtain its hyperparameters
    
    mll_ei, model_ei = initialize_model(train_x_ei, train_obj_ei, train_con_ei)
    
    # Let's suppose that also for the "Noisy" case we keep the same results given by "generate_initial_data"
    
    train_x_nei, train_obj_nei, train_con_nei = train_x_ei, train_obj_ei, train_con_ei
    best_observed_value_nei = best_observed_value_ei
    mll_nei, model_nei = initialize_model(train_x_nei, train_obj_nei, train_con_nei)
    
    # Get the x vector corresponding to the best observed value
    
    x_train_best_nei = train_x_nei[idx_trial]
    
    # Augment the 'local' arrays initialized above
    
    best_observed_ei.append(best_observed_value_ei)
    best_observed_nei.append(best_observed_value_nei)
    best_random.append(best_observed_value_ei)
    
    print(f'\n\tThe initial dataset for the training of qNEI is:\n {train_x_ei}')
    print(f'\n\tThe best observed value for the qNEIis:\n {best_observed_value_ei}')
    print(f'\n\tThe element corresponding to the best value is:\n {x_train_best_nei} and the index is: {idx_trial}')
    
    # Focus on each bacth separately
    
    for iteration in range(1, N_BATCH + 1):
        
        print(f'\n\t\tBatch number :{iteration}')
        
        # Start the time counter to account for the evaluation of the batch
        
        t0_batch = time.monotonic()
        
        # fit the models (with the hyperparameters found above)
        
        fit_gpytorch_mll(mll_ei)
        fit_gpytorch_mll(mll_nei)
        
        # Define the sampler
        
        qmc_sampler = SobolQMCNormalSampler(sample_shape = torch.Size([MC_SAMPLES]))
        
        # For us, the best values are the best ones observed so far
        
        qEI = qExpectedImprovement(
            model = model_ei,
            best_f = (train_obj_ei * (train_con_ei <= 0).type_as(train_obj_ei)).max(),
            sampler = qmc_sampler,
            objective = constrained_obj,
        )
        
        qNEI = qNoisyExpectedImprovement(
            model = model_nei,
            X_baseline = train_x_nei,
            sampler = qmc_sampler,
            objective = constrained_obj,
        )
        
        # Optimize and find the new observations (new_x_ei are the new layouts to be tried)
        
        new_x_ei, new_obj_ei, new_con_ei = optimize_acqf_and_get_observation(qEI)
        new_x_nei, new_obj_nei, new_con_nei = optimize_acqf_and_get_observation(qNEI)
        
        print(f'\n\t\tThe new {BATCH_SIZE} candidates for the non - noisy EI objective are:\n {new_x_ei}')
        
        # Augment the training data by concatenating tensors
        
        train_x_ei = torch.cat([train_x_ei, new_x_ei])
        train_obj_ei = torch.cat([train_obj_ei, new_obj_ei])
        train_con_ei = torch.cat([train_con_ei, new_con_ei])
        
        train_x_nei = torch.cat([train_x_nei, new_x_nei])
        train_obj_nei = torch.cat([train_obj_nei, new_obj_nei])
        train_con_nei = torch.cat([train_con_nei, new_con_nei])
        
        print(f'\n\t\tThe augmented dataset, at the iteration {trial} for the batch number {iteration}, for the qNEI is: {train_x_nei}')
        
        # Update the progress by evaluating the function on the augmented training data
        
        best_random = update_random_observations(best_random)
        best_value_ei = weighted_obj(train_x_ei).max().item()
        best_value_nei = weighted_obj(train_x_nei).max().item()
        best_observed_ei.append(best_value_ei)
        best_observed_nei.append(best_value_nei)
        
        # Get the best x_vector among the new 'q' ones
        
        idx_batch = weighted_obj(train_x_nei).argmax().item()
        x_vector.append(train_x_nei[idx_batch])
        
        print(f'\n\t\tThe best observed value, at simualtion {trial} for the batch {iteration}, for the qNEI is: {best_observed_nei}')
        print(f'\n\tThe index is {idx_batch} and the best value is {train_x_nei[idx_batch]}')
        print(f'\n\tThe vector x_vector is : {x_vector}')
        
        # Re-initialize the models to make them ready for the next iteration (use the current state dict)
        
        mll_ei, model_ei = initialize_model(
            train_x_ei,
            train_obj_ei,
            train_con_ei,
            model_ei.state_dict(),
        )
        
        mll_nei, model_nei = initialize_model(
            train_x_nei,
            train_obj_nei,
            train_con_nei,
            model_nei.state_dict(),
        )
        
        # Stop the time counter
        
        t1_batch = time.monotonic()
        
        print(f'\n\t\tThe time taken for the evaluation of batch {iteration} is: {t1_batch - t0_batch}')
        
        if verbose:
            print(
                f"\n Batch {iteration:>2}: best_value (random, qEI, qNEI) = "
                f"({max(best_random):>4.2f}, {best_value_ei:>4.2f}, {best_value_nei:>4.2f}), "
                f"time = {t1_batch - t0_batch:>4.2f}.",
                end = "",
            )
            
        else:
            
            print(".", end = "")
    
    # stop the counter
    
    t1_trial = time.monotonic()
    print(f'\n\t\tThe time taken for the trial number {trial} is: {t1_trial - t0_trial}')
    
    # Augment the arrays for the final plot
           
    best_observed_all_ei.append(best_observed_ei)
    best_observed_all_nei.append(best_observed_nei)
    best_random_all.append(best_random)
    tensor_of_results.append(x_vector)
    
    first_dim = len(tensor_of_results)
    second_dim = len(tensor_of_results[0])
    third_dim = len(tensor_of_results[0][0])
    
    print(f'\nThe final best observed value for the non-noisy EI is: {best_observed_all_nei}')
    print(f'\nThe resulting x_vectors are: {tensor_of_results}')
    print(f'\nThe first sequece of the tensor is: {tensor_of_results[0]}')
    print(f'\nThe first element of the first sequence is: {tensor_of_results[0][0].numpy()}')
    #print(f'\nThe first size of the tensor is: {first_dim}')
    #print(f'\nThe second size of the tensor is: {second_dim}')
    #print(f'\nThe third size of the tensor is: {third_dim}')
    
"""
Block 6: Plot the results
"""

# Evaluate the final x vector corresponding to the best observed value

print(f'\nThe result is: {sum(tensor_of_results[:][-1]) / second_dim}')

# Definition of a function that computes the confidence interval of an array

def ci(y):
    
    return 1.96 * y.std(axis = 0) / np.sqrt(N_TRIALS)

# Known target value for the Hartmann6 function (-f(x) = 3.32237)

GLOBAL_MAXIMUM = neg_hartmann6.optimal_value

# Define the vectors for the x axis and the y axis

iters = np.arange(N_BATCH + 1) * BATCH_SIZE
y_ei = np.asarray(best_observed_all_ei)
y_nei = np.asarray(best_observed_all_nei)
y_rnd = np.asarray(best_random_all)

print(f'\nThe best values for EI, transformed in numpy array, are:\n {y_ei}')
print(f'\nThe vector in the plot is:\n{y_ei.mean(axis = 0)}')

fig, ax = plt.subplots(1, 1, figsize = (8, 6))

# Plot the results (They are the mean values of the best observed values)

ax.errorbar(iters, y_rnd.mean(axis = 0), yerr = ci(y_rnd), label = "random", linewidth = 1.5)
ax.errorbar(iters, y_ei.mean(axis = 0), yerr = ci(y_ei), label = "qEI", linewidth = 1.5)
ax.errorbar(iters, y_nei.mean(axis = 0), yerr = ci(y_nei), label = "qNEI", linewidth = 1.5)

# Plot a horizontal line

plt.plot(
    [0, N_BATCH * BATCH_SIZE],
    [GLOBAL_MAXIMUM] * 2,
    "k",
    label = "$f_{min}$",
    linewidth = 2,
)
ax.set_ylim(bottom = 0.5)
ax.set(
    xlabel = "number of observations (beyond initial points)",
    ylabel = "Best objective values",
)
ax.legend(loc = "lower right")
plt.title("Comparison of qEI and qNEI for function and constraint", fontsize = 20)
plt.grid()
plt.show()
    




