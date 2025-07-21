""" 
This code deals with the implementation of the SCBO algorithm using BoTorch.
* SCBO = Scalable Constrained Bayesian Optimization
* Again, the target function is the Ackley function, this time with 10 dimensions.
* This time we have 2 constraints: we need c1(x) <= 0 and c2(x) <= 0.
* SCBO is basically a constrained version of TuRBO
"""

from dataclasses import dataclass

import math
from torch import Tensor
import warnings
import os
import torch
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from botorch.generation.sampling import MaxPosteriorSampling, ConstrainedMaxPosteriorSampling
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch
from botorch.fit import fit_gpytorch_mll
from botorch.models.model_list_gp_regression import ModelListGP

from torch.quasirandom import SobolEngine

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
tkwargs = {"device": device, "dtype": dtype}

SMOKE_TEST = os.environ.get("SMOKE_TEST")

"""
Definition of the objective function
"""

fun = Ackley(dim = 10, negate = True).to(**tkwargs)
fun.bounds[0, :].fill_(-5)
fun.bounds[1, :].fill_(10)
dim = fun.dim 
lb, ub = fun.bounds

batch_size = 4
n_init = 10
max_cholesky_size = float("inf")

"""
Definition of a function that allows to pass from the function expressed on [0, 1] ^ d in the main loop
    to the original function defined on the original domain
"""

def eval_objective(x):
    return fun(unnormalize(x, fun.bounds))

"""
Define two constraint functions
* SCBO expects constraints to be of the form c_i(x) <= 0
* The constraints are very simple, but nothing would change with more complex constraints 
"""

def c1(x):
    
    # Enforce that sum(x) <= 0
    
    return x.sum()

def c2(x):
    
    # Enforce also that ||x||_2 - 5 <= 0
    
    return torch.norm(x, p = 2) - 5

# Hp: we assume that also c1 and c2 have the same bounds as the objective function

def eval_c1(x):
    return c1(unnormalize(x, fun.bounds))

def eval_c2(x):
    return c2(unnormalize(x, fun.bounds))

"""
Definition of the TuRBO class
* We need to hold the trust region state and a method to update the length of the hyper-cube TR
* The side length of the TR is updated according to sequential successes or failures 
"""

@dataclass
class ScboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan") # NOTE: post - initialized
    success_counter: int = 0
    success_tolerance: int = 10 # NOTE: the original paper suggests 3
    best_value: float = -float("inf")
    best_constraint_values : Tensor = torch.ones(2, **tkwargs) * float("inf")
    restart_triggered: bool = False
    
    def __post_init__(self):
        self.failure_tolerance = math.ceil(max([4.0 / self.batch_size, float(self.dim) / self.batch_size]))

"""
Definition of the method to update the length of the hyper-cube TR 
"""
      
def update_tr_length(state: ScboState):

    # Expand the trust region

    if state.success_counter == state.success_tolerance:
        
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    
    # Shrink the trust region
    
    elif state.failure_counter == state.failure_tolerance:
        
        state.length /= 2.0
        state.failure_counter = 0
        
    # When the trust region become stoo small: restart
    
    if state.length < state.length_min:
        
        state.restart_triggered = True
        
    return state

"""
Function that returns the index for the best point 
* Y represents the observations of the objective function
* C represents the observations of the constraints
"""

def get_best_index_for_batch(Y: Tensor, C: Tensor):
    
    # Check if all the constraints are smaller than zero
    
    is_feas = (C <= 0).all(dim = 1)
    
    # If at least one element of 'is_feas' is True (F_hat is not an empty set), choose the best feasible candidate
    
    if is_feas.any():
        
        # Copy the objective function observations
        
        score = Y.clone()
        
        # Exclude the points that violate the constraints (is_feas = False)
        
        score[~is_feas] = -float("inf") 
        
        # Retrun the index of the elements with the maximum value
        
        return score.argmax()
    
    # Otherwise, there was not even a point that satisfied the constraints;
    # choose the point of minimum total constraint violation (what I return is the index of the point)
    
    return C.clamp(min = 0).sum(dim = -1).argmin()

""" 
Function for the update of the TuRBO state at each step of the optimization
* Success and failure counters are updated according to the objective values
    (Y_next) and constraint values (C_next) of the batch of candidate points
    evaluated on the optimization step.
* As in the original TuRBO paper, a success is counted whenever any one of the
    new candidate points improves upon the incumbent best point. 
*The key difference for SCBO is that we only compare points by their objective values when both points
    are valid (meet all constraints). 
* If exactly one of the two points being compared violates a constraint, the other valid point
    is automatically considered to be better.
* If both points violate some constraints, we compare them by their constraint values.
* The better point in this case is the one with minimum total constraint violation
    (the minimum sum of constraint values)
"""

def update_state(state, Y_next, C_next):
    
    # Pick the best point from the batch (access Y at the position whose index is the one of the best candidate)
    
    best_ind = get_best_index_for_batch(Y = Y_next, C = C_next)
    
    # 'Local' variables
    
    y_next, c_next = Y_next[best_ind], C_next[best_ind]
    
    # If all elements in c_next are less than zero: at least one of the new candidates is feasible
    
    if (c_next <= 0).all:
        
        improvement_threshold = state.best_value + 1e-3 * math.fabs(state.best_value)
        
        # In case that the new observaion is better than the best so far, or
        # in case that any of the elements in the vector 'best_constraint_values' is greater than zero
        
        if y_next > improvement_threshold or (state.best_constraint_values > 0).any():
            
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = y_next.item()
            state.best_constraint_values = c_next
            
        else:
            
            # Despite I found a candidate, it does not show an improvement: I failed
               
            state.success_counter = 0
            state.failure_counter += 1
    
    # There's no new candidate that is feasible: we compare the constraint values
           
    else:
        
        total_violation_next = c_next.clamp(min = 0).sum(dim = -1)
        total_violation_center = state.best_constraint_values.clamp(min = 0).sum(dim = -1)
        
        # If the new total violation is lower than it was previously: success
        
        if total_violation_next < total_violation_center:
            
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = y_next.item()
            state.best_constraint_values = c_next
            
        # Failure condition
            
        else:
                
                state.success_counter = 0
                state.failure_counter += 1
                
    # Update the length of the trust region according to success or failure conditions
    
    state = update_tr_length(state)
    return state

"""
Define an example state 
"""

state = ScboState(dim = dim, batch_size = batch_size)
print(f'The state is:{state}')

""" 
Generate initial points with the Latin Hypercube Sampling
"""

def get_initial_points(dim, n_pts, seed = 0):
    
    sobol = SobolEngine(dimension = dim, scramble = True, seed = seed)
    X_init = sobol.draw(n = n_pts).to(dtype = dtype, device = device)
    return X_init

""" 
Generate a batch of candidates for SCBO
* Just as in TuRBO, we define a method to generate a batch of candidates using Thompson Sampling
* The key difference here from TuRBO is that, instead of using MaxPosteriorSampling to simply grab 
    the candidates within the trust region with the maximum posterior values, we use ConstrainedMaxPosteriorSampling 
    to instead grab the candidates within the trust region with the maximum posterior values subject to the 
    constraint that the posteriors for the constraint models for c1(x) and c2(x) must be less than or equal to 0 
    for both candidates.
* We use additional GPs ('constraint models') to model each black-box constraint (c1 and c2), and throw out all 
    candidates for which the sampled value for these constraint models is greater than 0
"""

def generate_batch(state, model, X, Y, C, batch_size, n_candidates, constraint_model, sobol: SobolEngine):
    
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    
    # Create the TR bounds
    
    best_ind = get_best_index_for_batch(Y = Y, C = C)
    x_center = X[best_ind, :].clone()
    tr_lb = torch.clamp(x_center - state.length / 2, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + state.length / 2, 0.0, 1.0)

    """ 
    Acquisition function = Thompson sampling
    * Overall, the combination of Thompson sampling with the perturbation mask provides 
    a flexible framework for balancing exploration and exploitation in multi-armed bandit problems, 
    allowing for adaptive decision-making in uncertain environments.
    """
    
    dim = X.shape[-1]
    pert = sobol.draw(n_candidates).to(dtype = dtype, device = device)
    pert = tr_lb + (tr_ub - tr_lb) * pert

    # Create a perturbation mask
    
    prob_perturb = min(20 / dim, 1)
    
    # random perturbtion: exporation - exploitation tarde-off. 'torch.rand' returns a tensor of random numbers between 0 and 1
    
    mask = torch.rand(n_candidates, dim, **tkwargs) <= prob_perturb 
    
    # Access the row indices ([0]) of the rows that sum to 0
    
    ind = torch.where(mask.sum(dim = 1) == 0)[0]
    
    # sets the elements of the 'mask' tensor at the indices specified by 'ind' and randomly generated column indices to 1.
    
    mask[ind, torch.randint(0, dim, size = (len(ind),), device = device)] = 1
    
    # Create candidate points from the perturbations and the mask
    
    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]
    
    # Sample on the candidate points using 'Constrained Max Posterior Sampling'
    
    constrained_thompson_sampling = ConstrainedMaxPosteriorSampling(
        model = model, constraint_model = constraint_model, replacement = False
    )
    
    with torch.no_grad():
        
        # 'X_next' has dimension (batch_size = q, dim)
        
        X_next = constrained_thompson_sampling(X = X_cand, num_samples = batch_size)
        
    return X_next

"""
Main optimization loop 
"""

# Generation of initial data (Latin Hypercube Sampling)

train_X = get_initial_points(dim = dim, n_pts = n_init)
train_Y = torch.tensor([eval_objective(x) for x in train_X], **tkwargs).unsqueeze(-1)
C1 = torch.tensor([eval_c1(x) for x in train_X], **tkwargs).unsqueeze(-1)
C2 = torch.tensor([eval_c2(x) for x in train_X], **tkwargs).unsqueeze(-1)

# Initialize the TuRBO state (we use 2000 candidate points for the Thompson sampling)

state = ScboState(dim = dim, batch_size = batch_size)
N_CANDIDATES = 2000
# N_CANDIDATES = min(5000, int(1e4 * dim))
sobol = SobolEngine(dimension = dim, scramble = True, seed = 1)

# Define a function to fit the models (used both for the objective functions and the constraints)

def get_fitted_model(X, Y):
    
    likelihood = GaussianLikelihood(noise_constraint = Interval(1e-8, 1e-3))
    covar_module = ScaleKernel(
        MaternKernel(nu = 2.5, ard_num_dims = dim, lengthscale_constraint = Interval(0.005, 4.0))
    )
    
    model = SingleTaskGP(
        X,
        Y, 
        likelihood = likelihood, 
        covar_module = covar_module,
        outcome_transform = Standardize(m = 1)
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    
    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
        
        fit_gpytorch_mll(mll)
        
    return model

# Optimization loop: run until TuRBO converges

while not state.restart_triggered:
    
    # Fit GP models for objective and constraints
    
    model = get_fitted_model(train_X, train_Y)
    c1_model = get_fitted_model(train_X, C1)
    c2_model = get_fitted_model(train_X, C2)

    # Generate a batch of 'q = batch_size' candidates
    
    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
        X_next = generate_batch(
            state = state,
            model = model,
            X = train_X,
            Y = train_Y,
            C = torch.cat((C1, C2), dim=-1),
            batch_size = batch_size, # q ==> index of the parallelization
            n_candidates = N_CANDIDATES, # size for the Thompson sampling
            constraint_model = ModelListGP(c1_model, c2_model),
            sobol = sobol,
        )

    # Print the X_next found
    
    print(f'X_next found: {X_next * (ub - lb) + lb}')    
    
    # Evaluate both the objective and constraints for the selected candidates
    
    Y_next = torch.tensor([eval_objective(x) for x in X_next], dtype = dtype, device = device).unsqueeze(-1)
    C1_next = torch.tensor([eval_c1(x) for x in X_next], dtype = dtype, device = device).unsqueeze(-1)
    C2_next = torch.tensor([eval_c2(x) for x in X_next], dtype = dtype, device = device).unsqueeze(-1)
    C_next = torch.cat([C1_next, C2_next], dim = -1)

    # Update TuRBO state
    
    state = update_state(state = state, Y_next = Y_next, C_next = C_next)

    """
    Append the data.
    * NOTE: we decide to append all data, even points that violate the constraints. This is to be sure that
        our constraint model can learn more about the constraint functions and gain confidence in where 
        violation occur. 
    """
    
    train_X = torch.cat((train_X, X_next), dim = 0)
    train_Y = torch.cat((train_Y, Y_next), dim = 0)
    C1 = torch.cat((C1, C1_next), dim = 0)
    C2 = torch.cat((C2, C2_next), dim = 0)

    """
    Display the current status of the optimization.
    * NOTE: 'state_best_value' is the best value of the objective function found so far that meets the constraints,
        or the objective value of the point with the minimum constraint violation (in case no point has been found yet). 
    """
    
    if (state.best_constraint_values <= 0).all():
        
        print(f"{len(train_X)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}")
        
    else:
        
        violation = state.best_constraint_values.clamp(min=0).sum()
        print(
            f"{len(train_X)}) No feasible point yet! Smallest total violation: "
            f"{violation:.2e}, TR length: {state.length:.2e}"
        )
        
"""
Plot the results 
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

fig, ax = plt.subplots(figsize = (8, 6))

score = train_Y.clone()

# Set infeasible to -inf

score[~(torch.cat((C1, C2), dim = -1) <= 0).all(dim = -1)] = float("-inf")
fx = np.maximum.accumulate(score.cpu())

plt.plot(fx, marker = "", lw = 3)
plt.plot([0, len(train_Y)], [fun.optimal_value, fun.optimal_value], "k--", lw = 3)
plt.ylabel("Function value", fontsize = 18)
plt.xlabel("Number of evaluations", fontsize = 18)
plt.title("10D Ackley with 2 outcome constraints", fontsize = 20)
plt.xlim([0, len(train_Y)])
plt.ylim([-15, 1])

plt.grid(True)
plt.show()