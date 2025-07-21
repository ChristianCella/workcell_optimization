
"""
* This code refers to the paper: 'Scalable global optimization via local Bayesian optimization'
* TuRBO = Trust Region Bayesian Optimization
* In this implementation we use just one trust region (TuRBO - 1)
* We focus on the 20-D Ackley function: it has a minimum global at 0, obtained when all input variables are equal to 0
* The idea is to see how a single TR works: if the length of the TR is kept big enough,
    the algorithm tends to become the standard Bayesian optimization
"""

import warnings
warnings.filterwarnings("ignore")

import os
import torch

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")

from botorch.test_functions import Ackley, Levy

fun = Levy(dim = 30, negate = True).to(dtype = dtype, device = device)
fun.bounds[0, :].fill_(-5)
fun.bounds[1, :].fill_(10)
dim = fun.dim
lb, ub = fun.bounds # 20-dimensional vectors composed of all -5 and all 10, respectively

batch_size = 5
n_init = 40
max_cholesky_size = float('inf')

# Define a helper function used to unnromanize and evaluate a point

from botorch.utils.transforms import unnormalize

def eval_objective(x):
    return fun(unnormalize(x, fun.bounds))

"""
TuRBO needs to maintain a state that includes:
* length of the trust region (the actual side length is obtained by rescaling the initial one, byb keeping the total volume constant)
* success and failure counters
* success and failure tolerance
The state is updated after each iteration and is stired inside a dataclass:
* The doamin must be normalized to [0, 1]^dim
* The batch size must be kept constant
If the number of consecutive counters is reached (for example, 10), the trust region is updated;
After a certain number of failures (for example, 1), the trust region is halved;
"""

import math
from dataclasses import dataclass

@dataclass
class TurboState:
    dim: int
    batch_size: int 
    length: float = 0.8 # L_init
    length_min: float = 0.2 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    #failure_tolerance: int = float("nan") # This will be post-initialized
    failure_tolerance: int = 1
    success_counter: int = 0
    success_tolerance: int = 10
    best_value: float = -float("inf")
    restart_triggered: bool = False
    
    # This is quite useless
    
    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )

state = TurboState(dim = dim, batch_size = batch_size)
print(f'The state is: {state}')
       
# Define a function to update the state ('fabs' ==> returns the absolute value of a float called x)

def update_state(state, Y_next):
    
    # Update the counters
    
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter += 0
        state.failure_counter += 1
        
    # Update the trust regions: after we change the size of the trust region, we reset the counters
    
    if state.success_counter == state.success_tolerance: # Expand the trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance: # Shrink the trust region
        state.length /= 2.0
        state.failure_counter = 0
    
    # Update the state: you take the max between the best value and the max of the new function evaluations
    
    state.best_value = max(state.best_value, max(Y_next).item())
    
    # Need of a restart in case that the state length is less than the minimum: we discard the current TR
    # And initialize a new one
    
    if state.length < state.length_min:
        state.restart_triggered = True
        
    # Return
    
    return state

"""
Generation of the initial Sobol points for the training
"""

from torch.quasirandom import SobolEngine

def get_initial_points(dim, n_pts, seed = 0):
    
    sobol = SobolEngine(dimension = dim, scramble = True, seed = seed)
    X_init = sobol.draw(n = n_pts).to(dtype = dtype, device = device) # the result is :math:`(n, dimension)`.
    return X_init

"""
Generation of a new batch of points (new candidate)
* Given: state and probabilistic GP model built on X and Y
* Output: new candidate
* This method works on [0, 1]^dim: Unnromalize is used to go abck to the original domain before the evaluation
* Both Ts and qEI are supported
* The TR are chosen as the hyperrectangles centered at the best solution found so far 
"""

def generate_batch(
    state, 
    model, # GP model
    X, # Evaluated points on the domain [0, 1] ^ dim
    Y, # Function values
    batch_size,
    n_candidates = None, # Number of candidates for Thompson sampling
    num_restarts = 10,
    raw_samples = 512,
    acqf = "ts", # This can either be "ts" or "ei"
):
    assert acqf in ("ts", "ei")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))
        
    # Scale the trust region to be proportional to the lengthscales
    
    x_center = X[Y.argmax(), :].clone() # Clone the point in X that has the maximum value in Y
    
    # Evaluation of the weights (page 3 of the paper)
    
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights))) # '.pow' ==> element-wise power
    
    """
    Limit ('clamp') the trust region to be within the bounds (0.0, 1.0)
    * The lower bound is the tensor x_center - weights * state.length / 2.0
    * The upper bound is the tensor x_center + weights * state.length / 2.0
    """
    
    from botorch.generation import MaxPosteriorSampling
    from botorch.acquisition import qExpectedImprovement
    from botorch.optim import optimize_acqf
    
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)
    
    # Definition of the acquisition functions
    
    if acqf == "ts":
        
        dim = X.shape[-1]
        sobol = SobolEngine(dimension = dim, scramble = True)
        pert = sobol.draw(n = n_candidates).to(dtype = dtype, device = device)
        pert = tr_lb + (tr_ub - tr_lb) * pert
        
        # Creation of a perturbation mask
        
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, dtype = dtype, device = device) <= prob_perturb # Tensor of size n_candidates x dim
        ind = torch.where(mask.sum(dim = -1) == 0)[0] # Save in 'ind' the first element of the tuple generated by 'where'
        mask[ind, torch.randint(0, dim - 1, size = (len(ind),), device = device)] = 1
        
        # Creation of candidate points from the perturbations and the mask
        
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]
        
        # Sample on the candidate points (here you could also introduce constraints)
        
        thompson_sampling = MaxPosteriorSampling(model = model, replacement = False)
        with torch.no_grad(): # No need of gradients with Thompson sampling
            
            X_next = thompson_sampling(X_cand, num_samples = batch_size)
            
    elif acqf == "ei":
        
        ei = qExpectedImprovement(model = model, best_f = Y.max())
        X_next, acq_value = optimize_acqf(
            ei,
            bounds = torch.stack([tr_lb, tr_ub]),
            q = batch_size,
            num_restarts = num_restarts,
            raw_samples = raw_samples,
        )   
        
    return X_next

"""
Optimization loop:
* One instance of TuRBO-1 with Thompson sampling
* TuRBO-1 is a local optimizer that can be used in a multi-start fashion. 
* We use a SingleTaskGP with noise constraint to keep the noise from getting too large
"""

import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval
from gpytorch.kernels import ScaleKernel, MaternKernel
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

X_turbo = get_initial_points(dim = dim, n_pts = n_init) # tensor of 40 elements, each containing 20 elements
Y_turbo = torch.tensor(
    [eval_objective(x) for x in X_turbo], dtype = dtype, device = device
    ).unsqueeze(-1) # tensor of 40 elements

state = TurboState(dim = dim, batch_size = batch_size)

NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 3001 if not SMOKE_TEST else 4 
#N_CANDIDATES = min(5000, max(2000, 200 * dim)) if not SMOKE_TEST else 4
N_CANDIDATES = 3000

torch.manual_seed(0)

# Not a for loop anymore: we run until the restart is triggered
     
while not state.restart_triggered: # Run until TuRBO converges
    
    # Fit a Gaussian Process Model
    
    train_Y = (Y_turbo - Y_turbo.mean()) / Y_turbo.std()
    likelihood = GaussianLikelihood(noise_constraint = Interval(1e-8, 1e-3))
    covar_module = ScaleKernel(
        MaternKernel(
            nu = 2.5,
            ard_num_dims = dim,
            lengthscale_constraint = Interval(0.005, 4.0),
        )
    )
    
    model = SingleTaskGP(X_turbo, train_Y, covar_module = covar_module, likelihood = likelihood)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    
    # Fitting and optimization of the acquisition function inside Cholesky context
    
    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
        
        # Fit the model
        
        fit_gpytorch_mll(mll)
        
        # Create a batch
        
        X_next = generate_batch(
            state = state,
            model = model,
            X = X_turbo,
            Y = train_Y,
            batch_size = batch_size,
            n_candidates = N_CANDIDATES,
            num_restarts = NUM_RESTARTS,
            raw_samples = RAW_SAMPLES,
            acqf = "ei", # Thompspon sampling
        )
        
    Y_next = torch.tensor([eval_objective(x) for x in X_next], dtype = dtype, device = device).unsqueeze(-1)
    
    # Update the state
    
    state = update_state(state, Y_next)
    
    # Append data
    
    X_turbo = torch.cat((X_turbo, X_next), dim = 0)
    Y_turbo = torch.cat((Y_turbo, Y_next), dim = 0)
    
    # Print the current status
    
    print(f"{len(X_turbo)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}")