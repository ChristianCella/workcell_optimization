"""
This code is used to see the difference between the qUCB from BoTorch and the UCB function I implemented by hand
"""

# Import the library for the constraint
from botorch.optim.parameter_constraints import make_scipy_linear_constraints

# Filters for warnings
import warnings
import os, sys
from botorch.exceptions import BadInitialCandidatesWarning

warnings.filterwarnings('ignore', category = BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category = RuntimeWarning)
warnings.filterwarnings('ignore')

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils'))
sys.path.append(base_dir)
import fonts

# Block 1
import torch

# Block 2.1
from botorch.models import SingleTaskGP 
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll  # <<<< for BoTorch 0.14.0

# Block 2.2
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.sampling.normal import SobolQMCNormalSampler

# Block 2.3
from botorch.optim import optimize_acqf
import time

# Block 3
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

matplotlib.rcParams['mathtext.fontset'] = 'cm' # 'cm' or 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rc('xtick', labelsize = 15)
matplotlib.rc('ytick', labelsize = 15)

# Define the function
def f(x):
    fun = x[:, 0] ** 2 + x[:, 1] ** 2
    return -fun

# Main
if __name__ == "__main__":
    # Constants
    n_var = 2   
    NTrials = 1
    q_batch = 32
    t_batch = 20
    n_training_points = 20
    x_min = -10.0
    x_max = 10.0
    kappa = 0.1
    n_restarts = 20
    MC_samples = 256
    RAW_samples = 512
    point_plot = 100
    fig_size = 10 
    global_minimum = 0

    """
    Block 1: Select the device and generate the training dataset
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    print(f'\nThe selected device is: {device}')
    print(f'{fonts.green}PyTorch version: {torch.__version__}{fonts.reset}')
    print(f'{fonts.yellow}CUDA availability: {torch.cuda.is_available()}{fonts.reset}')
    if torch.cuda.is_available():
        print(f'{fonts.purple}The device for the training is: {torch.cuda.get_device_name(0)}{fonts.reset}')
    
    """
    Block 2: Bayesian optimization
    """
    best_all_evaluations, tensor_of_results = [], []
    t0_tot = time.monotonic()
    
    for trial in range(1, NTrials + 1):
        print(f'\n\t------------------Trial number: {trial}-----------------')
        t0_trial = time.monotonic()
        x_vector, best_evaluations = [], []
        train_x = torch.rand(n_training_points, n_var, device=device, dtype=dtype) * (x_max - x_min) + x_min
        train_y = f(train_x).unsqueeze(-1)
        idx = train_y.argmax().item()
        best_evaluation_training = train_y.max().item()
        best_vector_training = train_x[idx]
        best_evaluations.append(best_evaluation_training)
        print(f'\n\tThe training dataset is:\n{train_x},\nwhile the output is:\n{train_y}')
        print(f'\n\tThe best vector in the training dataset is:\n{best_vector_training},\nwhile the output is:\n{best_evaluation_training}')

        for iteration in range(1, t_batch + 1):
            print(f'\n\t\t*************The current batch for the trial {trial} is {iteration}****************')
            to_batch = time.monotonic()
            
            # Block 2.1: Initialize the model
            model = SingleTaskGP(train_X=train_x, train_Y=train_y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Block 2.2: Define the acquisition function and the sampler
            qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_samples]))
            qUCB = qUpperConfidenceBound(
                model=model,
                beta=kappa, 
                sampler=qmc_sampler,
            )
            
            # Block 2.3: Optimization of the acquisition function, with a constraint
            bounds = torch.tensor(
                [[x_min] * n_var, [x_max] * n_var], device=device, dtype=dtype
            )
            inequality_constraints = [
                (torch.tensor([0, 1], device=device), torch.tensor([-1.0, -1.0], device=device, dtype=dtype), -1.0),
                (torch.tensor([0, 1], device=device), torch.tensor([1.0, 1.0], device=device, dtype=dtype), -1)
            ]
            candidate, _ = optimize_acqf(
                acq_function=qUCB,
                bounds=bounds,
                q=q_batch,
                num_restarts=n_restarts,
                raw_samples=RAW_samples,
                options={"batch_limit": 5, "maxiter": 200},
                inequality_constraints=inequality_constraints,
            )
            print(f'\n\t\tThe new {q_batch} candidates for the batch number {iteration} are:\n{candidate}')

            # Augment the 'x' dataset with the new candidates and evaluate the function again (save the maximum)
            train_x = torch.cat([train_x, candidate.detach()])
            train_y = torch.cat([train_y, f(candidate).unsqueeze(-1)])
            evaluation = f(train_x).max().item()
            best_evaluations.append(evaluation)

            # Get the best 'x' vector among the augmented variables in the dataset
            idx_batch = f(train_x).argmax().item()
            x_vector.append(train_x[idx_batch].cpu().numpy())

            t1_batch = time.monotonic()
            print(f'\n\t\tThe time taken by the batch number {iteration} is: {t1_batch - to_batch} seconds')

        t1_trial = time.monotonic() 
        print(f'\n\tThe time taken by the trial number {trial} is: {t1_trial - t0_trial} seconds')

        best_all_evaluations.append(best_evaluations)
        tensor_of_results.append(x_vector)
    
    t1_end = time.monotonic()
    print(f'\nThe total time taken is: {t1_end - t0_tot} seconds')
    
    # Display the final results and calculate the main dimensions of the tensor
    first_dim = len(tensor_of_results)
    second_dim = len(tensor_of_results[0])
    third_dim = len(tensor_of_results[0][0])
    print(f'The resulting x vector is: {sum(tensor_of_results[:][-1]) / first_dim}')
    print(f'The complete tensor is: {tensor_of_results}')
    print(f'The first element is: {np.asarray(tensor_of_results)}')
    print(f'The first elements are: {np.asarray(tensor_of_results)[0][:, 0]}')
    
    """
    Block 3: plot the results
    """
    # Plot the function evaluation
    iters = np.arange(t_batch + 1) * q_batch
    y_plot = np.asarray(best_all_evaluations)
    print(f'\nThe vector to be plotted is {y_plot}')
    print(f'\nThe vector in the plot is: {y_plot.mean(axis = 0)}')
    fig, ax = plt.subplots(1, 1, figsize = (fig_size, fig_size))
    ax.plot(iters, y_plot.mean(axis = 0), label = 'qUCB')
    ax.scatter(iters, y_plot.mean(axis = 0), color = 'r', s = 25)
    plt.plot(
        [0, t_batch * q_batch],
        [global_minimum] * 2,
        "k",
        label = "$f_{min}$",
        linewidth = 2,
    )   
    ax.legend(loc = "lower right")
    plt.xlabel("($t_{batch} + 1)\dot{q_{batch}}$", fontsize = 15)
    plt.ylabel("Best evaluation", fontsize = 15)
    plt.title(f'Constrained Batch optimization, q = {q_batch} and Nsim = {t_batch + 1}', fontsize = 20)
    plt.grid()
    plt.show()

    # Plot the constraint
    fig1, ax1 = plt.subplots(1, 1, figsize = (fig_size, fig_size))
    ax1.scatter(
        np.arange(1, t_batch + 1),
        np.asarray(tensor_of_results)[0][:, 0] + np.asarray(tensor_of_results)[0][:, 1],
        color = 'r', s = 25
    )
    plt.plot(
        [0, t_batch],
        [0] * 2,
        "k--",
        label = "$Constraint$",
        linewidth = 2,
    ) 
    plt.grid()
    plt.xlabel('Batch number', fontsize = 15)
    plt.ylabel('Constraint: $x_1+x_2\geq -1$, $x_1+x_2\leq1$', fontsize = 15)
    plt.title('Constrained optimization', fontsize = 20)
    plt.show()
