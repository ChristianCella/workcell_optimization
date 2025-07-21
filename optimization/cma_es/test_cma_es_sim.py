#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from cma import CMA

# ───── Your MuJoCo simulation wrapper ─────
def run_simulation(params: np.ndarray) -> float:
    """
    Replace the body of this function with your actual MuJoCo call.
    'params' is a 1D numpy array of length N.
    Return a scalar fitness (lower is better).
    """
    # e.g.
    # from your_mujoco_module import make_env
    # env = make_env()
    # result = env.simulate(params)
    # return result.cost
    #
    # Here we just use Sphere as a stand‐in:
    return float(np.sum(params**2))

# ───── Wrap it in a TF‐eager fitness function ─────
def fitness_fn(x: tf.Tensor) -> tf.Tensor:
    """
    x: tf.Tensor of shape (popsize, dim)
    must return a tf.Tensor of shape (popsize,)
    """
    # pull it into NumPy so we can call your simulator
    x_np = x.numpy()
    fits = [run_simulation(ind) for ind in x_np]
    return tf.constant(fits, dtype=x.dtype)

# ───── Callback to print per‐gen stats ─────
def print_callback(cma, logger):
    # cma.generation has already been incremented
    print(f'Gen {cma.generation:3d}  best f = {cma.best_fitness():.6f}')

if __name__ == '__main__':
    # problem setup
    dim       = 5                   # problem dimension
    x0        = np.zeros(dim)       # initial guess
    sigma0    = 1.0                 # initial step‐size
    popsize   = 20                  # number of candidates per generation
    max_gens  = 100                 # stopping criterion

    # build the optimizer
    cma = CMA(
        initial_solution = x0,
        initial_step_size= sigma0,
        fitness_function = fitness_fn,
        population_size  = popsize,
        callback_function= print_callback,
        dtype            = tf.float64,  # use float64 if you like
    )

    # ───── run the search ─────
    best_solution, best_fitness = cma.search(max_generations=max_gens)

    # ───── results ─────
    print('\nOptimization terminated:')
    print('  best solution:', best_solution)
    print(f'  best fitness : {best_fitness:.6f}')
