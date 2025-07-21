import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from cma import CMA
import tensorflow as tf
tf.random.set_seed(444)  # set random seed for reproducibility

# Ensure the utils directory is in the Python path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils'))
sys.path.append(base_dir)
from cma_es_utils import *
import fonts

''' 
This test script demonstrates how to use the CMA-ES algorithm to optimize a fitness function.
Original repo: https://github.com/srom/cma-es/tree/master
This code contains the same exampels as those provided in the jupyter notebook of the original repo.
'''

#! 1) Define the desired fitness functions

def fitness_fn_six_hump(x):
    """
    Six-Hump Camel Function
    """
    return (
        (4 - 2.1 * x[:,0]**2 + x[:,0]**4 / 3) * x[:,0]**2 +
        x[:,0] * x[:,1] +
        (-4 + 4 * x[:,1]**2) * x[:,1]**2
    )

def fitness_fn_schwefel(x):
    # use x.dtype (float32) everywhere
    dim = tf.cast(tf.shape(x)[1], x.dtype)
    return tf.constant(418.9829, dtype=x.dtype) * dim \
           - tf.reduce_sum(x * tf.sin(tf.sqrt(tf.abs(x))), axis=1)


# Plot the 3d surface
fig = plt.figure(figsize=(20, 6))
gs = mpl.gridspec.GridSpec(1, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0], projection='3d')

#plot_3d_surface(fitness_fn_six_hump, xlim=[-2, 2], ylim=[-1.1, 1.1], zlim=[-1.1, 5], view_init=[50, -80], fig=fig, ax=ax1)
plot_3d_surface(fitness_fn_schwefel, [-500, 500], [-500, 500], fig=fig, ax=ax1)
ax1.set_xlabel('\nX axis')
ax1.set_ylabel('\n\nY axis')
ax1.set_title('3D surface\n')

# Plot the 2d contour
ax2 = fig.add_subplot(gs[0, 1])
#plot_2d_contour(fitness_fn_six_hump, xlim=[-2, 2], ylim=[-1.1, 1.1], fig=fig, ax=ax2)
plot_2d_contour(fitness_fn_schwefel, [-500, 500], [-500, 500], fig=fig, ax=ax2)
ax2.set_title('2D contour')
fig.patch.set_facecolor('white')
fig.suptitle('Six-Hump Camel Function\n', fontsize='xx-large')
plt.show()

#! 2) Run CMA-ES optimization

cma = CMA(
  initial_solution=[0., 0.],
  initial_step_size=1000.0,
  fitness_function=fitness_fn_schwefel,
  population_size=50,
  store_trace=True,
  enforce_bounds=[[-500, 500], [-500, 500]],
)

best_solution, best_fitness = cma.search()

print('Number of generations:', cma.generation)
print(f'{fonts.purple}Best solution: [{best_solution[0]:.5f}, {best_solution[1]:.5f}]{fonts.reset}')
print(f'Best fitness: {best_fitness:.4f}')

#! 3) Visualize the optimization trace

# Evolution of the coordinates of the best solution
sns.set_theme(palette='colorblind', font_scale=1.2)  # better default style for matplotlib
plot_mean_coordinates(cma.trace)
plt.show()

# Plot a few generations
generations = [0, 8, 16, 20, 25, 30]
fig, _ = plot_generations(
    generations,
    cma.trace,
    fitness_fn_schwefel,
    xlim=[-500, 500], 
    ylim=[-500, 500],
)
plt.show()