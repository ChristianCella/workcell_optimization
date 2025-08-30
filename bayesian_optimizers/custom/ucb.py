"""
First attempt to implement the Bayesian Optimization algorithm for the constrained problem.
You are not currently using a virtual environment (not good).
"""

# General libraries

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import mpl_toolkits.mplot3d as tool
import warnings

# Do not display warnings

warnings.filterwarnings('ignore')

# Import the libraries for the bayesian optimization

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
import sklearn.gaussian_process.kernels as ker
import numpy as np
from scipy.optimize import minimize

# libraries for latex

matplotlib.rcParams['mathtext.fontset'] = 'cm' # 'cm' or 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

matplotlib.rc('xtick', labelsize = 15)
matplotlib.rc('ytick', labelsize = 15)

# defiition of the "hidden" objective function, this is like cheating because in the real case I will not have it

def fobj(x1, x2) :

    return x1 ** 2 + x2 ** 2

"""
Define the constraint function (the constraints are intended to be expressed as ">= 0")
For example, "return - (x[0] + x[1]) + (5)" means that (x(0) + x(1)) <= 5  
What is the meaning of this? we are asking the "minimize" function, that works on the variable on which f_acq depends:
what I will find is that the "x_next" value will respect exactly the constraint I imposed
"""

def constraint_func1(x):
    
    return - x[0] - x[1] + custom_param1


# Definition of the acquisition function (UCB)

def acquisition_UCB(X, GPR_model, kappa) :

    if len(X.shape) == 1 :

        X = np.expand_dims(X, axis = 0)

    mean, std = GPR_model.predict(X, return_std = True) # this is actually implementing the Surrogate Function

    # adjust the dimensions of the vectors

    mean = mean.flatten()
    std = std.flatten()

    # real acquisition function (UCB); "kappa" regulates the exploration and exploitation

    ucb = mean + kappa * std

    return ucb

"""
define the optimization function: I first apply the acquisition function to get the "best" points; then I take
the first "anchor_number" points among the "n" I randomly generated on the domain
"""

def optimize_acquisition(GPR_model, n, anchor_number, kappa, x_inf, x_sup, constraint) :

    # creation of the random points (n = 100 in the main)

    random_points = np.random.uniform(x_inf, x_sup, (n,2)) # I create a matrix (2) of random numbers from -10 to 10
    acquisition_values = acquisition_UCB(random_points, GPR_model, kappa)

    # keep the best N = "anchor_number" points

    best_predictions = np.argsort(acquisition_values)[0 : anchor_number] # find their positions
    selected_anchors = random_points[best_predictions] # get the anchor points
    
    # Initialize a vector to store the optimized points
    
    optimized_points = []
    
    for anchor in selected_anchors :

        # in "acq" store the acquisition function (UCB) evaluated at the i-th anchor point
        
        acq = lambda anchor, GPR_model: acquisition_UCB(anchor, GPR_model, kappa)
        
        """
        Real minimization procedure: the constraints DO NOT work on "Nelder-Mead" method, but, for example, 
        they work with SLSQP
        """
        
        result = minimize(acq, anchor, GPR_model, method = 'SLSQP', bounds = ((x_inf[0], x_sup[0]), (x_inf[1], x_sup[1])),
                          constraints = constraint)
        optimized_points.append(result.x)

    # of the N = "anchor_points" optimal points I found, I choose the best

    optimized_points = np.array(optimized_points)
    optimized_acquisition_values = acquisition_UCB(optimized_points, GPR_model, kappa) # get K_RBF of all the opt. points
    best = np.argsort(optimized_acquisition_values)[0]
    
    # The "x_next" is the tentative point taht will respect the constraints
    
    x_next = optimized_points[best] # store the best among the best

    return np.expand_dims(x_next, axis = 0)

# This function is the same as the objective function

def hidden_f(X) :

    return X[:, 0] ** 2 + X[:, 1] ** 2

"""
Main: the code allows to verify that the constraints can be imposed in a BAyesian scheme
"""

# set the limits (constraints given by the text)

xmin = -10
xmax = 10

x_inf = np.array([xmin, xmin])
x_sup = np.array([xmax, xmax])

# Trade-off exploration-exploitation

kappa = 0.1

# number of random points, number of anchor points and number of iterations

n = 100 
anchor_number = 10
num_iters = 25
step_plot = 0.5

# Specify the parameter of the constraint

custom_param1 = 5
custom_param2 = 2

# Number of initial points for the training

N = 1 # we start with only one point selected randomly

X_dataset = np.random.uniform(-10, 10, (N, 2))
Y_dataset = hidden_f(X_dataset).reshape(-1, 1)

# creation and training of the initial GPR using the dataset above

GP = GaussianProcessRegressor(kernel = 1.5 * ker.RBF(length_scale = 1, length_scale_bounds = (1e-05, 1000)),
                            n_restarts_optimizer = 1)
GP.fit(X_dataset, Y_dataset)

#
# Start the loop procedure to get the final candidate
#

# Liste per tracciare l'andamento dei valori durante le iterazioni
history_x1 = []
history_x2 = []
history_y = []
history_ucb = []

for i in range(num_iters) :

    # Define the constraint(s)
    
    constraint = [
        {'type': 'ineq', 'fun': constraint_func1}
    ]
    
    
    # Get the new "tentative" point
    x_next = optimize_acquisition(GP, n, anchor_number, kappa, x_inf, x_sup, constraint)

    # Evaluate the new candidate (Perform a new simulation)
    eval_x_next = hidden_f(x_next).reshape(-1, 1)

    # Tracciamento dei valori per ogni iterazione
    history_x1.append(x_next[0, 0])
    history_x2.append(x_next[0, 1])
    history_y.append(eval_x_next[0, 0])
    history_ucb.append(acquisition_UCB(x_next, GP, kappa)[0])

    # Print some useful numbers
    print('Iteration number : ' + str(i) + ')\n\n')
    print('X_next : ' + str(x_next.flatten()))
    print('Acquisition function value of X_next : ' + str(acquisition_UCB(x_next, GP, kappa)))

    # Augment the dataset
    X_dataset = np.append(X_dataset, x_next, axis = 0)
    Y_dataset = np.append(Y_dataset, eval_x_next, axis = 0)

    # redefinition and re-train of the GPR using the updated (augmented) dataset (valid from the next iteration)
    GP = GaussianProcessRegressor(kernel = 1.5 * ker.RBF(length_scale = 1.0, length_scale_bounds = (1e-05, 1000)),
                                n_restarts_optimizer = 1)
    GP.fit(X_dataset, Y_dataset)
    
"""
Plot the "hidden" function (the one that I am trying to minimize) and the minimum value obtained by the 
constrained minimization.
The red dot in the plot is the minimum value obtained by the constrained minimization.
"""

# Define the domain to plot the objective function

x_1 = np.arange(xmin, xmax, step_plot)
x_2 = np.arange(xmin, xmax, step_plot)

X1, X2 = np.meshgrid(x_1, x_2)
fun = fobj(X1, X2)

fig, ax = plt.subplots(subplot_kw = {"projection": "3d"}, figsize = [10, 10])
ax.plot_surface(X1, X2, fun, alpha = 0.5)
ax.scatter(X_dataset[: - 1, 0], X_dataset[: - 1, 1], Y_dataset[: - 1], c = 'blue', s = 25)
ax.scatter(X_dataset[- 1, 0], X_dataset[- 1, 1], Y_dataset[- 1], c = 'red', s = 25)

plt.xlabel('x1 set')
plt.ylabel('X2 set')
plt.title('Hidden function', fontsize = 20)
# Mostra il plot della funzione nascosta
plt.show()
    
# Define the domain to plot the yhperplanes of the constraints

x_p1 = np.arange(xmin, xmax, step_plot)
y_p1 = np.arange(xmin, xmax, step_plot)
x_plane1, y_plane1 = np.meshgrid(x_p1, y_p1)
z_plane1 = - 5 + x_plane1 + y_plane1

x_p2 = np.arange(xmin, xmax, step_plot)
y_p2 = np.arange(xmin, xmax, step_plot)
x_plane2, y_plane2 = np.meshgrid(x_p2, y_p2)
z_plane2 = 2 - x_plane2 - y_plane2

fig, ax = plt.subplots(subplot_kw = {"projection": "3d"}, figsize = [10, 10])
ax.plot_surface(x_plane1, y_plane1, z_plane1, alpha = 0.5, color = 'green')
ax.plot_surface(x_plane2, y_plane2, z_plane2, alpha = 0.5, color = 'blue')
ax.scatter(X_dataset[: - 1, 0], X_dataset[: - 1, 1], X_dataset[: - 1, 0] + X_dataset[: - 1, 1], c = 'blue', s = 25)

plt.xlabel('x1 set')
plt.ylabel('X2 set')
plt.title('Graphical meaning of the constraint', fontsize = 20)
plt.show()

# Plot dell'andamento nel tempo
iterations = np.arange(1, num_iters + 1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(iterations, history_x1, label='x1')
plt.plot(iterations, history_x2, label='x2')
plt.xlabel('Iterazione')
plt.ylabel('Valore')
plt.title('Andamento di x1 e x2')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(iterations, history_y, color='green')
plt.xlabel('Iterazione')
plt.ylabel('f(x)')
plt.title('Andamento di y')

plt.subplot(1, 3, 3)
plt.plot(iterations, history_ucb, color='orange')
plt.xlabel('Iterazione')
plt.ylabel('Upper Confidence Bound')
plt.title('Andamento di UCB')

plt.tight_layout()
plt.show()
