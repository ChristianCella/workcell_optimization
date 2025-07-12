# --------------------------------------------christian.cella@polimi.it----------------------------------------------- #
# --------------------academic year: 2022 - 2023, personal code: 10615676, ID number: 235974-------------------------- #

# ____PhD PROGRAM: AI-ENHANCED ENGINEERING TOOLS FOR THE OPTIMAL DEPLOYMENT OF COLLABORATIVE ROBOTICS APPLICATIONS____ #
# ____________________________________________________________________________________________________________________ #

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import mpl_toolkits.mplot3d as tool
import random

import warnings
warnings.filterwarnings('ignore')

from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as ker
from scipy.optimize import minimize, linprog

# ------------------------------------- LINES OF CODE TU USE LaTeX IN THE PLOTS -------------------------------------- #

matplotlib.rcParams['mathtext.fontset'] = 'cm' # 'cm' or 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

matplotlib.rc('xtick', labelsize = 15)
matplotlib.rc('ytick', labelsize = 15)

# _________________________________________ DEFINITION OF THE FUNCTIONS ______________________________________________ #

# definition of the function

def target_function(x) :

    return np.exp(-(x - 2) ** 2) + np.exp(-(x - 6) ** 2 / 10) + 1 / (x ** 2 + 1)

# generation of the initial set

def generate_initial_data(n) :

    train_x = np.random.uniform(0, 6, n)
    exact_obj = target_function(train_x)
    best_observed_value = np.max(exact_obj)

    return train_x, exact_obj, best_observed_value

# definition of the acquisition function (UCB = Upper Confidence Bound)

def acquisition_UCB(X, GPR_model, kappa) :

    if len(X.shape) == 1 :

        X = np.expand_dims(X, axis = 0)

    mean, std = GPR_model.predict(X.reshape(-1, 1), return_std = True) # this is actually implementing the Surrogate Function

    # adjust the dimensions of the vectors

    mean = mean.flatten()
    std = std.flatten()

    # real acquisition function (UCB); "kappa" regulates the exploration and exploitation

    ucb = mean - kappa * std

    return ucb

# definition of the optimization procedure

def optimize_acquisition(GPR_model, n, anchor_number, kappa) :

    # creation of the random points (n = 100 in the main)

    random_points = np.random.uniform(0, 6, n)

    acquisition_values = acquisition_UCB(random_points, GPR_model, kappa)

    # keep the best N = "anchor_number" points

    best_predictions = np.argsort(acquisition_values)[0 : anchor_number] # find their positions
    selected_anchors = random_points[best_predictions] # get the anchor points

    #print("the random points for anchors are : " + str(random_points))
    #print("the selected anchors are : " + str(selected_anchors))

    optimized_points = []
    for anchor in selected_anchors :

        acq= lambda anchor, GPR_model: acquisition_UCB(anchor, GPR_model, kappa)
        result = minimize(acq, anchor, GPR_model, method = 'Nelder-Mead', bounds = [(0, 6)])
        optimized_points.append(result.x)

    # of the N = "anchor_points" optimal points I found, I choose the best

    optimized_points = np.array(optimized_points)
    optimized_acquisition_values= acquisition_UCB(optimized_points, GPR_model, kappa) # get K_RBF of all the opt. points
    best = np.argsort(optimized_acquisition_values)[0]
    x_next = optimized_points[best] # store the best among the best

    return np.expand_dims(x_next, axis = 0)

# ---------------------------------------------------- CONSTANTS ----------------------------------------------------- #

x_inf = 0
x_sup = 6
disc_func = 100
training_set_dim = 10
dot_size = 30
font_size = 20
fig_size = 10

kappa = 0.8 # if kappa tends towards 0: importance of the mean increases (exploitation); if kappa increases: the
# importance of sigma increases (exploration)
n = 100
anchor_number = 5
num_iters = 15

# ____________________________ GENERATION OF THE INITIAL SET AND FUNCTION RECONSTRUCTION _____________________________ #

x = np.linspace(x_inf, x_sup, disc_func)
z = target_function(x)

# definition of the dataset for the training

train_x = np.random.uniform(x_inf, x_sup, training_set_dim)
exact_obj = target_function(train_x)

# evaluation of the best observed value

best_observed_value = np.max(exact_obj)

# get a first optimization iteration based on initial data ==> get the next sampling point

init_x, init_y, best_init_y = generate_initial_data(training_set_dim)

# Find the predictions of the function (try with different kernels)

konst = ker.ConstantKernel(2)
kernels = [
    ker.RationalQuadratic() * konst + ker.RBF() + ker.ConstantKernel(0.5),
    ker.RBF(length_scale = 10, length_scale_bounds=(10, 10.01)),
    ker.RBF() * konst
]

# start the gaussian process and get the prediction of the function based on the initial training set

gpr = GaussianProcessRegressor(kernels[1])
gpr.fit(init_x.reshape(-1, 1), init_y)

y_pred = gpr.predict(init_x.reshape(-1, 1))

# plot the predictions of the function and the initial set (the result is based on the selected kernel)

fig, ax = plt.subplots(figsize = [fig_size, fig_size])
plt.plot(x, z)

for i in range(len(y_pred)) :

    sampled = plt.scatter(init_x[i], init_y[i], c = 'red', s = dot_size)
    predicted = plt.scatter(init_x[i], y_pred[i], c = 'blue', s = dot_size)

plt.xlabel('x = input', fontsize = font_size)
plt.ylabel('y = KPI', fontsize = font_size)
plt.title('Predicted points', fontsize = font_size)
plt.legend([sampled, predicted], ["sampled", "predicted"], loc = 'upper right')
plt.grid()
plt.show()

# ___________________________________________ BAYESIAN OPTIMIZATION __________________________________________________ #

x_dataset = np.random.uniform(x_inf, x_sup, 1) # generate only one sample
y_dataset = target_function(x_dataset)

# create the gaussian process

GP = GaussianProcessRegressor(kernel = 1.5 * ker.RBF(length_scale = 1, length_scale_bounds = (1e-05, 1000)),
                              n_restarts_optimizer = 9)
GP.fit(x_dataset.reshape(-1, 1), y_dataset)

# start the iteration procedure

for i in range (num_iters) :

    # evaluate the next point (observation)

    x_next = optimize_acquisition(GP, n, anchor_number, kappa)
    # At this point the simulation with the DT starts
    eval_x_next = target_function(x_next) # In reality I can not do this because I do not know the function

    # get the shape of the acquisition function (its min is going to be the next sample)

    total_acquisition_function = acquisition_UCB(x, GP, kappa)

    # print some instructions

    print('Iteration number : ' + str(i) + ')\n\n')
    print('X_next : ' + str(x_next.flatten()))
    print('Acquisition function value of X_next : ' + str(acquisition_UCB(x_next, GP, kappa)))
    print('Measured value of X_next : ' + str(eval_x_next.flatten()) + '\n\n')

    # increment the dataset (augmented vectors)

    x_dataset = np.append(x_dataset, x_next[0], axis=0)
    y_dataset = np.append(y_dataset, eval_x_next[0], axis=0)

    # redefinition and re-train of the GPR using the updated (augmented) dataset (valid from the next iteration)
    # at each iteration the Y_dataset is augmented and more values are available for the probability evaluation

    GP = GaussianProcessRegressor(kernel=1.5 * ker.RBF(length_scale=1.0, length_scale_bounds=(1e-05, 1000)),
                                  n_restarts_optimizer=9)
    GP.fit(x_dataset.reshape(-1, 1), y_dataset)

    # get the vectors of mean values and standard deviations (their dimensions increase at each iteration)

    mean_pred, std_pred = GP.predict(x_dataset.reshape(-1, 1), return_std = True)

    fig, ax = plt.subplots(figsize = [fig_size, fig_size])

    if i == 0 :

        init_plot = plt.scatter(x_dataset[i], y_dataset[i], c = 'red', s = dot_size)

    elif i == num_iters - 1 :

        final_plot = plt.scatter(x_dataset[i], y_dataset[i], c = 'blue', s = dot_size)

    else :

        plt.scatter(x_dataset[i], y_dataset[i], c = 'green', s = dot_size)


    plt.plot(x, z) # function
    plt.plot(x, total_acquisition_function) # acquisition function
    plt.xlabel('x = input', fontsize = 15)
    plt.ylabel('y = KPI to minimize', fontsize = 15)
    plt.title('Iteration : ' + str(i), fontsize = 20)
    plt.grid()
    plt.show()


