# --------------------------------------------christian.cella@polimi.it----------------------------------------------- #
# --------------------academic year: 2022 - 2023, personal code: 10615676, ID number: 235974-------------------------- #

# ____PhD PROGRAM: AI-ENHANCED ENGINEERING TOOLS FOR THE OPTIMAL DEPLOYMENT OF COLLABORATIVE ROBOTICS APPLICATIONS____ #
# ____________________________________________________________________________________________________________________ #
# how to handle out of reach coordinates? in differential evolution , i implemented a thing
#that  if the coordinates are out of reach, the function returns a very high value
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import mpl_toolkits.mplot3d as tool
import warnings


#from position_viewer import init_viewer, update_viewer, close_viewer

# Do not display warnings

warnings.filterwarnings('ignore')

# Import the libraries for the bayesian optimization

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
import sklearn.gaussian_process.kernels as ker
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from Differential_evolution_slim import func
from sklearn.gaussian_process.kernels import RationalQuadratic
# libraries for latex

matplotlib.rcParams['mathtext.fontset'] = 'cm' # 'cm' or 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

matplotlib.rc('xtick', labelsize = 15)
matplotlib.rc('ytick', labelsize = 15)
from Differential_evolution_slim import func 




# ------------------------------------- LINES OF CODE TU USE LaTeX IN THE PLOTS -------------------------------------- #

matplotlib.rcParams['mathtext.fontset'] = 'cm' # 'cm' or 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

matplotlib.rc('xtick', labelsize = 15)
matplotlib.rc('ytick', labelsize = 15)

# _________________________________________ DEFINITION OF THE FUNCTIONS ______________________________________________ #

# definition of the function

#def target_function(x,y) :

    #return func(x,y)


def constraint_func1(x): # la base del robot deve essere dentro un cerchio di raggio radius centrato in 0
    
    return - (x[0]**2 + x[1]**2) + radius**2

def constraint_func2(x): # z della bassa deve essere inferiore ad height
    
    return - x[2] + height

def constraint_func3(x): # z deve essere maggiore di zero
    
    return  x[2] + 0

def expected_improvement(X, GPR_model, best_y):
    
    if len(X.shape) == 1 :

        X = np.expand_dims(X, axis = 0)
    
    mean, std = GPR_model.predict(X, return_std = True)
    mean = mean.flatten()
    std = std.flatten() 
 
    z = (mean - best_y) / std
    ei = (mean - best_y) * norm.cdf(z) + std * norm.pdf(z)
    # ei = std * (z * norm.cdf(z) + norm.pdf(z))
    
    all_greater_than_zero = np.all(ei > 0)
    
    return ei
    
    """
    if all_greater_than_zero:
    
        return ei
    
    else:
        
        print("MIAO")
        return np.zeros(100)
        
    """

"""
define the optimization function: I first apply the acquisition function to get the "best" points; then I take
the first "anchor_number" points among the "n" I randomly generated on the domain
"""

def optimize_acquisition(GPR_model, n, anchor_number, best_evaluation, x_inf, x_sup,constraint):

    # creation of the random points (n = 100 in the main)

    random_points = np.random.uniform(x_inf, x_sup, (n,3)) # I create a matrix (2) of random numbers from -10 to 10
    acquisition_values = expected_improvement(random_points, GPR_model, best_evaluation)

    # keep the best N = "anchor_number" points

    best_predictions = np.argsort(acquisition_values)[0 : anchor_number] # find their positions
    selected_anchors = random_points[best_predictions] # get the anchor points
    
    # Initialize a vector to store the optimized points
    
    optimized_points = []
    
    for anchor in selected_anchors :

        # in "acq" store the acquisition function (UCB) evaluated at the i-th anchor point
        
        acq = lambda anchor, GPR_model: expected_improvement(anchor, GPR_model, best_evaluation)
        
        """
        Real minimization procedure: the constraints DO NOT work on "Nelder-Mead" method, but, for example, 
        they work with SLSQP
        """
        
        result = minimize(acq, anchor, GPR_model, method = 'SLSQP', bounds = ((x_inf[0], x_sup[0]), (x_inf[1], x_sup[1]),(x_inf[2], x_sup[2])),constraints=constraint)
        optimized_points.append(result.x)

    # of the N = "anchor_points" optimal points I found, I choose the best

    optimized_points = np.array(optimized_points)
    optimized_acquisition_values = expected_improvement(optimized_points, GPR_model, best_evaluation) # get K_RBF of all the opt. points
    best = np.argsort(optimized_acquisition_values)[0]
    
    # The "x_next" is the tentative point taht will respect the constraints
    
    x_next = optimized_points[best] # store the best among the best

    return np.expand_dims(x_next, axis = 0)

# This function is the same as the objective function

def hidden_f(X): #questa è per il caso x_next
 
    X = np.atleast_2d(X)                        # garantisce shape (N,2)
    return np.array([(func(row[0], row[1],row[2])/3) for row in X])
   
def hidden_f_test(X): # questa è per il caso di sampling
    Y = np.empty(X.shape[0])          
    for i in range(X.shape[0]):
        Y[i] = func(X[i, 0], X[i, 1],X[i,2])/3
        print("valore iteazione:",i) 
    return Y

"""
Main: the code allows to verify that the constraints can be imposed in a BAyesian scheme
"""

# set the limits (constraints given by the text)

xmin = -0.5
xmax = 0.5

xminh=0
xmaxh=0.5

x_inf = np.array([xmin, xmin,xminh])
x_sup = np.array([xmax, xmax,xmaxh])

# Trade-off exploration-exploitation

kappa = 0.5

# number of random points, number of anchor points and number of iterations

n = 100 
anchor_number = 20
num_iters = 100
step_plot = 0.5
 

# Specify the parameter of the constraint

radius = 0.5#
height=0.5 #0.5
# Number of initial points for the training

N = 100# we start with only one point selected randomly
#generazione dei point
xy = np.random.uniform(xmin, xmax, (N, 2))
z = np.random.uniform(xminh, xmaxh, (N, 1))
X_dataset = np.hstack((xy, z))
Y_dataset = hidden_f_test(X_dataset).reshape(-1, 1)
#Y_dataset = hidden_f(X_dataset).reshape(-1, 1)
#init_viewer(X_dataset[0,0], X_dataset[0,1], X_dataset[0,2])
# creation and training of the initial GPR using the dataset above
kernel = 1.5 * RationalQuadratic(length_scale=0.001, alpha=1.0,
                                 length_scale_bounds=(1e-3, 5.0),
                                 alpha_bounds=(1e-2, 100))

# -----------------------
# Creazione modello GPR
# -----------------------
GP = GaussianProcessRegressor(kernel=kernel,
                              normalize_y=True,
                              alpha=1e-6,  # noise level
                              n_restarts_optimizer=10)
#GP = GaussianProcessRegressor(kernel = 1.5 * ker.Matern(length_scale = 0.05, length_scale_bounds = (0.01, 0.1)),n_restarts_optimizer = 15)
#GP = GaussianProcessRegressor(kernel = ker.RationalQuadratic(), n_restarts_optimizer = 10)
GP.fit(X_dataset, Y_dataset)

# Start the loop procedure to get the final candidate
y_history = [] 
for i in range(num_iters) :

    constraint=[{'type': 'ineq', 'fun': constraint_func1},{'type': 'ineq', 'fun': constraint_func2},{'type': 'ineq', 'fun': constraint_func3}]
   
   # Get the new "tentative" point
    
    best_evaluation = np.min(Y_dataset)
    x_next = optimize_acquisition(GP, n, anchor_number, best_evaluation, x_inf, x_sup,constraint)
    
    # Evaluate the new candidate (Perform a new simulation)
    
    eval_x_next = hidden_f(x_next).reshape(-1, 1)
    y_history.append(float(eval_x_next)) 
    # Print some useful numbers

    print('Iteration number : ' + str(i) + ')\n\n')
    print('X_next : ' + str(x_next.flatten()))
    print('Acquisition function value of X_next : ' + str(expected_improvement(x_next, GP, best_evaluation)))
    print('Evaluation of the objective function at X_next : ' + str(eval_x_next.flatten()))
    print('tau : ' + str(eval_x_next.flatten()))
    # Augment the dataset
    #update_viewer(x_next[0,0], x_next[0,1], x_next[0,2])
    X_dataset = np.append(X_dataset, x_next, axis = 0)
    Y_dataset = np.append(Y_dataset, eval_x_next, axis = 0)

    # redefinition and re-train of the GPR using the updated (augmented) dataset (valid from the next iteration)

    #GP = GaussianProcessRegressor(kernel = 1.5 * ker.RBF(length_scale = 1.0, length_scale_bounds = (1e-05, 1000)),
                              #  n_restarts_optimizer = 1)
   # GP = GaussianProcessRegressor(kernel = 1.5 * ker.Matern(length_scale = 0.05, length_scale_bounds = (0.01, 0.1)),n_restarts_optimizer = 2)                       
    #GP = GaussianProcessRegressor(kernel = ker.RationalQuadratic(), n_restarts_optimizer = 10)
    GP = GaussianProcessRegressor(kernel=kernel,normalize_y=True,alpha=1e-6, n_restarts_optimizer=10)
    GP.fit(X_dataset, Y_dataset)
 

# plot dei risultati
best_idx = np.argmin(Y_dataset)          

# coord. di tutti i punti valutati
xs = X_dataset[:, 0]
ys = X_dataset[:, 1]
zs = Y_dataset.flatten()

# coord. del punto ottimo
best_x, best_y, best_z = xs[best_idx], ys[best_idx], zs[best_idx]

  # necessario per la proiezione 3-D
import matplotlib
matplotlib.use('Qt5Agg')     # oppure 'Agg' se non ti serve la finestra
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 6))
ax  = fig.add_subplot(111, projection="3d")

# punti valutati (blu)
ax.scatter(xs, ys, zs, c="blue", label="Punti valutati")

# punto ottimo (rosso)
ax.scatter(best_x, best_y, best_z, c="red", s=80, label="Ottimo")

ax.set_xlabel(r"$x$", fontsize=14)
ax.set_ylabel(r"$y$", fontsize=14)
ax.set_zlabel(r"$f(x,y)$", fontsize=14)
ax.set_title("Valutazioni Bayesiane e minimo trovato", fontsize=15)
ax.legend()
plt.tight_layout()
#plt.savefig("bayesian_optimization_result.png", dpi=300)
plt.show()





# ---- PLOT 2-D dell'andamento di y per iterazione ----
import matplotlib.pyplot as plt

plt.figure(figsize=(7,4))
plt.plot(range(1, len(y_history)+1), y_history, marker='o')
plt.xlabel('Iterazione')
plt.ylabel('Valore y')
plt.title('Evoluzione dell\'obiettivo')
plt.grid(True)
plt.tight_layout()
# plt.savefig("progressione_y.png", dpi=150)   
plt.show()




