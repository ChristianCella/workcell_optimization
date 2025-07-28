'''
QUESTO CODICE OTTIMIZZA LA POSIZIONE X,Y DELLA BASE DEL ROBOT, COME ACQUISITION FUNCTION USA UCB
LA FUNZIONE HIDDEN_f CHIAMA LO SCRIPT DE_XY.py CHE DPO UN PROCESSO DI OTTIMIZZAZIONE 
RITORNA IL MODULO DELLE COPPIE NELLA CONFIGURAZIONE OTTIMIZZATA (SINGOLO FRAME NON TRE)
  N  è IL NUMERO  DI PUNTI UTILIZZATI PER ADDESTRARE IL GPR
  n è IL NUMERO DI PUNTI UTILIZZATI PER OGNI SINGOLA ITERAZIOEN
LO SCRIPT RITORNA 3 GRAFICI , IL PRIMO è LA FUNZIONE VALUTATA NELLO SPAZIO X,Y, IL PUNTO ROSSO è L'OTTIMO
IL SECONDO GRAFICO è IL VALORE DELLA FUNZIONE AD OGNI ITERAZIONE
IL TERZO GRAFICO PLOTTA ANCHE IL BEST UP TO NOW
KAPPA REGLOA L'ESPLORAZIOE E LO SFRUTTAMENTO
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import mpl_toolkits.mplot3d as tool
import warnings


from position_viewer import init_viewer, update_viewer, close_viewer

# Do not display warnings

warnings.filterwarnings('ignore')

# Import the libraries for the bayesian optimization

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
import sklearn.gaussian_process.kernels as ker
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from DE_XY import func
from sklearn.gaussian_process.kernels import RationalQuadratic
# libraries for latex

matplotlib.rcParams['mathtext.fontset'] = 'cm' # 'cm' or 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

matplotlib.rc('xtick', labelsize = 15)
matplotlib.rc('ytick', labelsize = 15)
#from Differential_evolution_slim import func 




# ------------------------------------- LINES OF CODE TU USE LaTeX IN THE PLOTS -------------------------------------- #

matplotlib.rcParams['mathtext.fontset'] = 'cm' # 'cm' or 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

matplotlib.rc('xtick', labelsize = 15)
matplotlib.rc('ytick', labelsize = 15)

# _________________________________________ DEFINITION OF THE FUNCTIONS ______________________________________________ #




def constraint_func1(x): # la base del robot deve essere dentro un cerchio di raggio radius centrato in 0
    
    return - (x[0]**2 + x[1]**2) + radius**2


#DEFINIZIONE ACQUISITION FUNCTION
def UCB(X, GPR_model, kappa) :

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

def optimize_acquisition(GPR_model, n, anchor_number, x_inf, x_sup,constraint,kappa):

    # creation of the random points (n = 100 in the main)

    random_points = np.random.uniform(x_inf, x_sup, (n,2)) # I create a matrix (2) of random numbers from -10 to 10
    acquisition_values = UCB(random_points, GPR_model, kappa) # I apply the UCB acquisition function to these points

    # keep the best N = "anchor_number" points

    best_predictions = np.argsort(acquisition_values)[0 : anchor_number] # find their positions
    selected_anchors = random_points[best_predictions] # get the anchor points
    
    # Initialize a vector to store the optimized points
    
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

    # of the N = "anchor_points" optimal points I found, I choose the best

    optimized_points = np.array(optimized_points)
    optimized_acquisition_values = UCB(optimized_points, GPR_model, kappa) # get K_RBF of all the opt. points
    best = np.argsort(optimized_acquisition_values)[0]
    
    # The "x_next" is the tentative point taht will respect the constraints
    
    x_next = optimized_points[best] # store the best among the best

    return np.expand_dims(x_next, axis = 0)

# This function is the same as the objective function

def hidden_f(X): #questa è per il caso x_next
 
    X = np.atleast_2d(X)                        # garantisce shape (N,2)
    return np.array([func(row[0], row[1]) for row in X])
   
def hidden_f_test(X): # questa è per il caso di sampling
    Y = np.empty(X.shape[0])          
    for i in range(X.shape[0]):
        Y[i] = func(X[i, 0], X[i, 1])
        print("valore iteazione:",i) 
    return Y

"""
Main: the code allows to verify that the constraints can be imposed in a BAyesian scheme
"""

#DEFINIZIONE DELLO SPAZIO DI SAMPLIMG

xmin = -0.5
xmax = 0.5

x_inf = np.array([xmin, xmin])
x_sup = np.array([xmax, xmax])

# Trade-off exploration-exploitation

kappa = 0.000001

# number of random points, number of anchor points and number of iterations

n = 20 #PUNTI VALUTATI AD OGNI ITERAZIONE
anchor_number = 30
num_iters = 300 
step_plot = 0.5
 
# Specify the parameter of the constraint

radius = 0.5

# Number of initial points for the training

N = 300 #PUNTI UTILIZZATI PER ADDESTRARE IL GPR

X_dataset=np.random.uniform(xmin, xmax, (N, 2))
Y_dataset = hidden_f_test(X_dataset).reshape(-1, 1)
#init_viewer(X_dataset[0,0], X_dataset[0,1], X_dataset[0,2])

# creation and training of the initial GPR using the dataset above
kernel = 1.5 * ker.Matern(length_scale = 1, length_scale_bounds = (1e-05, 1000))
GP = GaussianProcessRegressor(kernel=kernel,
                              normalize_y=True,
                              alpha=1e-6,  # noise level
                              n_restarts_optimizer=10)
GP.fit(X_dataset, Y_dataset)

# Start the loop procedure to get the final candidate
y_history = [] 
best_sofar_hist = [] 
# aggiorna il minimo cumulativo

for i in range(num_iters) :

    constraint=[{'type': 'ineq', 'fun': constraint_func1}]
   
   # Get the new "tentative" point
    
    best_evaluation = np.min(Y_dataset)
    x_next = optimize_acquisition(GP, n, anchor_number, x_inf, x_sup,constraint,kappa)
    
    # Evaluate the new candidate (Perform a new simulation)
    
    eval_x_next = hidden_f(x_next).reshape(-1, 1)
    y_history.append(float(eval_x_next)) 
    current_best = float(eval_x_next) if not best_sofar_hist else min(best_sofar_hist[-1], float(eval_x_next))
    best_sofar_hist.append(current_best)
    # Print some useful numbers

    print('Iteration number : ' + str(i) + ')\n\n')
    print('X_next : ' + str(x_next.flatten()))
    print('Acquisition function value of X_next : ' + str(UCB(x_next, GP, kappa)))
    print('Evaluation of the objective function at X_next : ' + str(eval_x_next.flatten()))
    print('tau : ' + str(eval_x_next.flatten()))
 
  #  update_viewer(x_next[0,0], x_next[0,1], x_next[0,2])
    X_dataset = np.append(X_dataset, x_next, axis = 0)
    Y_dataset = np.append(Y_dataset, eval_x_next, axis = 0)

    # redefinition and re-train of the GPR using the updated (augmented) dataset (valid from the next iteration)

    #GP = GaussianProcessRegressor(kernel = 1.5 * ker.RBF(length_scale = 1.0, length_scale_bounds = (1e-05, 1000)),
                              #  n_restarts_optimizer = 1)
    # GP = GaussianProcessRegressor(kernel = 1.5 * ker.Matern(length_scale = 0.05, length_scale_bounds = (0.01, 0.1)),n_restarts_optimizer = 2)                       
    #GP = GaussianProcessRegressor(kernel = ker.RationalQuadratic(), n_restarts_optimizer = 10)
    GP = GaussianProcessRegressor(kernel=kernel,normalize_y=True,alpha=1e-6, n_restarts_optimizer=10)
    GP.fit(X_dataset, Y_dataset)
 

# plot 3D FUNZIONE
best_idx = np.argmin(Y_dataset)          

# coord. di tutti i punti valutati
xs = X_dataset[:, 0]
ys = X_dataset[:, 1]
zs = Y_dataset.flatten()

# coord. del punto ottimo
best_x, best_y, best_z = xs[best_idx], ys[best_idx], zs[best_idx]

 
import matplotlib
matplotlib.use('Qt5Agg')     
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

plt.figure(figsize=(7,4))
plt.plot(range(1, len(y_history)+1), y_history, marker='o', label='y all iter')
plt.plot(range(1, len(best_sofar_hist)+1), best_sofar_hist, marker='s', color='red', label='best so-far')
plt.xlabel('Iterazione')
plt.ylabel('Valore obiettivo')
plt.title('Evoluzione dell\'obiettivo e best-so-far')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


