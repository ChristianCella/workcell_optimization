"""
Constrained Bayesian Optimization with Gaussian Processes as the surrogate and
acquisition function Expected Improvement.

Aggiornamento: grafico dell'**obiettivo calcolato in x_next ad ogni iterazione**.
"""

# Standard libs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
from mpl_toolkits.mplot3d import Axes3D  # noqa

warnings.filterwarnings('ignore')

# ML / stats libs
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as ker
from scipy.optimize import minimize
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Plot style
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)

# ---------------------------------------------------------------------------
# Problem definition

def fobj(x1, x2):
    return x1 ** 2 + x2 ** 2


def hidden_f(X):
    return X[:, 0] ** 2 + X[:, 1] ** 2


def constraint_func1(x):
    return -x[0] - x[1] + custom_param1  # x0 + x1 ≤ custom_param1


def constraint_func2(x):
    return x[0] + x[1] - custom_param2   # x0 + x1 ≥ custom_param2

# ---------------------------------------------------------------------------
# Acquisition: Expected Improvement (minimisation)

def expected_improvement(X, gpr, best_y):
    X = np.atleast_2d(X)
    m, s = gpr.predict(X, return_std=True)
    m, s = m.ravel(), s.ravel()
    s = np.maximum(s, 1e-12)

    z = (best_y - m) / s
    ei = (best_y - m) * norm.cdf(z) + s * norm.pdf(z)
    return ei

# ---------------------------------------------------------------------------
# Acquisition optimiser

def optimise_acquisition(gpr, n_rand, n_anchor, best_y, x_inf, x_sup, constraints):
    rnd = np.random.uniform(x_inf, x_sup, (n_rand, 2))
    acq_vals = expected_improvement(rnd, gpr, best_y)
    anchors = rnd[np.argsort(acq_vals)[:n_anchor]]

    best_x = None
    best_val = np.inf
    for a in anchors:
        acq = lambda x: expected_improvement(x, gpr, best_y)
        res = minimize(acq, a, method='SLSQP', bounds=((x_inf[0], x_sup[0]), (x_inf[1], x_sup[1])),
                       constraints=constraints)
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    return best_x[None, :]

# ---------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    # Domain
    xmin, xmax = -10, 10
    x_inf, x_sup = np.array([xmin, xmin]), np.array([xmax, xmax])

    # BO params
    n_rand, n_anchor = 100, 10
    n_iters = 40
    step_plot = 0.5

    # Constraint constants
    custom_param1, custom_param2 = 5, 2

    # Initial design
    X_data = np.random.uniform(x_inf, x_sup, (1, 2))
    Y_data = hidden_f(X_data)[:, None]

    # History arrays
    x_next_hist = []  # optional, still stored
    y_next_hist = []  # requested: objective value at each iter

    # GP model
    kernel = 1.5 * ker.RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e3))
    GP = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, normalize_y=True, alpha=1e-6)
    GP.fit(X_data, Y_data)

    # BO loop
    for it in range(n_iters):
        constraints = [
            {'type': 'ineq', 'fun': constraint_func1},
            {'type': 'ineq', 'fun': constraint_func2}
        ]

        best_y = np.min(Y_data)
        x_next = optimise_acquisition(GP, n_rand, n_anchor, best_y, x_inf, x_sup, constraints)
        y_next = hidden_f(x_next)[:, None]

        # Store history
        x_next_hist.append(x_next.ravel())
        y_next_hist.append(float(y_next))

        # Update dataset & GP
        X_data = np.vstack([X_data, x_next])
        Y_data = np.vstack([Y_data, y_next])
        GP.fit(X_data, Y_data)

        print(f"Iter {it:02d}: x_next = {x_next.ravel()}, f(x_next) = {y_next.item():.4f}")

    # -----------------------------------------------------------------------
    # Plot 1: f(x_next) per iter
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(range(1, n_iters + 1), y_next_hist, marker='o', linestyle='-')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('$f(x_{\mathrm{next}})$')
    ax1.set_title('Objective value found at each iteration')
    ax1.grid(True)
    plt.tight_layout()

    # -----------------------------------------------------------------------
    # (Optional) Plot 2: evolution of x_next coordinates
    x_arr = np.array(x_next_hist)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(range(1, n_iters + 1), x_arr[:, 0], marker='o', label='$x_1$')
    ax2.plot(range(1, n_iters + 1), x_arr[:, 1], marker='s', label='$x_2$')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Coordinate value')
    ax2.set_title('Evolution of $x_{\mathrm{next}}$')
    ax2.grid(True)
    ax2.legend()
    plt.tight_layout()

    plt.show()