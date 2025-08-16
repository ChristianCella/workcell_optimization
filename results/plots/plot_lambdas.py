import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# --- paths ---
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
alpha_path = os.path.join(base_dir, "complete_alpha_trend_wide.csv")
beta_path  = os.path.join(base_dir, "complete_beta_trend_wide.csv")
gamma_path = os.path.join(base_dir, "complete_gamma_trend_wide.csv")
index_path = os.path.join(base_dir, "best_individuals_indices.csv")

# --- load ---
df_alpha = pd.read_csv(alpha_path)
df_beta  = pd.read_csv(beta_path)
df_gamma = pd.read_csv(gamma_path)
df_idx   = pd.read_csv(index_path).set_index("generation").sort_index()

# infer NT (joints per 'piece' group)
NT = sum(c.startswith("piece1_joint") for c in df_alpha.columns)
gens = range(len(df_alpha))

# compute lambda_hat for the best individual each generation
rows = []
eps = 1e-12

for gen in gens:
    # columns are 1-based: piece1_joint1, piece1_joint2, ...
    best_ind = int(df_idx.loc[gen, "best_individual"]) + 1
    cols = [f"piece{best_ind}_joint{j}" for j in range(1, NT + 1)]

    alpha = df_alpha.loc[gen, cols].to_numpy(dtype=float)
    beta  = df_beta.loc[gen,  cols].to_numpy(dtype=float)
    gamma = df_gamma.loc[gen, cols].to_numpy(dtype=float)

    # hat-lambda normally does NOT depend on f*, use -beta/alpha
    lambda_hat = np.full_like(alpha, np.nan, dtype=float)
    ok = np.abs(alpha) > eps
    lambda_hat[ok] = -beta[ok] / alpha[ok]

    # fallback when alpha ~ 0 but beta != 0:
    # use f_*^2 = alpha + beta + gamma (elementwise) as you requested
    lin = (~ok) & (np.abs(beta) > eps)
    fstar_sq_local = np.clip(alpha + beta + gamma, a_min=0.0, a_max=None)
    lambda_hat[lin] = (fstar_sq_local[lin] - gamma[lin]) / (2.0 * beta[lin])

    for j in range(NT):
        rows.append({"generation": gen, "piece": j+1, "lambda_hat": lambda_hat[j]})

lam_long = pd.DataFrame(rows).sort_values(["generation", "piece"])
lam_wide = lam_long.pivot(index="generation", columns="piece", values="lambda_hat").sort_index()
lam_wide.columns = [f"lambda_hat_piece{p}" for p in lam_wide.columns]

# ---- plot: stars, piece 1 = red ... piece NT = blue ----
plt.figure(figsize=(9, 4.5))

# build colors: first = red, last = blue, interpolate in between
if NT == 1:
    colors = ["red"]
else:
    cmap = plt.cm.coolwarm_r  # red -> ... -> blue
    colors = [cmap(i/(NT-1)) for i in range(NT)]

x = lam_wide.index.values  # iterations (generations)
for p in range(1, NT+1):
    y = lam_wide[f"lambda_hat_piece{p}"].to_numpy()
    mask = ~np.isnan(y)
    plt.plot(x[mask], y[mask], marker='*', linestyle='None', markersize=10,
             label=f"piece {p}", color=colors[p-1])

plt.xlabel("Iteration (generation)")
plt.ylabel(r"$\hat{\lambda}$")
plt.title(r"$\hat{\lambda}$ per piece (best individual each generation)")
plt.legend(frameon=False, ncol=min(NT, 4))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
