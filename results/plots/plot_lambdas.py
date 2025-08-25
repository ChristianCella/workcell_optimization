import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Config / paths
# -----------------------------
NT = 4          # pieces per individual
NU = 6          # ν in Eq. (5); set this to the value used in your paper

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
alpha_path     = os.path.join(base_dir, "complete_alpha_trend_wide.csv")
beta_path      = os.path.join(base_dir, "complete_beta_trend_wide.csv")
gamma_path     = os.path.join(base_dir, "complete_gamma_trend_wide.csv")
best_idx_path  = os.path.join(base_dir, "best_individuals_indices.csv")
fitness_fl_path= os.path.join(base_dir, "fitness_fL.csv")  # used ONLY to detect improvements

# -----------------------------
# Helpers
# -----------------------------
def load_with_gen_index(csv_path: str, expected_len: int | None = None) -> pd.DataFrame:
    """Load CSV and set 'generation' index (accepts generation/iter/iteration/gen/g or an unnamed first col)."""
    df = pd.read_csv(csv_path)
    for c in df.columns:
        if str(c).strip().lower() in {"generation", "iter", "iteration", "gen", "g"}:
            df = df.set_index(c)
            break
    else:
        first = df.columns[0]
        if str(first).startswith("Unnamed") and pd.api.types.is_integer_dtype(df[first]):
            df = df.set_index(first)
        else:
            df.index = range(len(df))
    df.index.name = "generation"
    df = df.sort_index()
    if expected_len is not None:
        df = df.reindex(range(expected_len)).ffill().bfill()
    return df

def cols_for_individual(ind_1b: int, df_alpha: pd.DataFrame) -> list[str]:
    cols = [f"piece{ind_1b}_joint{j}" for j in range(1, NT + 1)]
    missing = [c for c in cols if c not in df_alpha.columns]
    if missing:
        raise KeyError(f"Missing columns for individual {ind_1b}: {missing}")
    return cols

def lambdas_eq5(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray, nu: float, eps: float = 1e-12):
    """
    Equation (5): α λ^2 + β λ + (γ - ν) = 0
                  λ± = (-β ± sqrt(β^2 + 4 α (ν - γ))) / (2 α)
    Safeguards:
      - if |α| ≤ eps and |β| > eps -> linear root λ = (ν - γ)/β (put into both)
      - if discriminant < 0 -> NaN
    """
    a, b, g = alpha, beta, gamma
    disc = b**2 + 4.0 * a * (nu - g)

    lam_p = np.full_like(a, np.nan, dtype=float)
    lam_m = np.full_like(a, np.nan, dtype=float)

    # Quadratic branch
    mask_q = (np.abs(a) > eps) & (disc >= 0.0)
    sqrt_disc = np.zeros_like(a, dtype=float)
    sqrt_disc[mask_q] = np.sqrt(disc[mask_q])
    lam_p[mask_q] = (-b[mask_q] + sqrt_disc[mask_q]) / (2.0 * a[mask_q])
    lam_m[mask_q] = (-b[mask_q] - sqrt_disc[mask_q]) / (2.0 * a[mask_q])

    # Linear fallback (α ~ 0, β != 0)
    mask_lin = (np.abs(a) <= eps) & (np.abs(b) > eps)
    lin = (nu - g[mask_lin]) / b[mask_lin]
    lam_p[mask_lin] = lin
    lam_m[mask_lin] = lin  # single root when quadratic collapses

    return lam_p, lam_m

def to_wide(rows, colname: str, prefix: str):
    wide = (pd.DataFrame(rows)
            .pivot(index="generation", columns="piece", values=colname)
            .sort_index())
    wide.columns = [f"{prefix}_piece{p}" for p in wide.columns]
    return wide

def plot_lambda_pairs(lam_plus_wide: pd.DataFrame, lam_minus_wide: pd.DataFrame, title: str):
    """Markers plot for λ+ (stars) and λ− (circles), 4 per generation."""
    plt.figure(figsize=(11, 5))
    cmap = plt.cm.coolwarm_r
    colors = [cmap(i/(NT-1)) for i in range(NT)] if NT > 1 else ["red"]
    x = lam_plus_wide.index.values
    for p in range(1, NT+1):
        y_plus  = lam_plus_wide.get(f"λ+_piece{p}",  pd.Series(index=x, dtype=float)).to_numpy()
        y_minus = lam_minus_wide.get(f"λ-_piece{p}", pd.Series(index=x, dtype=float)).to_numpy()
        m1, m2 = ~np.isnan(y_plus), ~np.isnan(y_minus)
        plt.plot(x[m1], y_plus[m1],  marker="*", linestyle="None", markersize=9,
                 label=f"piece {p} (λ⁺)", color=colors[p-1])
        plt.plot(x[m2], y_minus[m2], marker="o", linestyle="None", markersize=4, alpha=0.75,
                 label=f"piece {p} (λ⁻)", color=colors[p-1])
    plt.xlabel("Iteration (generation)")
    plt.ylabel("λ from Eq. (5)")
    plt.title(title)
    plt.legend(frameon=False, ncol=min(2*NT, 8))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_lambda_max_lines(max_wide: pd.DataFrame, title: str):
    """Lines plot for λ_max per piece with fixed colors."""
    plt.figure(figsize=(11, 5))
    piece_colors = {1: "red", 2: "green", 3: "yellow", 4: "blue"}
    x = max_wide.index.values
    for p in range(1, NT+1):
        y = max_wide.get(f"λmax_piece{p}", pd.Series(index=x, dtype=float)).to_numpy()
        m = ~np.isnan(y)
        plt.plot(x[m], y[m], linestyle="-", linewidth=1.8,
                 color=piece_colors.get(p, "black"), label=f"piece {p}")
    plt.xlabel("Iteration (generation)")
    plt.ylabel(r"$\lambda_{\max}$ (Eq. 5)")
    plt.title(title)
    plt.legend(frameon=False, ncol=min(NT, 4))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Load tables
# -----------------------------
df_alpha = pd.read_csv(alpha_path)
df_beta  = pd.read_csv(beta_path)
df_gamma = pd.read_csv(gamma_path)

G = len(df_alpha)
gens = range(G)

df_idx = load_with_gen_index(best_idx_path, expected_len=G)
df_fl  = load_with_gen_index(fitness_fl_path, expected_len=G)

if "best_individual" not in df_idx.columns:
    raise KeyError("best_individuals_indices.csv must contain 'best_individual' (0-based).")

# pick fitness column in fitness_fL.csv
fl_col = "fitness" if "fitness" in df_fl.columns else next(
    (c for c in df_fl.columns if pd.api.types.is_numeric_dtype(df_fl[c])), None)
if fl_col is None:
    raise KeyError("No numeric fitness column found in fitness_fL.csv")

# -----------------------------
# Build best-so-far (plateau) mapping from fitness_fL.csv
# -----------------------------
fl = df_fl[fl_col].astype(float).to_numpy()
best_gen = np.empty(G, dtype=int)   # for each g: last generation g* with strict improvement
best_val = np.inf
best_at  = 0
for g in gens:
    if fl[g] < best_val - 0.0:      # strict improvement; tune tolerance if needed
        best_val = fl[g]
        best_at  = g
    best_gen[g] = best_at

best_ind_per_gen = df_idx["best_individual"].astype(int).to_numpy()
leader_ind_0b = best_ind_per_gen[best_gen]   # identity (0-based) chosen at last improvement g*

# -----------------------------
# Plot 1: per-generation best (current row)
# -----------------------------
rows_plus_best, rows_minus_best = [], []
for gen in gens:
    ind0 = int(best_ind_per_gen[gen])
    cols = cols_for_individual(ind0 + 1, df_alpha)

    a = df_alpha.loc[gen, cols].to_numpy(float)
    b = df_beta .loc[gen, cols].to_numpy(float)
    g = df_gamma.loc[gen, cols].to_numpy(float)

    lplus, lminus = lambdas_eq5(a, b, g, nu=NU)
    for p in range(NT):
        rows_plus_best.append( {"generation": gen, "piece": p+1, "lambda_plus":  lplus[p]})
        rows_minus_best.append({"generation": gen, "piece": p+1, "lambda_minus": lminus[p]})

best_plus_wide  = to_wide(rows_plus_best,  "lambda_plus",  "λ+")
best_minus_wide = to_wide(rows_minus_best, "lambda_minus", "λ-")

# -----------------------------
# Plot 2: best-so-far (α,β,γ frozen at last improvement g*)
# -----------------------------
rows_plus_lead, rows_minus_lead = [], []
for gen in gens:
    g_star = int(best_gen[gen])         # last improvement generation
    ind0   = int(leader_ind_0b[gen])    # leader identity chosen at g*
    cols   = cols_for_individual(ind0 + 1, df_alpha)

    a = df_alpha.loc[g_star, cols].to_numpy(float)  # FROZEN at g*
    b = df_beta .loc[g_star, cols].to_numpy(float)
    g = df_gamma.loc[g_star, cols].to_numpy(float)

    lplus, lminus = lambdas_eq5(a, b, g, nu=NU)
    for p in range(NT):
        rows_plus_lead.append( {"generation": gen, "piece": p+1, "lambda_plus":  lplus[p]})
        rows_minus_lead.append({"generation": gen, "piece": p+1, "lambda_minus": lminus[p]})

leader_plus_wide  = to_wide(rows_plus_lead,  "lambda_plus",  "λ+")
leader_minus_wide = to_wide(rows_minus_lead, "lambda_minus", "λ-")

# Build λ_max for the second plot (elementwise nan-safe max of λ+ and λ−)
leader_max_wide = leader_plus_wide.copy()
leader_max_wide.iloc[:, :] = np.fmax(
    leader_plus_wide.to_numpy(),
    leader_minus_wide.to_numpy()
)
leader_max_wide.columns = [c.replace("λ+", "λmax") for c in leader_max_wide.columns]

# -----------------------------
# Make the plots
# -----------------------------
# Plot 1: per-generation best, λ+ and λ− (markers)
plot_lambda_pairs(best_plus_wide, best_minus_wide,
                  r"Quadratic roots $\lambda^{(\pm)}$ (Eq. 5) — per-generation best individual")

# Plot 2: best-so-far, λ_max only (continuous lines), colors: red/green/yellow/blue
plot_lambda_max_lines(leader_max_wide,
                      r"$\lambda_{\max}$ (Eq. 5) — best-so-far (frozen at last improvement)")
