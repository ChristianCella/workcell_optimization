import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# --- load ---
cma_es_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/screwing/cma_es_ikflow'))
turbo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/screwing/turbo_ikflow'))
turbo_bioik2_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/screwing/turbo_bioik2'))
random_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/screwing/random'))

# Create dataframes 
cma_es_df_alfa  = pd.read_csv(os.path.join(cma_es_dir, "complete_alpha_trend_wide.csv"))
cma_es_df_beta  = pd.read_csv(os.path.join(cma_es_dir, "complete_beta_trend_wide.csv"))
cma_es_df_gamma = pd.read_csv(os.path.join(cma_es_dir, "complete_gamma_trend_wide.csv"))
cma_es_df_best  = pd.read_csv(os.path.join(cma_es_dir, "best_individuals_indices.csv"))

turbo_df_alfa  = pd.read_csv(os.path.join(turbo_dir, "complete_alpha_trend_wide.csv"))
turbo_df_beta  = pd.read_csv(os.path.join(turbo_dir, "complete_beta_trend_wide.csv"))
turbo_df_gamma = pd.read_csv(os.path.join(turbo_dir, "complete_gamma_trend_wide.csv"))
turbo_df_best  = pd.read_csv(os.path.join(turbo_dir, "best_individuals_indices.csv"))

turbo_bioik2_df_alfa  = pd.read_csv(os.path.join(turbo_bioik2_dir, "complete_alpha_trend_wide.csv"))
turbo_bioik2_df_beta  = pd.read_csv(os.path.join(turbo_bioik2_dir, "complete_beta_trend_wide.csv"))
turbo_bioik2_df_gamma = pd.read_csv(os.path.join(turbo_bioik2_dir, "complete_gamma_trend_wide.csv"))
turbo_bioik2_df_best  = pd.read_csv(os.path.join(turbo_bioik2_dir, "best_individuals_indices.csv"))

random_df_alfa  = pd.read_csv(os.path.join(random_dir, "complete_alpha_trend_wide.csv"))
random_df_beta  = pd.read_csv(os.path.join(random_dir, "complete_beta_trend_wide.csv"))
random_df_gamma = pd.read_csv(os.path.join(random_dir, "complete_gamma_trend_wide.csv"))
random_df_best  = pd.read_csv(os.path.join(random_dir, "best_individuals_indices.csv"))

# Ensure integer indices
cma_es_df_best["best_individual"] = cma_es_df_best["best_individual"].astype(int)
turbo_df_best["best_individual"] = turbo_df_best["best_individual"].astype(int)
turbo_bioik2_df_best["best_individual"] = turbo_bioik2_df_best["best_individual"].astype(int)
random_df_best["best_individual"] = random_df_best["best_individual"].astype(int)

# --- detect whether best_individual is 0-based or 1-based ---
def piece_numbers(df):
    nums = set()
    for c in df.columns:
        if c.startswith("piece") and "_joint" in c:
            try:
                nums.add(int(c.split("_")[0].replace("piece","")))
            except ValueError:
                pass
    return sorted(nums)

cma_es_pieces = piece_numbers(cma_es_df_alfa)
if not cma_es_pieces:
    raise ValueError("Could not detect piece columns like 'piece1_joint1' in the alpha CSV.")
min_piece = min(cma_es_pieces)

turbo_pieces = piece_numbers(turbo_df_alfa)
if not turbo_pieces:
    raise ValueError("Could not detect piece columns like 'piece1_joint1' in the alpha CSV.")
min_piece = min(turbo_pieces)

turbo_bioik2_pieces = piece_numbers(turbo_bioik2_df_alfa)
if not turbo_bioik2_pieces:
    raise ValueError("Could not detect piece columns like 'piece1_joint1' in the alpha CSV.")
min_piece = min(turbo_bioik2_pieces)

random_pieces = piece_numbers(random_df_alfa)
if not random_pieces:
    raise ValueError("Could not detect piece columns like 'piece1_joint1' in the alpha CSV.")
min_piece = min(random_pieces)

# If best_individual contains zeros and columns start at piece1, we need an offset of +1
cma_es_offset = 1 if (cma_es_df_best["best_individual"].min() == 0 and min_piece == 1) else 0
turbo_offset = 1 if (turbo_df_best["best_individual"].min() == 0 and min_piece == 1) else 0
turbo_bioik2_offset = 1 if (turbo_bioik2_df_best["best_individual"].min() == 0 and min_piece == 1) else 0
random_offset = 1 if (random_df_best["best_individual"].min() == 0 and min_piece == 1) else 0

# --- compute aggregated values per iteration ---
cma_es_n_iter = min(len(cma_es_df_best), len(cma_es_df_alfa), len(cma_es_df_beta), len(cma_es_df_gamma))
cma_es_vals = []

turbo_n_iter = min(len(turbo_df_best), len(turbo_df_alfa), len(turbo_df_beta), len(turbo_df_gamma))
turbo_vals = []

turbo_bioik2_n_iter = min(len(turbo_bioik2_df_best), len(turbo_bioik2_df_alfa), len(turbo_bioik2_df_beta), len(turbo_bioik2_df_gamma))
turbo_bioik2_vals = []

random_n_iter = min(len(random_df_best), len(random_df_alfa), len(random_df_beta), len(random_df_gamma))
random_vals = []

def compute_values(df_best, df_alfa, df_beta, df_gamma, offset, n_iter, vals):
    for i in range(n_iter):
        best_idx = int(df_best.loc[i, "best_individual"]) + offset  # map to column numbering
        per_joint = []
        for j in range(1, 5):
            a = df_alfa.loc[i,  f"piece{best_idx}_joint{j}"]
            b = df_beta.loc[i,  f"piece{best_idx}_joint{j}"]
            g = df_gamma.loc[i, f"piece{best_idx}_joint{j}"]
            per_joint.append(np.sqrt(a + b + g))
        vals.append(np.mean(per_joint))
    return vals

cma_es_vals = np.asarray(compute_values(cma_es_df_best, cma_es_df_alfa, cma_es_df_beta, cma_es_df_gamma, cma_es_offset, cma_es_n_iter, cma_es_vals), dtype=float)
turbo_vals = np.asarray(compute_values(turbo_df_best, turbo_df_alfa, turbo_df_beta, turbo_df_gamma, turbo_offset, turbo_n_iter, turbo_vals), dtype=float)
turbo_bioik2_vals = np.asarray(compute_values(turbo_bioik2_df_best, turbo_bioik2_df_alfa, turbo_bioik2_df_beta, turbo_bioik2_df_gamma, turbo_bioik2_offset, turbo_bioik2_n_iter, turbo_bioik2_vals), dtype=float)
random_vals = np.asarray(compute_values(random_df_best, random_df_alfa, random_df_beta, random_df_gamma, random_offset, random_n_iter, random_vals), dtype=float)

# replace spikes (>10) with the previous value 
def replace_spikes_previous(arr, thresh=10.0):
    out = arr.copy().astype(float)
    last_valid = np.nan
    for i in range(len(out)):
        x = out[i]
        if not np.isfinite(x) or x > thresh:
            if np.isfinite(last_valid):
                out[i] = last_valid  # use previous valid value
        else:
            last_valid = x
    # If the first few entries were > thresh/NaN and thus stayed NaN,
    # backfill them with the first valid value so the length stays the same.
    if not np.isfinite(out[0]):
        idx = np.where(np.isfinite(out))[0]
        if idx.size > 0:
            out[:idx[0]] = out[idx[0]]
    return out

cma_es_vals = replace_spikes_previous(cma_es_vals, thresh=10.0)
turbo_vals = replace_spikes_previous(turbo_vals, thresh=10.0)
turbo_bioik2_vals = replace_spikes_previous(turbo_bioik2_vals, thresh=10.0)
random_vals = replace_spikes_previous(random_vals, thresh=10.0)

# --- Plot 1: Trend ---
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(turbo_vals)+1), turbo_vals, linewidth=2, label=r"\texttt{TuRBO-ikflow}")
plt.plot(range(1, len(cma_es_vals)+1), cma_es_vals, linewidth=2, label=r"\texttt{cma_es-ikflow}")
plt.plot(range(1, len(turbo_bioik2_vals)+1), turbo_bioik2_vals, linewidth=2, label=r"\texttt{TuRBO-bioik2}")
plt.plot(range(1, len(random_vals)+1), random_vals, linewidth=2, label=r"\texttt{random}")
plt.xlabel(r"Iteration $i$", fontsize=16)
plt.ylabel(r"$\bar{f}_{\tau} = \frac{1}{N_T}\sum_{j=1}^{N_T} \sqrt{\bar{\alpha}^*_j + \bar{\beta}^*_j + \bar{\gamma}^*_j}$", fontsize=16)
plt.title(r"\textbf{Best local individual}", fontsize=20)
plt.grid(True)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()

# --- Plot 2: Boxplot ---
labels = [
    r"$\texttt{TuRBO}_{\texttt{ikf}}$",
    r"$\texttt{cma-es}_{\texttt{ikf}}$",
    r"$\texttt{TuRBO}_{\texttt{bio}}$",
    r"$\texttt{random}$",
]

# Matplotlib's default cycle: blue, orange, green, red
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
fig, ax = plt.subplots(figsize=(8, 4))
bp = ax.boxplot([turbo_vals, cma_es_vals, turbo_bioik2_vals, random_vals], 
            vert=True, 
            patch_artist=True, 
            labels=labels,
            widths=0.6,
            medianprops=dict(linewidth=2),
            boxprops=dict(linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            flierprops=dict(markersize=4, markeredgewidth=0.8))

# Color each series consistently
for i, col in enumerate(colors):
    bp["boxes"][i].set(facecolor=col, edgecolor=col, alpha=0.75)
    for w in bp["whiskers"][2*i:2*i+2]:
        w.set(color=col)
    for c in bp["caps"][2*i:2*i+2]:
        c.set(color=col)
    bp["medians"][i].set(color=col, linewidth=2)
    if len(bp["fliers"]) > i:
        bp["fliers"][i].set(markerfacecolor=col, markeredgecolor=col, alpha=0.7)

plt.xticks(fontsize=16)
plt.ylabel(r"$\bar{f}_{\tau}$", fontsize=18)
#plt.title("Boxplot of Aggregated Values", fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()
