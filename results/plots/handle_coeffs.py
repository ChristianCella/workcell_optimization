import pandas as pd
import numpy as np
import os

# --- load ---
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
alpha_path = os.path.join(base_dir, "complete_alpha_trend_wide.csv")
beta_path  = os.path.join(base_dir, "complete_beta_trend_wide.csv")
gamma_path = os.path.join(base_dir, "complete_gamma_trend_wide.csv")
index_path = os.path.join(base_dir, "best_individuals_indices.csv")

df_alpha = pd.read_csv(alpha_path)
df_beta  = pd.read_csv(beta_path)
df_gamma = pd.read_csv(gamma_path)
df_idx   = pd.read_csv(index_path)

# infer joints per piece (columns like piece1_jointX)
n_joints = sum(c.startswith("piece1_joint") for c in df_alpha.columns)

# pick the best piece per generation
selected = []
for _, row in df_idx.iterrows():
    gen = int(row["generation"])
    best_piece = int(row["best_individual"]) + 1  # columns are 1-based
    cols = [f"piece{best_piece}_joint{j}" for j in range(1, n_joints + 1)]
    selected.append({
        "generation": gen,
        "alpha": df_alpha.loc[gen, cols].to_list(),
        "beta":  df_beta.loc[gen,  cols].to_list(),
        "gamma": df_gamma.loc[gen, cols].to_list()
    })

# tidy rows
records = []
for s in selected:
    for j, (a, b, g) in enumerate(zip(s["alpha"], s["beta"], s["gamma"]), start=1):
        records.append({"generation": s["generation"], "joint": j,
                        "alpha": a, "beta": b, "gamma": g})
merged = pd.DataFrame(records).sort_values("generation")

# per-generation indicator (mean over joints of sqrt(alpha+beta+gamma))
merged["sqrt_sum"] = np.sqrt(merged["alpha"] + merged["beta"] + merged["gamma"])
per_gen = merged.groupby("generation", as_index=False)["sqrt_sum"].mean()

# running best-so-far to match best_fitness_trend
per_gen["fitness_trend"] = per_gen["sqrt_sum"].cummin()

print(per_gen[["generation", "sqrt_sum", "fitness_trend"]])
