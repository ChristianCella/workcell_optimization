import pandas as pd
import numpy as np
import sys, os

# Load the CSV files
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
path_alpha = os.path.join(base_dir, "data/alpha_trend.csv")
path_beta = os.path.join(base_dir, "data/beta_trend.csv")
path_gamma = os.path.join(base_dir, "data/gamma_trend.csv")

# Load the CSV files
alpha_df = pd.read_csv(path_alpha)
beta_df = pd.read_csv(path_beta)
gamma_df = pd.read_csv(path_gamma)

# Merge them on generation & piece
merged = alpha_df.merge(beta_df, on=["generation", "piece"]) \
                 .merge(gamma_df, on=["generation", "piece"])

# Compute sqrt(alpha + beta + gamma) for each row
merged["sqrt_sum"] = np.sqrt(merged["alpha"] + merged["beta"] + merged["gamma"])

# Group by generation and compute the mean of sqrt_sum
indicator_df = merged.groupby("generation")["sqrt_sum"].mean().reset_index()

print(indicator_df) #! If everything goes fine, you obtain exactly the values inside best_fitness.csv
