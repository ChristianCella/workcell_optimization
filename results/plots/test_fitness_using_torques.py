import pandas as pd
import numpy as np
import sys, os

# Load the CSV files
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
path_ext = os.path.join(base_dir, "data/screwing/turbo_ikflow/best_external_torques.csv")
path_grav = os.path.join(base_dir, "data/screwing/turbo_ikflow/best_gravity_torques.csv")

df_ext = pd.read_csv(path_ext)
df_grav = pd.read_csv(path_grav)

# Number of pieces and joints (assumed from column names)
n_pieces = 4
n_joints = 6

# Check assumption matches the number of columns
expected_cols = n_pieces * n_joints
assert df_ext.shape[1] == expected_cols and df_grav.shape[1] == expected_cols, \
    "CSV column count mismatch — check number of pieces or joints."

# For each generation, compute the mean norm of (tau_g + tau_ext) across all pieces
for gen_idx in range(len(df_ext)):
    total_norms = []
    gear_ratios = [100, 100, 100, 100, 100, 100]
    max_torques = [1.50, 1.50, 1.50, 0.28, 0.28, 0.28]
    for piece_idx in range(n_pieces):
        # Extract joint torques for this piece
        ext = df_ext.iloc[gen_idx, piece_idx * n_joints : (piece_idx + 1) * n_joints].to_numpy()
        grav = df_grav.iloc[gen_idx, piece_idx * n_joints : (piece_idx + 1) * n_joints].to_numpy()
        
        tau_sum = (ext + grav) / (np.array(gear_ratios) * np.array(max_torques))
        print(f"the normalized torques are: {tau_sum}")
        norm = np.linalg.norm(tau_sum)  # 2-norm
        total_norms.append(norm)
    
    mean_norm = np.mean(total_norms)
    print(f"Generation {gen_idx}: Mean 2-norm of total torques = {mean_norm}")
