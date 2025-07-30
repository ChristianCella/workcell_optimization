import matplotlib    
import matplotlib.pyplot as plt
import sys, os
import pandas as pd
import numpy as np

matplotlib.rcParams['mathtext.fontset'] = 'cm' # 'cm' or 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rc('xtick', labelsize = 15)
matplotlib.rc('ytick', labelsize = 15)
matplotlib.rcParams['mathtext.fontset'] = 'cm' # 'cm' or 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rc('xtick', labelsize = 15)
matplotlib.rc('ytick', labelsize = 15)

# For relative imports
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_dir)

#! Import the data
dataset_dir = os.path.join(base_dir, "datasets")
dataset_files = [f for f in os.listdir(dataset_dir) if f.startswith("final_dataset_") and f.endswith(".csv")]

if not dataset_files:
    raise FileNotFoundError("No training dataset found in the datasets directory.") 

# Optional: sort to get the most recent one by timestamp in filename
dataset_files.sort(reverse=True)
dataset_csv_path = os.path.join(dataset_dir, dataset_files[0])

df = pd.read_csv(dataset_csv_path)
X_dataset = df[["x1", "x2"]].values
Y_dataset = df[["y"]].values

#! Plot 1 => Evaluated points
xs = X_dataset[:, 0]
ys = X_dataset[:, 1]
zs = Y_dataset.flatten()
best_x, best_y, best_z = xs[np.argmin(Y_dataset)], ys[np.argmin(Y_dataset)], zs[np.argmin(Y_dataset)]
fig = plt.figure(figsize=(8, 6))
ax  = fig.add_subplot(111, projection="3d")
ax.scatter(xs, ys, zs, c="blue", label = "Evaluated points") # All points in blue
ax.scatter(best_x, best_y, best_z, c="red", s=80, label="Best point") # Best point in red
ax.set_xlabel(r"$x$", fontsize=14)
ax.set_ylabel(r"$y$", fontsize=14)
ax.set_zlabel(r"$f(x,y)$", fontsize=14)
ax.legend()
plt.tight_layout()
plt.show()

#! Plot 2 => Evolution of the evaluations
# Plot the evaluations stored in the CSV file
plt.figure(figsize=(7, 4))
plt.plot(range(1, len(Y_dataset) + 1), Y_dataset.ravel(), marker='o', label='evaluations')

plt.xlabel('Iteration')
plt.ylabel(r"$\|\tau\|_2$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#! Plot 3 => Plot the x, y coordinates
# Plot the evaluations stored in the CSV file
plt.figure(figsize=(7, 4))
plt.plot(range(1, len(X_dataset) + 1), X_dataset[:, 0].ravel(), marker='o', label='x evaluations')
plt.plot(range(1, len(X_dataset) + 1), X_dataset[:, 1].ravel(), marker='o', label='y evaluations')
plt.xlabel('Iteration')
plt.ylabel(r"$\|\tau\|_2$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
