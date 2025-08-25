import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Load the CSV file
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dataset_csv_path = os.path.join(base_dir, "data/screwing/cma_es_ikflow/best_solutions.csv")
df = pd.read_csv(dataset_csv_path) 

# Plot the x and y base coordinates
plt.figure(figsize = (8, 5))
plt.plot(df['x_b'], marker='o')
plt.plot(df['y_b'], marker='o')
plt.title(rf'$\textbf{{Robot base position}}$', fontsize=20)
plt.xlabel('Iteration', fontsize = 15)
plt.ylabel(r'$x_{base}, y_{base}$', fontsize = 15)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot the theta_x coordinate
plt.figure(figsize = (8, 5))
plt.plot(np.degrees(df['theta_x_b']), marker='o')
plt.title(rf'$\textbf{{Robot base x orientation}}$', fontsize=20)
plt.xlabel('Iteration', fontsize = 15)
plt.ylabel(r'$\theta_{x}$', fontsize = 15)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot the robot configuration
plt.figure(figsize = (8, 5))
plt.plot(df['q01'], marker='o', label=r'$q_1$')
plt.plot(df['q02'], marker='o', label=r'$q_2$')
plt.plot(df['q03'], marker='o', label=r'$q_3$')
plt.plot(df['q04'], marker='o', label=r'$q_4$')
plt.plot(df['q05'], marker='o', label=r'$q_5$')
plt.plot(df['q06'], marker='o', label=r'$q_6$')
plt.title(rf'$\textbf{{Robot Configuration}}$', fontsize=20)
plt.xlabel('Iteration', fontsize = 15)
plt.ylabel(r'$q_{1}, q_{2}, q_{3}, q_{4}, q_{5}, q_{6}$', fontsize = 15)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()
