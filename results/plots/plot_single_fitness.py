import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Load the CSV file
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dataset_csv_path = os.path.join(base_dir, "data/screwing/random/fitness_fL.csv")
df = pd.read_csv(dataset_csv_path) 

# Plot the fitness values
plt.figure(figsize = (8, 5))
plt.plot(df['fitness'], marker='o')
plt.title(rf'$\textbf{{Fitness Trend}}$', fontsize=20)
plt.xlabel('Iteration', fontsize = 15)
plt.ylabel(r'$\|\tau\|_2$', fontsize = 15)
plt.grid(True)
plt.tight_layout()
plt.show()
