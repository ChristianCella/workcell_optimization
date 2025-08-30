import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Load the CSV files
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


turbo_ikflow_path = os.path.join(base_dir, "data/screwing/turbo_ikflow/fitness_fL.csv")
df_turbo_ikflow = pd.read_csv(turbo_ikflow_path)

cmaes_ikflow_path = os.path.join(base_dir, "data/screwing/cma_es_ikflow/fitness_fL.csv")
df_cmaes_ikflow = pd.read_csv(cmaes_ikflow_path)

random_path = os.path.join(base_dir, "data/screwing/random/fitness_fL.csv")
df_random = pd.read_csv(random_path)

# Plot the fitness values
plt.figure(figsize = (10, 5))
plt.tick_params(axis='both', which='major', labelsize=16)
plt.plot(df_turbo_ikflow['fitness'], label=r'$\texttt{TuRBO}$', linewidth=2)
plt.plot(df_cmaes_ikflow['fitness'], label=r'$\texttt{cma-es}$', linewidth=2)
plt.plot(df_random['fitness'], label=r'$\texttt{random}$', linewidth=2)
#plt.title(rf'$\textbf{{Fitness Trend}}$', fontsize=20)
plt.xlabel(r'Iteration $i$', fontsize = 16)
plt.ylabel(r'$f_L (\boldsymbol{\xi}, \mathbf{q}) = w_{\tau} f_{\tau}(\boldsymbol{\xi}, \mathbf{q}) + w^{\text{s}}_{\text{L}} f^{\text{s}}_{\text{L}}(\boldsymbol{\xi}, \mathbf{q})$', fontsize = 16)
plt.grid(True)
plt.legend(fontsize = 18)
plt.tight_layout()
plt.show()
