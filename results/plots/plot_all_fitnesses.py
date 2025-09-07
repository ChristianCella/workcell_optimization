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

turbo_bioik2_path = os.path.join(base_dir, "data/screwing/turbo_bioik2/fitness_fL.csv")
df_turbo_bioik2 = pd.read_csv(turbo_bioik2_path)

random_path = os.path.join(base_dir, "data/screwing/random/fitness_fL.csv")
df_random = pd.read_csv(random_path)

# Plot the fitness values
plt.figure(figsize = (10, 5))
plt.tick_params(axis='both', which='major', labelsize=16)
plt.plot(df_turbo_ikflow['fitness'], label=r'$\texttt{TuRBO}_{\texttt{ikf}}$', linewidth=2)
plt.plot(df_cmaes_ikflow['fitness'], label=r'$\texttt{cma-es}_{\texttt{ikf}}$', linewidth=2)
plt.plot(df_turbo_bioik2['fitness'], label=r'$\texttt{TuRBO}_{\texttt{bio}}$', linewidth=2)
plt.plot(df_random['fitness'], label=r'$\texttt{random}$', linewidth=2)
#plt.title(rf'$\textbf{{Fitness Trend}}$', fontsize=20)
plt.xlabel(r'Iteration $i$', fontsize = 16)
plt.ylabel(r'$\text{Fitness trend } \bar{f}_L$', fontsize = 16)
plt.grid(True)
plt.legend(fontsize = 16)
plt.tight_layout()
plt.show()
