import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# --- load dirs ---
cma_es_dir        = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/screwing/cma_es_ikflow'))
turbo_dir         = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/screwing/turbo_ikflow'))
turbo_bioik2_dir  = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/screwing/turbo_bioik2'))
random_dir        = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/screwing/random'))

# helper to read one value (convert to hours)
def read_hours(csv_path):
    s = pd.read_csv(csv_path)["total_time"].astype(str).str.replace(",", "", regex=False)
    val = pd.to_numeric(s, errors="coerce").iloc[0]
    return float(val) / 3600.0

# values
turbo_val       = read_hours(os.path.join(turbo_dir, "total_time.csv"))
cma_es_val      = read_hours(os.path.join(cma_es_dir, "total_time.csv"))
turbo_bioik2_val= read_hours(os.path.join(turbo_bioik2_dir, "total_time.csv"))
random_val      = read_hours(os.path.join(random_dir, "total_time.csv"))

print(f"The optimization with TuRBO_ikf took {turbo_val:.2f} hours.")
print(f"The optimization with cma-es_ikf took {cma_es_val:.2f} hours.")

# labels
labels = [
    r"$\texttt{TuRBO}_{\texttt{ikf}}$",
    r"$\texttt{cma-es}_{\texttt{ikf}}$",
    r"$\texttt{TuRBO}_{\texttt{bio}}$",
    r"$\texttt{random}$",
]
values = [turbo_val, cma_es_val, turbo_bioik2_val, random_val]

# colors (mpl default: blue, orange, green, red)
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# --- bar chart ---
fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(labels, values, color=colors)

ax.set_ylabel(r"Time [hours]", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=20)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
