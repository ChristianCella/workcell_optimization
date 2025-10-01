import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# ---- Parameters ----
sim_dt = 0.005   # same as model.opt.timestep
q_desired = np.radians(100)

# Load the CSV file
pos_array = np.loadtxt("positions.csv", delimiter=",")

# Build the time vector
time = np.arange(len(pos_array)) * sim_dt

print("Shape of data:", pos_array.shape)

# Plot measured response
plt.plot(time, pos_array, label=r"System response (II order)")

# Plot the step input (reference)
plt.plot(time, np.ones_like(time) * q_desired,
         "r--", linewidth=2, label=r"$(100 \cdot \pi / 180) \cdot$ Step input")

plt.xlabel(r"Time (s)", fontsize=15)
plt.ylabel(r"$q_1$ (rad)", fontsize=15)
plt.title(r"\textbf{Second order-like behavior}", fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend()
plt.grid(True)
plt.show()
