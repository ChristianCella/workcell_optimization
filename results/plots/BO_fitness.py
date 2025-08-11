#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------

#csv_path = "/Users/alessandrocasciani/Desktop//TO/final_dataset_200_EI_2.csv"
csv_path = "/Users/alessandrocasciani/Desktop/PYTHON_MUJOCO/robotic_contact_operations/workcell_optimization/utils/datasets/final_dataset_201_UCB_2.csv"
csv_hist="/Users/alessandrocasciani/Desktop/PYTHON_MUJOCO/robotic_contact_operations/workcell_optimization/utils/datasets/history_201_UCB_2.csv"
csv_EI= "/Users/alessandrocasciani/Desktop/PYTHON_MUJOCO/robotic_contact_operations/workcell_optimization/utils/datasets/UCB_201_UCB_2.csv"
# ------------------------------------------------------------------------

df_ei=pd.read_csv(csv_EI)


plt.figure(figsize=(12, 4))
plt.plot(df_ei.index, df_ei["UCB"], label="y", color='blue', marker='o', linestyle='-')
plt.xlabel("Iterazione")
plt.ylabel("UCB")
plt.title("UCB vs iterazione ")
plt.grid(True)


df = pd.read_csv(csv_path)
df_hist = pd.read_csv(csv_hist)
df_real = df[df["y"] <= 1000]

START_ITER = 300
df = df[df.index >= START_ITER].reset_index(drop=True)

df_plot = df[df["y"] <= 1000]

#df_real=df = pd.read_csv(csv_path)


#_____________________________________________ Calcolo x1 e x2 dei best so far_____________________



best_x1_list, best_x2_list = [], []
best_val = np.inf
current_best_x1, current_best_x2 = None, None

for _, row in df.iterrows():
    if row["y"] < best_val:
        best_val = row["y"]
        current_best_x1 = row["x1"]
        current_best_x2 = row["x2"]
    best_x1_list.append(current_best_x1)
    best_x2_list.append(current_best_x2)

best_x_df = pd.DataFrame({
    "best_x1": best_x1_list,
    "best_x2": best_x2_list
})
# -------------------------------------------------------------------------


# ────────────────────────── 1) y vs iterazione ───────────────────────────
plt.figure(figsize=(12, 4))
#plt.plot(df_plot.index, df_plot["y"])
#plt.plot(df.index, df["y"], label="y", color='blue')
plt.plot(df_real.index, df_real["y"], label="y", color='blue', marker='o', linestyle='-')
plt.xlabel("Iterazione")
plt.ylabel("y")
plt.title("y vs iterazione ")
plt.grid(True)


# ───────────────────────── 2) x1, x2 vs iterazione ───────────────────────
plt.figure(figsize=(12, 4))
plt.plot(df_real.index, df_real["x1"], label="x1", marker='o', linestyle='-')
plt.plot(df_real.index, df_real["x2"], label="x2", marker='o', linestyle='-')
plt.xlabel("Iterazione")
plt.ylabel("valore")
plt.title("x1 e x2 vs iterazione")
plt.legend()
plt.grid(True)
# ────────────────────────── 3) best_so_far vs iterazione ───────────────────────────
plt.figure(figsize=(12, 4))
plt.plot(df_hist.index, df_hist["best_so_far"], label="BSF", color='orange', marker='o', linestyle='-')
plt.xlabel("Iterazione")
plt.ylabel("Tau_BSF")
plt.title("Best So Far vs Iterazione")
plt.legend()
plt.grid(True)


# ───────────────────────── 4) x1,x2 dei BEST SO FAR ──────────────────────
plt.figure(figsize=(12, 4))
plt.plot(df.index, best_x_df["best_x1"], label="best_x1", marker='o', linestyle='-')
plt.plot(df.index, best_x_df["best_x2"], label="best_x2", marker='o', linestyle='-')
plt.xlabel("Iterazione")
plt.ylabel("valore (m)")
plt.title("x1 e x2 dei BEST SO FAR")
plt.legend()
plt.grid(True)


# Mostra le due finestre
plt.show()
