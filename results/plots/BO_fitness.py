#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------

#csv_path = "/Users/alessandrocasciani/Desktop//TO/final_dataset_200_EI_2.csv"
csv_path = "/Users/alessandrocasciani/Desktop/PYTHON_MUJOCO/robotic_contact_operations/workcell_optimization/results/data/fitness_fL.csv"
# Path al CSV con le variabili corrispondenti al best (ordine: x_b,y_b,theta_x_b,x_t,y_t,theta_x_t,x_p,y_p,q01,q02,q03,q04,q05,q06)
csv_path_vars = "/Users/alessandrocasciani/Desktop/PYTHON_MUJOCO/robotic_contact_operations/workcell_optimization/results/data/best_solutions.csv"

# Nomi delle colonne per le variabili (senza header nel CSV)
VAR_NAMES = [
    "x_b","y_b","theta_x_b",
    "x_t","y_t","theta_x_t",
    "x_p","y_p",
    "q01","q02","q03","q04","q05","q06"
]
# ------------------------------------------------------------------------


df = pd.read_csv(csv_path)
df_vars = pd.read_csv(csv_path_vars, header=None, names=VAR_NAMES)
# Normalizza ed assicura tipi numerici (gestisce anche eventuali virgole come separatore decimale)
df_vars = df_vars.replace(',', '.', regex=True)
for c in df_vars.columns:
    df_vars[c] = pd.to_numeric(df_vars[c], errors='coerce')
# Rimuovi eventuali righe totalmente non numeriche (es. header)
df_vars = df_vars.dropna(how='all').reset_index(drop=True)
#f_vars = pd.read_csv(csv_path_vars)
#_____________________________________________ Calcolo x1 e x2 dei best so far_____________________

'''

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
'''

# ────────────────────────── 1) y vs iterazione ───────────────────────────
plt.figure(figsize=(12, 4))
#plt.plot(df_plot.index, df_plot["y"])
#plt.plot(df.index, df["y"], label="y", color='blue')
plt.plot(df.index, df["fitness"], label="y", color='blue', marker='o', linestyle='-')
# Aggiungi linea rossa con minimo cumulativo
cummin_y = df["fitness"].cummin()
plt.plot(df.index, cummin_y, color='red', linewidth=3, label="Minimo cumulativo")
plt.xlabel("Iterazione")
plt.ylabel("y")
plt.title("y vs iterazione ")
plt.grid(True)
plt.legend()
'''

# ────────────────────────── 0) xb, yb, zb vs iterazione ───────────────────────────
plt.figure(figsize=(12, 4))
plt.plot(df_vars["iteration"], df_vars["xb"], marker='o', linestyle='-', label="xb")
plt.plot(df_vars["iteration"], df_vars["yb"], marker='o', linestyle='-', label="yb")
plt.plot(df_vars["iteration"], df_vars["zb"], marker='o', linestyle='-', label="zb")
plt.xlabel("Iterazione")
plt.ylabel("Valore")
plt.title("xb, yb, zb vs iterazione")
plt.legend()
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

plt.figure(figsize=(12, 4))
plt.plot(df_vars["iteration"], df_vars["theta_xb"], marker='o', linestyle='-', label="theta_xb")
plt.plot(df_vars["iteration"], df_vars["theta_yb"], marker='o', linestyle='-', label="theta_yb")
plt.plot(df_vars["iteration"], df_vars["theta_zb"], marker='o', linestyle='-', label="theta_zb")
plt.xlabel("Iterazione")
plt.ylabel("Valore")
plt.title("theta_xb, theta_yb, theta_zb vs iterazione")
plt.legend()
plt.grid(True)


plt.figure(figsize=(12, 4))
plt.plot(df_vars["iteration"], df_vars["xp"], marker='o', linestyle='-', label="xp")
plt.plot(df_vars["iteration"], df_vars["yp"], marker='o', linestyle='-', label="yp")
plt.plot(df_vars["iteration"], df_vars["zp"], marker='o', linestyle='-', label="zp")
plt.xlabel("Iterazione")
plt.ylabel("Valore")
plt.title("xp, yp, zp vs iterazione")
plt.legend()
plt.grid(True)

plt.figure(figsize=(12, 4))
plt.plot(df_vars["iteration"], df_vars["theta_xp"], marker='o', linestyle='-', label="theta_xp")
plt.plot(df_vars["iteration"], df_vars["theta_yp"], marker='o', linestyle='-', label="theta_yp")
plt.plot(df_vars["iteration"], df_vars["theta_zp"], marker='o', linestyle='-', label="theta_zp")
plt.xlabel("Iterazione")
plt.ylabel("Valore")
plt.title("theta_xp, theta_yp, theta_zp vs iterazione")
plt.legend()
plt.grid(True)

plt.figure(figsize=(12, 4))
plt.plot(df_vars["iteration"], df_vars["xe"], marker='o', linestyle='-', label="xe")
plt.plot(df_vars["iteration"], df_vars["ye"], marker='o', linestyle='-', label="ye")
plt.plot(df_vars["iteration"], df_vars["ze"], marker='o', linestyle='-', label="ze")
plt.xlabel("Iterazione")
plt.ylabel("Valore")
plt.title("xe, ye, ze vs iterazione")
plt.legend()
plt.grid(True)

plt.figure(figsize=(12, 4))
plt.plot(df_vars["iteration"], df_vars["theta_xe"], marker='o', linestyle='-', label="theta_xe")
plt.plot(df_vars["iteration"], df_vars["theta_ye"], marker='o', linestyle='-', label="theta_ye")
plt.plot(df_vars["iteration"], df_vars["theta_ze"], marker='o', linestyle='-', label="theta_ze")
plt.xlabel("Iterazione")
plt.ylabel("Valore")
plt.title("theta_xe, theta_ye, theta_ze vs iterazione")
plt.legend()
plt.grid(True)


# ────────────────────────── Immagine con minimo ───────────────────────────
# Trova il valore minimo e la riga corrispondente
min_idx = df["best_fitness"].idxmin()
min_val = df.loc[min_idx, "best_fitness"]
iter_min = df_vars.loc[min_idx, "iteration"]

# Prendi i valori delle variabili corrispondenti
vars_at_min = df_vars.loc[df_vars["iteration"] == iter_min].iloc[0]

# Crea messaggio
msg = f"Il minimo è {min_val:.4f} alla iterazione {iter_min}\n\n"
msg += f"xb = {vars_at_min['xb']:.4f}, yb = {vars_at_min['yb']:.4f}, zb = {vars_at_min['zb']:.4f}\n"
msg += f"theta_xb = {vars_at_min['theta_xb']:.4f}, theta_yb = {vars_at_min['theta_yb']:.4f}, theta_zb = {vars_at_min['theta_zb']:.4f}\n\n"
msg += f"xp = {vars_at_min['xp']:.4f}, yp = {vars_at_min['yp']:.4f}, zp = {vars_at_min['zp']:.4f}\n"
msg += f"theta_xp = {vars_at_min['theta_xp']:.4f}, theta_yp = {vars_at_min['theta_yp']:.4f}, theta_zp = {vars_at_min['theta_zp']:.4f}\n\n"
msg += f"xe = {vars_at_min['xe']:.4f}, ye = {vars_at_min['ye']:.4f}, ze = {vars_at_min['ze']:.4f}\n"
msg += f"theta_xe = {vars_at_min['theta_xe']:.4f}, theta_ye = {vars_at_min['theta_ye']:.4f}, theta_ze = {vars_at_min['theta_ze']:.4f}"
msg += f"\n\nè stato ottenuto con:\n max_evals=10.000\n n_init=25 \nbatch_size=32 \nn_trust_region=20 \nn_training_step=50"
msg += f"\n\n Ottimizzazione Completata in 2565.8min"
# Crea figura con solo il testo

plt.figure(figsize=(10, 6))
plt.text(0.01, 0.99, msg, fontsize=12, va="top", ha="left")
plt.axis("off")
plt.title("Risultato minimo", fontsize=14, weight="bold")





'''

# ────────────────────────── Minimo e variabili corrispondenti ───────────────────────────
# Scegli in modo robusto la colonna metrica disponibile
if "best_fitness" in df.columns:
    metric_col = "best_fitness"
elif "fitness" in df.columns:
    metric_col = "fitness"
elif "y" in df.columns:
    metric_col = "y"
else:
    raise KeyError("Nessuna colonna metrica trovata (attese: best_fitness, fitness, y)")

# Forza numerico la colonna metrica per sicurezza
df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce')

# Indice del minimo nella tabella fitness
min_idx = int(df[metric_col].idxmin())
min_val = float(df.loc[min_idx, metric_col])

# Allineamento per indice: df_vars ha le variabili per ciascuna iterazione nella stessa posizione
vars_at_min = df_vars.iloc[min_idx]

# Messaggio testuale con tutte le variabili nell'ordine richiesto
msg_lines = [
    f"Minimo: {min_val:.6f} (iterazione index={min_idx})",
    "\nVariabili corrispondenti:",
    f"x_b={vars_at_min['x_b']:.6f}, y_b={vars_at_min['y_b']:.6f}, theta_x_b={vars_at_min['theta_x_b']:.6f}",
    f"x_t={vars_at_min['x_t']:.6f}, y_t={vars_at_min['y_t']:.6f}, theta_x_t={vars_at_min['theta_x_t']:.6f}",
    f"x_p={vars_at_min['x_p']:.6f}, y_p={vars_at_min['y_p']:.6f}",
    f"q01={vars_at_min['q01']:.6f}, q02={vars_at_min['q02']:.6f}, q03={vars_at_min['q03']:.6f}",
    f"q04={vars_at_min['q04']:.6f}, q05={vars_at_min['q05']:.6f}, q06={vars_at_min['q06']:.6f}"
]
msg = "\n".join(msg_lines)

# Stampa su console
print("\n" + "-"*80)
print(msg)
print("-"*80 + "\n")

# Figura con il riepilogo del minimo e delle variabili
plt.figure(figsize=(10, 6))
plt.text(0.01, 0.99, msg, fontsize=12, va="top", ha="left")
plt.axis("off")
plt.title("Risultato minimo & variabili corrispondenti", fontsize=14, weight="bold")



# Mostra le due finestre
plt.show()
