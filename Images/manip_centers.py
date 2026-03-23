#!/usr/bin/env python3
import csv, os
import numpy as np

base_dir = os.path.abspath(os.path.dirname(__file__))
CSV_FILE_1  = os.path.join(base_dir, "simulation_metrics_results.csv")  # Update this path to your actual CSV file
CSV_FILE_2  = os.path.join(base_dir, "manip_results.csv")  # Update this path to your actual CSV file

def read_csv(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def main():
    rows1 = read_csv(CSV_FILE_1)
    rows2 = read_csv(CSV_FILE_2)

    # From file 1:
    # - skip the first row ("optimal")
    # - use inv_manip and delta_q from rows 2..21 => 20 values
    inv_manip_vec = np.array([float(row["inv_manip"]) for row in rows1[1:]], dtype=float)
    delta_q_vec = np.array([float(row["delta_q"]) for row in rows1[1:]], dtype=float)

    # From file 2:
    # - use manip and centering from all 20 rows
    manip_opt_vec = np.array([float(row["manip"]) for row in rows2], dtype=float)
    centering_opt_vec = np.array([float(row["centering"]) for row in rows2], dtype=float)

    if len(inv_manip_vec) != 20:
        raise ValueError(f"Expected 20 inv_manip values from file 1 after skipping first row, got {len(inv_manip_vec)}")

    if len(delta_q_vec) != 20:
        raise ValueError(f"Expected 20 delta_q values from file 1 after skipping first row, got {len(delta_q_vec)}")

    if len(manip_opt_vec) != 20:
        raise ValueError(f"Expected 20 manip values from file 2, got {len(manip_opt_vec)}")

    if len(centering_opt_vec) != 20:
        raise ValueError(f"Expected 20 centering values from file 2, got {len(centering_opt_vec)}")

    # 1) ratios: (manip_opt - inv_manip) / manip_opt
    manip_ratio_vec = (manip_opt_vec - inv_manip_vec) / manip_opt_vec

    # 2) same ratio, but with centerings:
    #    (centering_opt - delta_q) / centering_opt
    centering_ratio_vec = (centering_opt_vec - delta_q_vec) / centering_opt_vec

    np.set_printoptions(precision=6, suppress=True)

    print("\nVector 1: (manip_opt - inv_manip) / manip_opt")
    print(manip_ratio_vec)

    print("\nVector 2: (centering_opt - delta_q) / centering_opt")
    print(centering_ratio_vec)


if __name__ == "__main__":
    main()