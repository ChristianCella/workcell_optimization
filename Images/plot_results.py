#!/usr/bin/env python3
import csv, os
import numpy as np

base_dir = os.path.abspath(os.path.dirname(__file__))
CSV_PATH  = os.path.join(base_dir, "simulation_metrics_results.csv")  # Update this path to your actual CSV file

def read_csv_rows(csv_path):
    rows = []
    with open(csv_path, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float(row, key):
    return float(row[key])


def safe_ratio(ref_value, test_value):
    if abs(ref_value) < 1e-12:
        raise ZeroDivisionError(f"Reference value is zero for ratio computation.")
    return (ref_value - test_value) / ref_value


def main():
    rows = read_csv_rows(CSV_PATH)

    if len(rows) < 21:
        raise ValueError(f"Expected at least 21 rows in the CSV, found {len(rows)}.")

    # Test 1 in your request corresponds to the first row of the CSV
    ref_row = rows[0]

    tau_tot_norm_ref = to_float(ref_row, "tau_tot_norm")
    tau_tot_4_ref = to_float(ref_row, "tau_tot_4")
    tau_tot_5_ref = to_float(ref_row, "tau_tot_5")
    tau_tot_6_ref = to_float(ref_row, "tau_tot_6")
    inv_manip_ref = to_float(ref_row, "inv_manip")
    delta_q_ref = to_float(ref_row, "delta_q")

    # Compare test1 against all the other 20 tests
    tau_tot_norm_ratios = []
    tau_tot_4_ratios = []
    tau_tot_5_ratios = []
    tau_tot_6_ratios = []
    inv_manip_ratios = []
    delta_q_ratios = []

    for row in rows[1:]:
        tau_tot_norm_j = to_float(row, "tau_tot_norm")
        tau_tot_4_j = to_float(row, "tau_tot_4")
        tau_tot_5_j = to_float(row, "tau_tot_5")
        tau_tot_6_j = to_float(row, "tau_tot_6")
        inv_manip_j = to_float(row, "inv_manip")
        delta_q_j = to_float(row, "delta_q")

        tau_tot_norm_ratios.append(safe_ratio(tau_tot_norm_ref, tau_tot_norm_j))
        tau_tot_4_ratios.append(safe_ratio(tau_tot_4_ref, tau_tot_4_j))
        tau_tot_5_ratios.append(safe_ratio(tau_tot_5_ref, tau_tot_5_j))
        tau_tot_6_ratios.append(safe_ratio(tau_tot_6_ref, tau_tot_6_j))
        inv_manip_ratios.append(safe_ratio(inv_manip_ref, inv_manip_j))
        delta_q_ratios.append(safe_ratio(delta_q_ref, delta_q_j))

    # Print vectors
    np.set_printoptions(precision=6, suppress=True)

    print("\n1) leader ratios:")
    print("   (tau_tot_norm_test1 - tau_tot_norm_testj) / tau_tot_norm_test1")
    print(np.array(tau_tot_norm_ratios))

    print("\n2) Focus on joints 4, 5, 6:")
    print("   (tau_tot_4_test1 - tau_tot_4_testj) / tau_tot_4_test1")
    print("tau_tot_4 ratios:")
    print(np.array(tau_tot_4_ratios))

    print("\n   (tau_tot_5_test1 - tau_tot_5_testj) / tau_tot_5_test1")
    print("tau_tot_5 ratios:")
    print(np.array(tau_tot_5_ratios))

    print("\n   (tau_tot_6_test1 - tau_tot_6_testj) / tau_tot_6_test1")
    print("tau_tot_6 ratios:")
    print(np.array(tau_tot_6_ratios))

    print("\n3) Focus on inverse manipulability:")
    print("   (inv_manip_test1 - inv_manip_testj) / inv_manip_test1")
    print(np.array(inv_manip_ratios))

    print("\n4) Focus on joint position differences:")
    print("   (delta_q_test1 - delta_q_testj) / delta_q_test1")
    print(np.array(delta_q_ratios))


if __name__ == "__main__":
    main()