import os
import numpy as np
import csv

def get_gamma_dirs(results_root):
    gamma_dirs = []
    for entry in os.listdir(results_root):
        if entry.startswith("results_dqn_gamma_"):
            gamma_dirs.append(os.path.join(results_root, entry))
    return gamma_dirs

def extract_gamma_from_dir(dirname):
    # expects: results_dqn_gamma_0_9, results_dqn_gamma_0_95, etc.
    parts = dirname.split("_")
    gamma_str = parts[-1]
    try:
        gamma = float(gamma_str.replace("_", "."))
    except ValueError:
        gamma = None
    return gamma

def compute_npz_averages(npz_dir):
    metrics = {"rewards": [], "urllc_blocks": [], "embb_blocks": [], "urllc_sla": [], "embb_sla": []}
    for fname in os.listdir(npz_dir):
        if fname.endswith(".npz"):
            data = np.load(os.path.join(npz_dir, fname))
            for key in metrics:
                if key in data.files:
                    metrics[key].extend(data[key])
    # Compute averages
    for key in metrics:
        arr = np.array(metrics[key])
        metrics[key] = np.mean(arr) if arr.size > 0 else np.nan
    return metrics

def main():
    results_root = "/home/w5/pydemo/mecon25/results"
    out_csv = "/home/w5/pydemo/mecon25/averages_dqn_gamma.csv"
    gamma_dirs = get_gamma_dirs(results_root)
    rows = []
    for gamma_dir in gamma_dirs:
        gamma = extract_gamma_from_dir(gamma_dir)
        if gamma is None:
            continue
        averages = compute_npz_averages(gamma_dir)
        row = [
            round(gamma, 3) if gamma is not None else "",
            round(averages["rewards"], 3) if not np.isnan(averages["rewards"]) else "",
            round(averages["urllc_blocks"], 3) if not np.isnan(averages["urllc_blocks"]) else "",
            round(averages["embb_blocks"], 3) if not np.isnan(averages["embb_blocks"]) else "",
            round(averages["urllc_sla"], 3) if not np.isnan(averages["urllc_sla"]) else "",
            round(averages["embb_sla"], 3) if not np.isnan(averages["embb_sla"]) else ""
        ]
        rows.append(row)
    # Sort by gamma
    rows.sort(key=lambda x: x[0])
    # Write to CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gamma", "avg_reward", "avg_urllc_blocks", "avg_embb_blocks", "avg_urllc_sla", "avg_embb_sla"])
        writer.writerows(rows)
    print(f"Wrote averages for DQN gammas to {out_csv}")

if __name__ == "__main__":
    main()
