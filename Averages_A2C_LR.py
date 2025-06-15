import os
import numpy as np
import csv

def compute_averages_from_npz(directory):
    categories = {"embb_sla": [], "urllc_sla": []}
    for file_name in os.listdir(directory):
        if file_name.endswith(".npz"):
            file_path = os.path.join(directory, file_name)
            data = np.load(file_path)
            for category in categories.keys():
                if category in data.files:
                    categories[category].extend(data[category])
    for category in categories:
        categories[category] = np.mean(categories[category]) if categories[category] else 0.0
    return categories

if __name__ == "__main__":
    lr_values = [0.0005, 0.001, 0.0015]
    base_dir = "/home/w5/pydemo/mecon25/results/results_a2c_lr_{}"
    csv_file_path = "/home/w5/pydemo/mecon25/averages_a2c_lrs.csv"
    with open(csv_file_path, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Learning_Rate", "embb_sla_avg", "urllc_sla_avg"])
        for lr in lr_values:
            lr_str = str(lr).replace('.', '_')
            dir_path = base_dir.format(lr_str)
            if os.path.exists(dir_path):
                avgs = compute_averages_from_npz(dir_path)
                csv_writer.writerow([
                    lr,
                    f"{avgs['embb_sla']:.3f}",
                    f"{avgs['urllc_sla']:.3f}"
                ])
    print(f"Averages for all A2C learning rates written to {csv_file_path}")
