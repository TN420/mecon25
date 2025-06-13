import os
import numpy as np

def compute_averages_from_npz(directory):
    """
    Reads .npz files from a directory, computes the average for each category, and returns the results.
    :param directory: Path to the directory containing .npz files.
    :return: Dictionary with categories as keys and their average results as values.
    """
    categories = {"embb_sla": [], "urllc_sla": []}
    for file_name in os.listdir(directory):
        if file_name.endswith(".npz"):
            file_path = os.path.join(directory, file_name)
            data = np.load(file_path)
            for category in categories.keys():
                if category in data.files:
                    categories[category].extend(data[category])
    
    # Compute the average for each category
    for category in categories:
        categories[category] = np.mean(categories[category]) if categories[category] else 0.0
    return categories

if __name__ == "__main__":
    # Paths to the directories containing .npz files for each method
    directories = {
        "A2C": "/home/w5/pydemo/mecon25/results/results_a2c",
        "DQN": "/home/w5/pydemo/mecon25/results/results_dqn",
        "RDQN": "/home/w5/pydemo/mecon25/results/results_rdqn"
    }

    # Compute averages for each method and category
    all_averages = {}
    for method, path in directories.items():
        all_averages[method] = compute_averages_from_npz(path)

    # Output the average results as a table
    print("Average Results:")
    print(f"{'Method':<10} {'Category':<15} {'Average':<10}")
    print("-" * 40)
    for method, averages in all_averages.items():
        for category, avg in averages.items():
            print(f"{method:<10} {category:<15} {avg:.4f}")
