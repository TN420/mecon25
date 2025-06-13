import os
import numpy as np
import csv
import pandas as pd  # Add pandas for easier CSV handling

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

def compute_additional_params_from_npz(directory):
    """
    Reads .npz files from a directory and computes the average for additional parameters.
    :param directory: Path to the directory containing .npz files.
    :return: Dictionary with additional parameters as keys and their average results as values.
    """
    params = {}  # Remove unnecessary fields
    for file_name in os.listdir(directory):
        if file_name.endswith(".npz"):
            file_path = os.path.join(directory, file_name)
            data = np.load(file_path)
            for param in params.keys():
                if param in data.files:
                    params[param].append(data[param].item())  # Assuming scalar values in .npz

    # Compute the average for each parameter
    for param in params:
        params[param] = np.mean(params[param]) if params[param] else 0.0
    return params

def load_training_data(csv_file_path):
    """
    Loads training data from a CSV file and computes the average for relevant columns.
    :param csv_file_path: Path to the CSV file.
    :return: Dictionary with column names as keys and their average values as values.
    """
    df = pd.read_csv(csv_file_path)
    averages = df.mean(numeric_only=True).to_dict()  # Compute averages for numeric columns
    return averages

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

    # Prepare a flat dictionary for the six columns
    flat_averages = {
        f"{method}_{category}": averages.get(category, 0.0)
        for method, averages in all_averages.items()
        for category in ["embb_sla", "urllc_sla"]
    }

    # Output the average results as a table
    print("Average Results:")
    print(f"{'A2C_embb_sla':<15} {'A2C_urllc_sla':<15} {'DQN_embb_sla':<15} {'DQN_urllc_sla':<15} {'RDQN_embb_sla':<15} {'RDQN_urllc_sla':<15}")
    print("-" * 90)
    print(f"{flat_averages['A2C_embb_sla']:<15.4f} {flat_averages['A2C_urllc_sla']:<15.4f} {flat_averages['DQN_embb_sla']:<15.4f} {flat_averages['DQN_urllc_sla']:<15.4f} {flat_averages['RDQN_embb_sla']:<15.4f} {flat_averages['RDQN_urllc_sla']:<15.4f}")

    # Write the results to a CSV file (append mode)
    csv_file_path = "/home/w5/pydemo/mecon25/averages.csv"
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, mode="a", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header only if the file does not already exist
        if not file_exists:
            csv_writer.writerow(["A2C_embb_sla", "A2C_urllc_sla", "DQN_embb_sla", "DQN_urllc_sla", "RDQN_embb_sla", "RDQN_urllc_sla"])
        csv_writer.writerow([
            flat_averages["A2C_embb_sla"],
            flat_averages["A2C_urllc_sla"],
            flat_averages["DQN_embb_sla"],
            flat_averages["DQN_urllc_sla"],
            flat_averages["RDQN_embb_sla"],
            flat_averages["RDQN_urllc_sla"]
        ])
    print(f"Results have been appended to {csv_file_path}")

    # Compute averages for additional parameters from RDQN directory
    additional_params = compute_additional_params_from_npz(directories["RDQN"])

    # Append additional parameters to the CSV file
    with open(csv_file_path, mode="a", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        for param, value in additional_params.items():
            csv_writer.writerow([param, value])
    print("Hyperparameters have been appended to the CSV file.")

    # Paths to the training time CSV files
    training_csv_files = {
        "A2C": "/home/w5/pydemo/mecon25/training_times_a2c.csv",
        "DQN": "/home/w5/pydemo/mecon25/training_times_dqn.csv",
        "RDQN": "/home/w5/pydemo/mecon25/training_times_rdqn.csv"
    }

    # Load and compute averages from training time CSV files
    training_averages = {}
    for method, csv_path in training_csv_files.items():
        training_averages[method] = load_training_data(csv_path)

    # Append training data averages to the CSV file
    with open(csv_file_path, mode="a", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write training data averages for each method
        for method, averages in training_averages.items():
            for param, value in averages.items():
                csv_writer.writerow([f"{method}_{param}", value])
    print("Training data averages have been appended to the CSV file.")
    with open(csv_file_path, mode="a", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["-", "-", "-", "-", "-", "-"])  # Demarcation line
