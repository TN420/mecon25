# Compare.py

import numpy as np
import matplotlib.pyplot as plt
import os

# ================================
# Configuration
# ================================

NUM_RUNS = 5
RESULTS_DIR = "/home/w5/pydemo/mecon25/results"
DQN_FILE_PATTERN = "dqn_results_run_{}.npz"
A2C_FILE_PATTERN = "a2c_results_run_{}.npz"
RDQN_FILE_PATTERN = "rdqn_results_run_{}.npz"

# ================================
# Data Loading and Preparation
# ================================

dqn_usage = []
a2c_usage = []
rdqn_usage = []
dqn_mmtc = []
a2c_mmtc = []
rdqn_mmtc = []

print("Checking shapes of all metrics across runs...")

max_length = max([
    len(np.load(os.path.join(RESULTS_DIR, DQN_FILE_PATTERN.format(run_id)))['rewards'])
    for run_id in range(1, NUM_RUNS + 1)
])

print(f"Max length across all runs: {max_length}")

def resize_data(data, target_length):
    if len(data) > target_length:
        return data[:target_length]
    elif len(data) < target_length:
        return np.pad(data, (0, target_length - len(data)), mode='constant', constant_values=np.nan)
    return data

for run_id in range(1, NUM_RUNS + 1):
    dqn_file = os.path.join(RESULTS_DIR, DQN_FILE_PATTERN.format(run_id))
    a2c_file = os.path.join(RESULTS_DIR, A2C_FILE_PATTERN.format(run_id))
    rdqn_file = os.path.join(RESULTS_DIR, RDQN_FILE_PATTERN.format(run_id))
    if os.path.exists(dqn_file) and os.path.exists(a2c_file) and os.path.exists(rdqn_file):
        dqn_results = np.load(dqn_file)
        a2c_results = np.load(a2c_file)
        rdqn_results = np.load(rdqn_file)
        dqn_usage.append(resize_data(dqn_results['rewards'], max_length))
        a2c_usage.append(resize_data(a2c_results['rewards'], max_length))
        rdqn_usage.append(resize_data(rdqn_results['rewards'], max_length))
    else:
        print(f"Warning: One or more of {dqn_file}, {a2c_file}, or {rdqn_file} not found!")

dqn_usage = np.array(dqn_usage)
a2c_usage = np.array(a2c_usage)
rdqn_usage = np.array(rdqn_usage)
dqn_mmtc = np.array(dqn_mmtc)
a2c_mmtc = np.array(a2c_mmtc)
rdqn_mmtc = np.array(rdqn_mmtc)

# Generate random baseline for load balancing metric
random_usage = []
for _ in range(NUM_RUNS):
    random_metric = np.random.uniform(-1, 0, size=max_length)
    random_usage.append(random_metric)
random_usage = np.array(random_usage)

mean_dqn = np.nanmean(dqn_usage, axis=0)
std_dqn = np.nanstd(dqn_usage, axis=0)
mean_a2c = np.nanmean(a2c_usage, axis=0)
std_a2c = np.nanstd(a2c_usage, axis=0)
mean_rdqn = np.nanmean(rdqn_usage, axis=0)
std_rdqn = np.nanstd(rdqn_usage, axis=0)
mean_random = np.nanmean(random_usage, axis=0)
std_random = np.nanstd(random_usage, axis=0)

def smooth(data, window=20):
    return np.convolve(data, np.ones(window)/window, mode='valid')

def adjust_std_and_smooth(mean_data, std_data):
    smoothed_mean = smooth(mean_data)
    smoothed_std = smooth(std_data)[:len(smoothed_mean)]
    smoothed_std = np.clip(smoothed_std, 0, 1.0)
    return smoothed_mean, smoothed_std

SHOW_STD_DEV = False

def plot_results():
    plt.figure(figsize=(6, 4))
    dqn_mean, dqn_std = adjust_std_and_smooth(mean_dqn, std_dqn)
    a2c_mean, a2c_std = adjust_std_and_smooth(mean_a2c, std_a2c)
    rdqn_mean, rdqn_std = adjust_std_and_smooth(mean_rdqn, std_rdqn)
    random_mean, random_std = adjust_std_and_smooth(mean_random, std_random)

    plt.plot(dqn_mean, label='DQN', color='#d95f02', linewidth=2, marker='o', markevery=0.1)
    plt.plot(a2c_mean, label='A2C', color='#1b9e77', linewidth=2, marker='s', markevery=0.1)
    plt.plot(rdqn_mean, label='Rainbow', color='#7570b3', linewidth=2, marker='^', markevery=0.1)
    plt.plot(random_mean, label='Random', color='gray', linestyle='--', linewidth=2, marker='x', markevery=0.1)

    plt.xlim(0, max_length)
    plt.title("Network Slicing Load Balancing Metric")
    plt.xlabel("Number of Episodes")
    plt.ylabel("Normalized Usage Difference (proxy)")
    plt.legend()
    plt.tight_layout()
    plt.savefig('Load_Balance_Metric.png')
    plt.close()

plot_results()
