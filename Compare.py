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

# Only keep lists for load balancing calculation
dqn_urllc = []
a2c_urllc = []
rdqn_urllc = []
dqn_embb = []
a2c_embb = []
rdqn_embb = []

print("Checking shapes of all metrics across runs...")

max_length = max([
    len(np.load(os.path.join(RESULTS_DIR, DQN_FILE_PATTERN.format(run_id)))['urllc_blocks'])
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
        dqn_urllc.append(resize_data(dqn_results['urllc_blocks'], max_length))
        a2c_urllc.append(resize_data(a2c_results['urllc_blocks'], max_length))
        rdqn_urllc.append(resize_data(rdqn_results['urllc_blocks'], max_length))
        dqn_embb.append(resize_data(dqn_results['embb_blocks'], max_length))
        a2c_embb.append(resize_data(a2c_results['embb_blocks'], max_length))
        rdqn_embb.append(resize_data(rdqn_results['embb_blocks'], max_length))
    else:
        print(f"Warning: One or more of {dqn_file}, {a2c_file}, or {rdqn_file} not found!")

dqn_urllc = np.array(dqn_urllc)
a2c_urllc = np.array(a2c_urllc)
rdqn_urllc = np.array(rdqn_urllc)
dqn_embb = np.array(dqn_embb)
a2c_embb = np.array(a2c_embb)
rdqn_embb = np.array(rdqn_embb)

def compute_load_balance_metric(urllc, embb, urllc_quota=30, embb_quota=70):
    urllc_norm = urllc / urllc_quota
    embb_norm = embb / embb_quota
    return np.abs(urllc_norm - embb_norm)

dqn_load_balance_all = [compute_load_balance_metric(u, e) for u, e in zip(dqn_urllc, dqn_embb)]
a2c_load_balance_all = [compute_load_balance_metric(u, e) for u, e in zip(a2c_urllc, a2c_embb)]
rdqn_load_balance_all = [compute_load_balance_metric(u, e) for u, e in zip(rdqn_urllc, rdqn_embb)]

random_urllc = np.random.randint(0, 5, size=(NUM_RUNS, max_length))
random_embb = np.random.randint(0, 50, size=(NUM_RUNS, max_length))
random_load_balance_all = [compute_load_balance_metric(u, e) for u, e in zip(random_urllc, random_embb)]

dqn_load_balance_all = np.array(dqn_load_balance_all)
a2c_load_balance_all = np.array(a2c_load_balance_all)
rdqn_load_balance_all = np.array(rdqn_load_balance_all)
random_load_balance_all = np.array(random_load_balance_all)

mean_dqn = np.nanmean(dqn_load_balance_all, axis=0)
std_dqn = np.nanstd(dqn_load_balance_all, axis=0)
mean_a2c = np.nanmean(a2c_load_balance_all, axis=0)
std_a2c = np.nanstd(a2c_load_balance_all, axis=0)
mean_rdqn = np.nanmean(rdqn_load_balance_all, axis=0)
std_rdqn = np.nanstd(rdqn_load_balance_all, axis=0)
mean_random = np.nanmean(random_load_balance_all, axis=0)
std_random = np.nanstd(random_load_balance_all, axis=0)

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

    plt.plot(dqn_mean, label='DQN + SAC', color='#d95f02', linewidth=2, marker='o', markevery=0.1)
    if SHOW_STD_DEV:
        plt.fill_between(range(len(dqn_mean)), dqn_mean - dqn_std, dqn_mean + dqn_std, color='#d95f02', alpha=0.1)
    plt.plot(a2c_mean, label='A2C + SAC', color='#1b9e77', linewidth=2, marker='s', markevery=0.1)
    if SHOW_STD_DEV:
        plt.fill_between(range(len(a2c_mean)), a2c_mean - a2c_std, a2c_mean + a2c_std, color='#1b9e77', alpha=0.1)
    plt.plot(rdqn_mean, label='Rainbow + SAC', color='#7570b3', linewidth=2, marker='^', markevery=0.1)
    if SHOW_STD_DEV:
        plt.fill_between(range(len(rdqn_mean)), rdqn_mean - rdqn_std, rdqn_mean + rdqn_std, color='#7570b3', alpha=0.1)
    plt.plot(random_mean, label='SAC', color='gray', linestyle='--', linewidth=2, marker='x', markevery=0.1)
    if SHOW_STD_DEV:
        plt.fill_between(range(len(random_mean)), random_mean - random_std, random_mean + random_std, color='gray', alpha=0.1)

    plt.xlim(0, max_length)
    plt.title("Network Slicing Load Balancing Metric")
    plt.xlabel("Number of Episodes")
    plt.ylabel("Normalized Usage Difference")
    plt.legend()
    plt.tight_layout()
    plt.savefig('Load_Balance_Metric.png')
    plt.close()

plot_results()
