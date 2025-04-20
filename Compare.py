# Compare.py

import numpy as np
import matplotlib.pyplot as plt
import os

# ================================
# Configuration
# ================================

NUM_RUNS = 5  # Number of runs you performed

# Define the file pattern for the saved NPZ files
DQN_FILE_PATTERN = "dqn_results_run_{}.npz"
A2C_FILE_PATTERN = "a2c_results_run_{}.npz"

# Initialize lists to hold data from all runs
dqn_urllc_blocks_all = []
a2c_urllc_blocks_all = []
dqn_embb_blocks_all = []
a2c_embb_blocks_all = []
dqn_urllc_sla_all = []
a2c_urllc_sla_all = []
dqn_embb_sla_all = []
a2c_embb_sla_all = []

# ================================
# Check the Shapes of the Data
# ================================

print("Checking shapes of all metrics across runs...")

# Find the max length of the data across all runs
max_length = max([len(np.load(DQN_FILE_PATTERN.format(run_id))['urllc_blocks']) 
                  for run_id in range(1, NUM_RUNS + 1)])

print(f"Max length across all runs: {max_length}")

# ================================
# Resizing the Data for Consistency
# ================================

def resize_data(data, target_length):
    if len(data) > target_length:
        return data[:target_length]  # Trim
    elif len(data) < target_length:
        return np.pad(data, (0, target_length - len(data)), mode='constant', constant_values=np.nan)  # Pad with NaNs
    return data

# ================================
# Load Results from Multiple Runs
# ================================

for run_id in range(1, NUM_RUNS + 1):
    # Load DQN and A2C results for this run
    dqn_file = DQN_FILE_PATTERN.format(run_id)
    a2c_file = A2C_FILE_PATTERN.format(run_id)
    
    if os.path.exists(dqn_file) and os.path.exists(a2c_file):
        dqn_results = np.load(dqn_file)
        a2c_results = np.load(a2c_file)
        
        # Resize data to match the maximum length
        dqn_urllc_blocks_all.append(resize_data(dqn_results['urllc_blocks'], max_length))
        a2c_urllc_blocks_all.append(resize_data(a2c_results['urllc_blocks'], max_length))
        dqn_embb_blocks_all.append(resize_data(dqn_results['embb_blocks'], max_length))
        a2c_embb_blocks_all.append(resize_data(a2c_results['embb_blocks'], max_length))
        dqn_urllc_sla_all.append(resize_data(dqn_results['urllc_sla'], max_length))
        a2c_urllc_sla_all.append(resize_data(a2c_results['urllc_sla'], max_length))
        dqn_embb_sla_all.append(resize_data(dqn_results['embb_sla'], max_length))
        a2c_embb_sla_all.append(resize_data(a2c_results['embb_sla'], max_length))
    else:
        print(f"Warning: One or both of {dqn_file} or {a2c_file} not found!")

# Convert lists to numpy arrays for easier manipulation
dqn_urllc_blocks_all = np.array(dqn_urllc_blocks_all)
a2c_urllc_blocks_all = np.array(a2c_urllc_blocks_all)
dqn_embb_blocks_all = np.array(dqn_embb_blocks_all)
a2c_embb_blocks_all = np.array(a2c_embb_blocks_all)
dqn_urllc_sla_all = np.array(dqn_urllc_sla_all)
a2c_urllc_sla_all = np.array(a2c_urllc_sla_all)
dqn_embb_sla_all = np.array(dqn_embb_sla_all)
a2c_embb_sla_all = np.array(a2c_embb_sla_all)

# ================================
# Compute Averages and Standard Deviations
# ================================

# Calculate the mean and std deviation across all runs
mean_dqn_urllc_blocks = np.mean(dqn_urllc_blocks_all, axis=0)
std_dqn_urllc_blocks = np.std(dqn_urllc_blocks_all, axis=0)

mean_a2c_urllc_blocks = np.mean(a2c_urllc_blocks_all, axis=0)
std_a2c_urllc_blocks = np.std(a2c_urllc_blocks_all, axis=0)

mean_dqn_embb_blocks = np.mean(dqn_embb_blocks_all, axis=0)
std_dqn_embb_blocks = np.std(dqn_embb_blocks_all, axis=0)

mean_a2c_embb_blocks = np.mean(a2c_embb_blocks_all, axis=0)
std_a2c_embb_blocks = np.std(a2c_embb_blocks_all, axis=0)

mean_dqn_urllc_sla = np.mean(dqn_urllc_sla_all, axis=0)
std_dqn_urllc_sla = np.std(dqn_urllc_sla_all, axis=0)

mean_a2c_urllc_sla = np.mean(a2c_urllc_sla_all, axis=0)
std_a2c_urllc_sla = np.std(a2c_urllc_sla_all, axis=0)

mean_dqn_embb_sla = np.mean(dqn_embb_sla_all, axis=0)
std_dqn_embb_sla = np.std(dqn_embb_sla_all, axis=0)

mean_a2c_embb_sla = np.mean(a2c_embb_sla_all, axis=0)
std_a2c_embb_sla = np.std(a2c_embb_sla_all, axis=0)

# ================================
# Smoothing Helper
# ================================

def smooth(data, window=20):
    return np.convolve(data, np.ones(window)/window, mode='valid')

# ================================
# Adjust Standard Deviation and Smoothing
# ================================

def adjust_std_and_smooth(mean_data, std_data):
    # Smooth both mean and std before plotting to ensure consistent lengths
    smoothed_mean = smooth(mean_data)
    smoothed_std = smooth(std_data)[:len(smoothed_mean)]  # Trim std to match smoothed mean length
    return smoothed_mean, smoothed_std

# ================================
# Plot SLA and Block Rate by Slice Type
# ================================

fig, axs = plt.subplots(2, 2, figsize=(14, 8))

# URLLC Block Rate
smoothed_dqn_urllc, smoothed_std_dqn_urllc = adjust_std_and_smooth(mean_dqn_urllc_blocks, std_dqn_urllc_blocks)
smoothed_a2c_urllc, smoothed_std_a2c_urllc = adjust_std_and_smooth(mean_a2c_urllc_blocks, std_a2c_urllc_blocks)

axs[0, 0].plot(smoothed_dqn_urllc, label="DQN", alpha=0.8)
axs[0, 0].fill_between(range(len(smoothed_dqn_urllc)), smoothed_dqn_urllc - smoothed_std_dqn_urllc, smoothed_dqn_urllc + smoothed_std_dqn_urllc, alpha=0.3)
axs[0, 0].plot(smoothed_a2c_urllc, label="A2C", alpha=0.8)
axs[0, 0].fill_between(range(len(smoothed_a2c_urllc)), smoothed_a2c_urllc - smoothed_std_a2c_urllc, smoothed_a2c_urllc + smoothed_std_a2c_urllc, alpha=0.3)
axs[0, 0].set_title("URLLC Block Rate (Smoothed)")
axs[0, 0].set_ylabel("Block Ratio")
axs[0, 0].legend()
axs[0, 0].grid(True)

# eMBB Block Rate
smoothed_dqn_embb, smoothed_std_dqn_embb = adjust_std_and_smooth(mean_dqn_embb_blocks, std_dqn_embb_blocks)
smoothed_a2c_embb, smoothed_std_a2c_embb = adjust_std_and_smooth(mean_a2c_embb_blocks, std_a2c_embb_blocks)

axs[0, 1].plot(smoothed_dqn_embb, label="DQN", alpha=0.8)
axs[0, 1].fill_between(range(len(smoothed_dqn_embb)), smoothed_dqn_embb - smoothed_std_dqn_embb, smoothed_dqn_embb + smoothed_std_dqn_embb, alpha=0.3)
axs[0, 1].plot(smoothed_a2c_embb, label="A2C", alpha=0.8)
axs[0, 1].fill_between(range(len(smoothed_a2c_embb)), smoothed_a2c_embb - smoothed_std_a2c_embb, smoothed_a2c_embb + smoothed_std_a2c_embb, alpha=0.3)
axs[0, 1].set_title("eMBB Block Rate (Smoothed)")
axs[0, 1].legend()
axs[0, 1].grid(True)

# URLLC SLA
smoothed_dqn_urllc_sla, smoothed_std_dqn_urllc_sla = adjust_std_and_smooth(mean_dqn_urllc_sla, std_dqn_urllc_sla)
smoothed_a2c_urllc_sla, smoothed_std_a2c_urllc_sla = adjust_std_and_smooth(mean_a2c_urllc_sla, std_a2c_urllc_sla)

axs[1, 0].plot(smoothed_dqn_urllc_sla, label="DQN", alpha=0.8)
axs[1, 0].fill_between(range(len(smoothed_dqn_urllc_sla)), smoothed_dqn_urllc_sla - smoothed_std_dqn_urllc_sla, smoothed_dqn_urllc_sla + smoothed_std_dqn_urllc_sla, alpha=0.3)
axs[1, 0].plot(smoothed_a2c_urllc_sla, label="A2C", alpha=0.8)
axs[1, 0].fill_between(range(len(smoothed_a2c_urllc_sla)), smoothed_a2c_urllc_sla - smoothed_std_a2c_urllc_sla, smoothed_a2c_urllc_sla + smoothed_std_a2c_urllc_sla, alpha=0.3)
axs[1, 0].set_title("URLLC SLA Preservation (Smoothed)")
axs[1, 0].set_ylabel("SLA Ratio")
axs[1, 0].set_ylim(0, 1.05)
axs[1, 0].legend()
axs[1, 0].grid(True)

# eMBB SLA
smoothed_dqn_embb_sla, smoothed_std_dqn_embb_sla = adjust_std_and_smooth(mean_dqn_embb_sla, std_dqn_embb_sla)
smoothed_a2c_embb_sla, smoothed_std_a2c_embb_sla = adjust_std_and_smooth(mean_a2c_embb_sla, std_a2c_embb_sla)

axs[1, 1].plot(smoothed_dqn_embb_sla, label="DQN", alpha=0.8)
axs[1, 1].fill_between(range(len(smoothed_dqn_embb_sla)), smoothed_dqn_embb_sla - smoothed_std_dqn_embb_sla, smoothed_dqn_embb_sla + smoothed_std_dqn_embb_sla, alpha=0.3)
axs[1, 1].plot(smoothed_a2c_embb_sla, label="A2C", alpha=0.8)
axs[1, 1].fill_between(range(len(smoothed_a2c_embb_sla)), smoothed_a2c_embb_sla - smoothed_std_a2c_embb_sla, smoothed_a2c_embb_sla + smoothed_std_a2c_embb_sla, alpha=0.3)
axs[1, 1].set_title("eMBB SLA Preservation (Smoothed)")
axs[1, 1].set_ylim(0, 1.05)
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.savefig("benchmark_comparison.png")
plt.show()
