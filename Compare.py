# Compare.py

import numpy as np
import matplotlib.pyplot as plt
import os

# ================================
# Configuration
# ================================

NUM_RUNS = 100  # Number of runs you performed

# Define the file pattern for the saved NPZ files
DQN_FILE_PATTERN = "dqn_results_run_{}.npz"
A2C_FILE_PATTERN = "a2c_results_run_{}.npz"
LSTMDQN_FILE_PATTERN = "lstm_dqn_results_run_{}.npz"  # Added LSTM-DQN pattern

# Initialize lists to hold data from all runs
dqn_urllc_blocks_all = []
a2c_urllc_blocks_all = []
lstm_dqn_urllc_blocks_all = []  # Added LSTM-DQN list
dqn_embb_blocks_all = []
a2c_embb_blocks_all = []
lstm_dqn_embb_blocks_all = []  # Added LSTM-DQN list
dqn_urllc_sla_all = []
a2c_urllc_sla_all = []
lstm_dqn_urllc_sla_all = []  # Added LSTM-DQN list
dqn_embb_sla_all = []
a2c_embb_sla_all = []
lstm_dqn_embb_sla_all = []  # Added LSTM-DQN list

# ================================
# Check the Shapes of the Data
# ================================

print("Checking shapes of all metrics across runs...")

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
    # Load DQN, A2C, and LSTM-DQN results for this run
    dqn_file = DQN_FILE_PATTERN.format(run_id)
    a2c_file = A2C_FILE_PATTERN.format(run_id)
    lstm_dqn_file = LSTMDQN_FILE_PATTERN.format(run_id)
    
    if os.path.exists(dqn_file) and os.path.exists(a2c_file) and os.path.exists(lstm_dqn_file):
        dqn_results = np.load(dqn_file)
        a2c_results = np.load(a2c_file)
        lstm_dqn_results = np.load(lstm_dqn_file)
        
        # Resize data to match the maximum length
        dqn_urllc_blocks_all.append(resize_data(dqn_results['urllc_blocks'], max_length))
        a2c_urllc_blocks_all.append(resize_data(a2c_results['urllc_blocks'], max_length))
        lstm_dqn_urllc_blocks_all.append(resize_data(lstm_dqn_results['urllc_blocks'], max_length))  # LSTM-DQN
        dqn_embb_blocks_all.append(resize_data(dqn_results['embb_blocks'], max_length))
        a2c_embb_blocks_all.append(resize_data(a2c_results['embb_blocks'], max_length))
        lstm_dqn_embb_blocks_all.append(resize_data(lstm_dqn_results['embb_blocks'], max_length))  # LSTM-DQN
        dqn_urllc_sla_all.append(resize_data(dqn_results['urllc_sla'], max_length))
        a2c_urllc_sla_all.append(resize_data(a2c_results['urllc_sla'], max_length))
        lstm_dqn_urllc_sla_all.append(resize_data(lstm_dqn_results['urllc_sla'], max_length))  # LSTM-DQN
        dqn_embb_sla_all.append(resize_data(dqn_results['embb_sla'], max_length))
        a2c_embb_sla_all.append(resize_data(a2c_results['embb_sla'], max_length))
        lstm_dqn_embb_sla_all.append(resize_data(lstm_dqn_results['embb_sla'], max_length))  # LSTM-DQN
    else:
        print(f"Warning: One or more of {dqn_file}, {a2c_file}, or {lstm_dqn_file} not found!")

# Convert lists to numpy arrays for easier manipulation
dqn_urllc_blocks_all = np.array(dqn_urllc_blocks_all)
a2c_urllc_blocks_all = np.array(a2c_urllc_blocks_all)
lstm_dqn_urllc_blocks_all = np.array(lstm_dqn_urllc_blocks_all)  # LSTM-DQN
dqn_embb_blocks_all = np.array(dqn_embb_blocks_all)
a2c_embb_blocks_all = np.array(a2c_embb_blocks_all)
lstm_dqn_embb_blocks_all = np.array(lstm_dqn_embb_blocks_all)  # LSTM-DQN
dqn_urllc_sla_all = np.array(dqn_urllc_sla_all)
a2c_urllc_sla_all = np.array(a2c_urllc_sla_all)
lstm_dqn_urllc_sla_all = np.array(lstm_dqn_urllc_sla_all)  # LSTM-DQN
dqn_embb_sla_all = np.array(dqn_embb_sla_all)
a2c_embb_sla_all = np.array(a2c_embb_sla_all)
lstm_dqn_embb_sla_all = np.array(lstm_dqn_embb_sla_all)  # LSTM-DQN

# ================================
# Compute Averages and Standard Deviations
# ================================

# Calculate the mean and std deviation across all runs
mean_dqn_urllc_blocks = np.mean(dqn_urllc_blocks_all, axis=0)
std_dqn_urllc_blocks = np.std(dqn_urllc_blocks_all, axis=0)

mean_a2c_urllc_blocks = np.mean(a2c_urllc_blocks_all, axis=0)
std_a2c_urllc_blocks = np.std(a2c_urllc_blocks_all, axis=0)

mean_lstm_dqn_urllc_blocks = np.mean(lstm_dqn_urllc_blocks_all, axis=0)  # LSTM-DQN
std_lstm_dqn_urllc_blocks = np.std(lstm_dqn_urllc_blocks_all, axis=0)  # LSTM-DQN

mean_dqn_embb_blocks = np.mean(dqn_embb_blocks_all, axis=0)
std_dqn_embb_blocks = np.std(dqn_embb_blocks_all, axis=0)

mean_a2c_embb_blocks = np.mean(a2c_embb_blocks_all, axis=0)
std_a2c_embb_blocks = np.std(a2c_embb_blocks_all, axis=0)

mean_lstm_dqn_embb_blocks = np.mean(lstm_dqn_embb_blocks_all, axis=0)  # LSTM-DQN
std_lstm_dqn_embb_blocks = np.std(lstm_dqn_embb_blocks_all, axis=0)  # LSTM-DQN

mean_dqn_urllc_sla = np.mean(dqn_urllc_sla_all, axis=0)
std_dqn_urllc_sla = np.std(dqn_urllc_sla_all, axis=0)

mean_a2c_urllc_sla = np.mean(a2c_urllc_sla_all, axis=0)
std_a2c_urllc_sla = np.std(a2c_urllc_sla_all, axis=0)

mean_lstm_dqn_urllc_sla = np.mean(lstm_dqn_urllc_sla_all, axis=0)  # LSTM-DQN
std_lstm_dqn_urllc_sla = np.std(lstm_dqn_urllc_sla_all, axis=0)  # LSTM-DQN

mean_dqn_embb_sla = np.mean(dqn_embb_sla_all, axis=0)
std_dqn_embb_sla = np.std(dqn_embb_sla_all, axis=0)

mean_a2c_embb_sla = np.mean(a2c_embb_sla_all, axis=0)
std_a2c_embb_sla = np.std(a2c_embb_sla_all, axis=0)

mean_lstm_dqn_embb_sla = np.mean(lstm_dqn_embb_sla_all, axis=0)  # LSTM-DQN
std_lstm_dqn_embb_sla = np.std(lstm_dqn_embb_sla_all, axis=0)  # LSTM-DQN

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
    
    # Clip standard deviation values to a maximum limit
    smoothed_std = np.clip(smoothed_std, 0, 1.0)
    
    return smoothed_mean, smoothed_std

# ================================
# Plotting
# ================================

def plot_results():
    plt.figure(figsize=(12, 12))

    # Plot URLLC Blocks
    plt.subplot(2, 2, 1)
    dqn_mean, dqn_std = adjust_std_and_smooth(mean_dqn_urllc_blocks, std_dqn_urllc_blocks)
    a2c_mean, a2c_std = adjust_std_and_smooth(mean_a2c_urllc_blocks, std_a2c_urllc_blocks)
    lstm_dqn_mean, lstm_dqn_std = adjust_std_and_smooth(mean_lstm_dqn_urllc_blocks, std_lstm_dqn_urllc_blocks)

    # Clip block rates at 0 to avoid negative values
    dqn_mean = np.clip(dqn_mean, 0, None)
    a2c_mean = np.clip(a2c_mean, 0, None)
    lstm_dqn_mean = np.clip(lstm_dqn_mean, 0, None)

    # Clip the std dev to avoid exceeding bounds
    dqn_std = np.clip(dqn_std, 0, dqn_mean)
    a2c_std = np.clip(a2c_std, 0, a2c_mean)
    lstm_dqn_std = np.clip(lstm_dqn_std, 0, lstm_dqn_mean)

    plt.plot(dqn_mean, label='DQN', color='blue')
    plt.fill_between(range(len(dqn_mean)), dqn_mean - dqn_std, dqn_mean + dqn_std, color='blue', alpha=0.1)

    plt.plot(a2c_mean, label='A2C', color='orange')
    plt.fill_between(range(len(a2c_mean)), a2c_mean - a2c_std, a2c_mean + a2c_std, color='orange', alpha=0.1)

    plt.plot(lstm_dqn_mean, label='LSTM-DQN', color='green')
    plt.fill_between(range(len(lstm_dqn_mean)), lstm_dqn_mean - lstm_dqn_std, lstm_dqn_mean + lstm_dqn_std, color='green', alpha=0.1)

    plt.title("URLLC Blocks")
    plt.xlabel("Time Steps")
    plt.ylabel("Blocks")
    plt.legend()

    # Plot eMBB Blocks
    plt.subplot(2, 2, 2)
    dqn_mean, dqn_std = adjust_std_and_smooth(mean_dqn_embb_blocks, std_dqn_embb_blocks)
    a2c_mean, a2c_std = adjust_std_and_smooth(mean_a2c_embb_blocks, std_a2c_embb_blocks)
    lstm_dqn_mean, lstm_dqn_std = adjust_std_and_smooth(mean_lstm_dqn_embb_blocks, std_lstm_dqn_embb_blocks)

    # Clip block rates at 0 to avoid negative values
    dqn_mean = np.clip(dqn_mean, 0, None)
    a2c_mean = np.clip(a2c_mean, 0, None)
    lstm_dqn_mean = np.clip(lstm_dqn_mean, 0, None)

    # Clip the std dev to avoid exceeding bounds
    dqn_std = np.clip(dqn_std, 0, dqn_mean)
    a2c_std = np.clip(a2c_std, 0, a2c_mean)
    lstm_dqn_std = np.clip(lstm_dqn_std, 0, lstm_dqn_mean)

    plt.plot(dqn_mean, label='DQN', color='blue')
    plt.fill_between(range(len(dqn_mean)), dqn_mean - dqn_std, dqn_mean + dqn_std, color='blue', alpha=0.1)

    plt.plot(a2c_mean, label='A2C', color='orange')
    plt.fill_between(range(len(a2c_mean)), a2c_mean - a2c_std, a2c_mean + a2c_std, color='orange', alpha=0.1)

    plt.plot(lstm_dqn_mean, label='LSTM-DQN', color='green')
    plt.fill_between(range(len(lstm_dqn_mean)), lstm_dqn_mean - lstm_dqn_std, lstm_dqn_mean + lstm_dqn_std, color='green', alpha=0.1)

    plt.title("eMBB Blocks")
    plt.xlabel("Time Steps")
    plt.ylabel("Blocks")
    plt.legend()

    # Plot SLA for URLLC
    plt.subplot(2, 2, 3)
    dqn_mean, dqn_std = adjust_std_and_smooth(mean_dqn_urllc_sla, std_dqn_urllc_sla)
    a2c_mean, a2c_std = adjust_std_and_smooth(mean_a2c_urllc_sla, std_a2c_urllc_sla)
    lstm_dqn_mean, lstm_dqn_std = adjust_std_and_smooth(mean_lstm_dqn_urllc_sla, std_lstm_dqn_urllc_sla)

    # Clip SLA values between 0 and 1
    dqn_mean = np.clip(dqn_mean, 0, 1)
    a2c_mean = np.clip(a2c_mean, 0, 1)
    lstm_dqn_mean = np.clip(lstm_dqn_mean, 0, 1)

    # Clip the std dev to avoid exceeding bounds
    dqn_std = np.clip(dqn_std, 0, 1 - dqn_mean)
    a2c_std = np.clip(a2c_std, 0, 1 - a2c_mean)
    lstm_dqn_std = np.clip(lstm_dqn_std, 0, 1 - lstm_dqn_mean)

    plt.plot(dqn_mean, label='DQN', color='blue')
    plt.fill_between(range(len(dqn_mean)), dqn_mean - dqn_std, dqn_mean + dqn_std, color='blue', alpha=0.1)

    plt.plot(a2c_mean, label='A2C', color='orange')
    plt.fill_between(range(len(a2c_mean)), a2c_mean - a2c_std, a2c_mean + a2c_std, color='orange', alpha=0.1)

    plt.plot(lstm_dqn_mean, label='LSTM-DQN', color='green')
    plt.fill_between(range(len(lstm_dqn_mean)), lstm_dqn_mean - lstm_dqn_std, lstm_dqn_mean + lstm_dqn_std, color='green', alpha=0.1)

    plt.title("URLLC SLA")
    plt.xlabel("Time Steps")
    plt.ylabel("SLA")
    plt.legend()

    # Plot SLA for eMBB
    plt.subplot(2, 2, 4)
    dqn_mean, dqn_std = adjust_std_and_smooth(mean_dqn_embb_sla, std_dqn_embb_sla)
    a2c_mean, a2c_std = adjust_std_and_smooth(mean_a2c_embb_sla, std_a2c_embb_sla)
    lstm_dqn_mean, lstm_dqn_std = adjust_std_and_smooth(mean_lstm_dqn_embb_sla, std_lstm_dqn_embb_sla)

    # Clip SLA values between 0 and 1
    dqn_mean = np.clip(dqn_mean, 0, 1)
    a2c_mean = np.clip(a2c_mean, 0, 1)
    lstm_dqn_mean = np.clip(lstm_dqn_mean, 0, 1)

    # Clip the std dev to avoid exceeding bounds
    dqn_std = np.clip(dqn_std, 0, 1 - dqn_mean)
    a2c_std = np.clip(a2c_std, 0, 1 - a2c_mean)
    lstm_dqn_std = np.clip(lstm_dqn_std, 0, 1 - lstm_dqn_mean)

    plt.plot(dqn_mean, label='DQN', color='blue')
    plt.fill_between(range(len(dqn_mean)), dqn_mean - dqn_std, dqn_mean + dqn_std, color='blue', alpha=0.1)

    plt.plot(a2c_mean, label='A2C', color='orange')
    plt.fill_between(range(len(a2c_mean)), a2c_mean - a2c_std, a2c_mean + a2c_std, color='orange', alpha=0.1)

    plt.plot(lstm_dqn_mean, label='LSTM-DQN', color='green')
    plt.fill_between(range(len(lstm_dqn_mean)), lstm_dqn_mean - lstm_dqn_std, lstm_dqn_mean + lstm_dqn_std, color='green', alpha=0.1)

    plt.title("eMBB SLA")
    plt.xlabel("Time Steps")
    plt.ylabel("SLA")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Benchmark_plots.png')
    plt.show()

# Run plotting function
plot_results()
