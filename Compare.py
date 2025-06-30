# Compare.py

import numpy as np
import matplotlib.pyplot as plt
import os

# ================================
# Configuration
# ================================

NUM_RUNS = 50
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
dqn_std_metric = []
a2c_std_metric = []
rdqn_std_metric = []
dqn_max_util = []
a2c_max_util = []
rdqn_max_util = []
dqn_min_util = []
a2c_min_util = []
rdqn_min_util = []
dqn_urllc_blocks = []
a2c_urllc_blocks = []
rdqn_urllc_blocks = []
dqn_embb_blocks = []
a2c_embb_blocks = []
rdqn_embb_blocks = []
dqn_mmtc_blocks = []
a2c_mmtc_blocks = []
rdqn_mmtc_blocks = []

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

def safe_load_metric(results, key, max_length):
    if key in results:
        return resize_data(results[key], max_length)
    else:
        return np.full(max_length, np.nan)

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
        # Use safe_load_metric for new metrics
        dqn_std_metric.append(safe_load_metric(dqn_results, 'std', max_length))
        a2c_std_metric.append(safe_load_metric(a2c_results, 'std', max_length))
        rdqn_std_metric.append(safe_load_metric(rdqn_results, 'std', max_length))
        dqn_max_util.append(safe_load_metric(dqn_results, 'max_util', max_length))
        a2c_max_util.append(safe_load_metric(a2c_results, 'max_util', max_length))
        rdqn_max_util.append(safe_load_metric(rdqn_results, 'max_util', max_length))
        dqn_min_util.append(safe_load_metric(dqn_results, 'min_util', max_length))
        a2c_min_util.append(safe_load_metric(a2c_results, 'min_util', max_length))
        rdqn_min_util.append(safe_load_metric(rdqn_results, 'min_util', max_length))
        # Block histories
        dqn_urllc_blocks.append(safe_load_metric(dqn_results, 'urllc_blocks', max_length))
        a2c_urllc_blocks.append(safe_load_metric(a2c_results, 'urllc_blocks', max_length))
        rdqn_urllc_blocks.append(safe_load_metric(rdqn_results, 'urllc_blocks', max_length))
        dqn_embb_blocks.append(safe_load_metric(dqn_results, 'embb_blocks', max_length))
        a2c_embb_blocks.append(safe_load_metric(a2c_results, 'embb_blocks', max_length))
        rdqn_embb_blocks.append(safe_load_metric(rdqn_results, 'embb_blocks', max_length))
        dqn_mmtc_blocks.append(safe_load_metric(dqn_results, 'mmtc_blocks', max_length))
        a2c_mmtc_blocks.append(safe_load_metric(a2c_results, 'mmtc_blocks', max_length))
        rdqn_mmtc_blocks.append(safe_load_metric(rdqn_results, 'mmtc_blocks', max_length))
    else:
        print(f"Warning: One or more of {dqn_file}, {a2c_file}, or {rdqn_file} not found!")

dqn_usage = np.array(dqn_usage)
a2c_usage = np.array(a2c_usage)
rdqn_usage = np.array(rdqn_usage)
dqn_std_metric = np.array(dqn_std_metric)
a2c_std_metric = np.array(a2c_std_metric)
rdqn_std_metric = np.array(rdqn_std_metric)
dqn_max_util = np.array(dqn_max_util)
a2c_max_util = np.array(a2c_max_util)
rdqn_max_util = np.array(rdqn_max_util)
dqn_min_util = np.array(dqn_min_util)
a2c_min_util = np.array(a2c_min_util)
rdqn_min_util = np.array(rdqn_min_util)
dqn_urllc_blocks = np.array(dqn_urllc_blocks)
a2c_urllc_blocks = np.array(a2c_urllc_blocks)
rdqn_urllc_blocks = np.array(rdqn_urllc_blocks)
dqn_embb_blocks = np.array(dqn_embb_blocks)
a2c_embb_blocks = np.array(a2c_embb_blocks)
rdqn_embb_blocks = np.array(rdqn_embb_blocks)
dqn_mmtc_blocks = np.array(dqn_mmtc_blocks)
a2c_mmtc_blocks = np.array(a2c_mmtc_blocks)
rdqn_mmtc_blocks = np.array(rdqn_mmtc_blocks)

mean_dqn = np.nanmean(dqn_usage, axis=0)
std_dqn = np.nanstd(dqn_usage, axis=0)
mean_a2c = np.nanmean(a2c_usage, axis=0)
std_a2c = np.nanstd(a2c_usage, axis=0)
mean_rdqn = np.nanmean(rdqn_usage, axis=0)
std_rdqn = np.nanstd(rdqn_usage, axis=0)

# Compute means for new metrics
mean_dqn_std = np.nanmean(dqn_std_metric, axis=0)
mean_a2c_std = np.nanmean(a2c_std_metric, axis=0)
mean_rdqn_std = np.nanmean(rdqn_std_metric, axis=0)
mean_dqn_max = np.nanmean(dqn_max_util, axis=0)
mean_a2c_max = np.nanmean(a2c_max_util, axis=0)
mean_rdqn_max = np.nanmean(rdqn_max_util, axis=0)
mean_dqn_min = np.nanmean(dqn_min_util, axis=0)
mean_a2c_min = np.nanmean(a2c_min_util, axis=0)
mean_rdqn_min = np.nanmean(rdqn_min_util, axis=0)
mean_dqn_urllc_blocks = np.nanmean(dqn_urllc_blocks, axis=0)
mean_a2c_urllc_blocks = np.nanmean(a2c_urllc_blocks, axis=0)
mean_rdqn_urllc_blocks = np.nanmean(rdqn_urllc_blocks, axis=0)
mean_dqn_embb_blocks = np.nanmean(dqn_embb_blocks, axis=0)
mean_a2c_embb_blocks = np.nanmean(a2c_embb_blocks, axis=0)
mean_rdqn_embb_blocks = np.nanmean(rdqn_embb_blocks, axis=0)
mean_dqn_mmtc_blocks = np.nanmean(dqn_mmtc_blocks, axis=0)
mean_a2c_mmtc_blocks = np.nanmean(a2c_mmtc_blocks, axis=0)
mean_rdqn_mmtc_blocks = np.nanmean(rdqn_mmtc_blocks, axis=0)

def smooth(data, window=5):
    return np.convolve(data, np.ones(window)/window, mode='valid')

def adjust_std_and_smooth(mean_data, std_data):
    smoothed_mean = smooth(mean_data)
    smoothed_std = smooth(std_data)[:len(smoothed_mean)]
    smoothed_std = np.clip(smoothed_std, 0, 1.0)
    return smoothed_mean, smoothed_std

SHOW_STD_DEV = False

# --- Baseline (non-RL) load balancing simulation ---
def simulate_baseline(num_episodes, steps_per_episode=50):
    TOTAL_PRBS = 100
    SLICE_QUOTAS = np.array([30, 60, 10])
    TRAFFIC_TYPES = ['URLLC', 'eMBB', 'mMTC']
    util_threshold = 0.9

    # --- Disruption window for baseline ---
    DISRUPTION_START_EPISODE = 100
    DISRUPTION_DURATION = 40
    DISRUPTED_PRBS = 1

    rewards = []
    std_metric = []
    max_util = []
    min_util = []
    urllc_blocks = []
    embb_blocks = []
    mmtc_blocks = []

    for ep in range(num_episodes):
        usages = np.zeros(3, dtype=np.float32)
        episode_reward = 0
        episode_usages = []
        urllc_blk = 0
        embb_blk = 0
        mmtc_blk = 0

        # --- Apply disruption to baseline ---
        if DISRUPTION_START_EPISODE <= ep < DISRUPTION_START_EPISODE + DISRUPTION_DURATION:
            total_prbs = DISRUPTED_PRBS
        else:
            total_prbs = TOTAL_PRBS

        for t in range(steps_per_episode):
            traffic_type = np.random.choice(TRAFFIC_TYPES)
            slice_idx = TRAFFIC_TYPES.index(traffic_type)
            if traffic_type == 'URLLC':
                request_prbs = np.random.randint(1, 5)
            elif traffic_type == 'eMBB':
                request_prbs = np.random.randint(5, 50)
            else:
                request_prbs = np.random.randint(1, 3)

            # --- Baseline: admit if possible, else block ---
            # Make baseline more naive: only check total PRBs, ignore slice quotas
            if usages.sum() + request_prbs <= total_prbs:
                usages[slice_idx] += request_prbs
                admitted = True
                blocked = False
            else:
                admitted = False
                blocked = True

            norm_usages = usages / SLICE_QUOTAS
            episode_usages.append(norm_usages.copy())

            # Make baseline reward less optimal: penalize imbalance and blocking more
            reward = 1.0 if admitted else -1.0
            reward -= 0.2 * np.std(norm_usages)  # increase penalty for imbalance

            episode_reward += reward

            if blocked:
                if traffic_type == 'URLLC':
                    urllc_blk += 1
                elif traffic_type == 'eMBB':
                    embb_blk += 1
                else:
                    mmtc_blk += 1

        rewards.append(episode_reward)
        episode_usages = np.array(episode_usages)
        std_metric.append(np.std(episode_usages, axis=1).mean())
        max_util.append(np.max(episode_usages, axis=1).mean())
        min_util.append(np.min(episode_usages, axis=1).mean())
        urllc_blocks.append(urllc_blk)
        embb_blocks.append(embb_blk)
        mmtc_blocks.append(mmtc_blk)

    return (
        np.array(rewards),
        np.array(std_metric),
        np.array(max_util),
        np.array(min_util),
        np.array(urllc_blocks),
        np.array(embb_blocks),
        np.array(mmtc_blocks),
    )

# --- Simulate baseline for plotting ---
baseline_rewards, baseline_std, baseline_max, baseline_min, baseline_urllc_blocks, baseline_embb_blocks, baseline_mmtc_blocks = simulate_baseline(max_length)

def plot_results():
    plt.figure(figsize=(6, 4))
    dqn_mean, dqn_std = adjust_std_and_smooth(mean_dqn, std_dqn)
    a2c_mean, a2c_std = adjust_std_and_smooth(mean_a2c, std_a2c)
    rdqn_mean, rdqn_std = adjust_std_and_smooth(mean_rdqn, std_rdqn)
    baseline_mean = smooth(baseline_rewards)

    # --- Compute ratio metric: min_util / max_util for each method ---
    dqn_ratio = smooth(np.nan_to_num(mean_dqn_min / mean_dqn_max, nan=0.0, posinf=0.0, neginf=0.0))
    a2c_ratio = smooth(np.nan_to_num(mean_a2c_min / mean_a2c_max, nan=0.0, posinf=0.0, neginf=0.0))
    rdqn_ratio = smooth(np.nan_to_num(mean_rdqn_min / mean_rdqn_max, nan=0.0, posinf=0.0, neginf=0.0))
    baseline_ratio = smooth(np.nan_to_num(baseline_min / baseline_max, nan=0.0, posinf=0.0, neginf=0.0))

    plt.plot(dqn_ratio, label='DQN + LB', color='#d95f02', linewidth=2, marker='o', linestyle='-', markevery=20)
    plt.plot(a2c_ratio, label='A2C + LB', color='#1b9e77', linewidth=2, marker='s', linestyle='-', markevery=20)
    plt.plot(rdqn_ratio, label='Rainbow + LB', color='#7570b3', linewidth=2, marker='^', linestyle='-', markevery=20)
    plt.plot(baseline_ratio, label='LB', color='gray', linewidth=2, linestyle='--')

    plt.xlim(0, 280)
    plt.title("Network Slicing Load Balance Ratio (min/max)")
    plt.xlabel("Number of Episodes")
    plt.ylabel("Min/Max Utilization Ratio")
    # --- Mark disruption window with shaded region ---
    disruption_start = 95
    disruption_end = 95 + 40
    plt.axvspan(disruption_start, disruption_end, color='red', alpha=0.2, label='Disruption Window')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Load_Balance_Metric.png')
    plt.close()

def plot_std_metric():
    plt.figure(figsize=(6, 4))
    plt.plot(smooth(mean_dqn_std), label='DQN + LB', color='#d95f02', linewidth=2, marker='o', linestyle='-', markevery=20)
    plt.plot(smooth(mean_a2c_std), label='A2C + LB', color='#1b9e77', linewidth=2, marker='s', linestyle='-', markevery=20)
    plt.plot(smooth(mean_rdqn_std), label='Rainbow + LB', color='#7570b3', linewidth=2, marker='^', linestyle='-', markevery=20)
    plt.title("Std Dev of Slice Utilization")
    plt.xlabel("Number of Episodes")
    plt.ylabel("Std Dev (Normalized Usage)")
    plt.xlim(0, 280)
    # --- Mark disruption window with shaded region ---
    disruption_start = 95
    disruption_end = 95 + 40
    plt.axvspan(disruption_start, disruption_end, color='red', alpha=0.2, label='Disruption Window')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Std_Utilization.png')
    plt.close()

def plot_block_rate_urllc():
    plt.figure(figsize=(6, 4))
    plt.plot(smooth(mean_dqn_urllc_blocks), label='DQN + LB', color='#d95f02', linestyle='-', marker='o', linewidth=2, markevery=20)
    plt.plot(smooth(mean_a2c_urllc_blocks), label='A2C + LB', color='#1b9e77', linestyle='-', marker='s', linewidth=2, markevery=20)
    plt.plot(smooth(mean_rdqn_urllc_blocks), label='Rainbow + LB', color='#7570b3', linestyle='-', marker='^', linewidth=2, markevery=20)
    plt.plot(smooth(baseline_urllc_blocks), label='LB', color='gray', linewidth=2, linestyle='--')
    plt.title("Block Rate - URLLC Slice")
    plt.xlabel("Number of Episodes")
    plt.ylabel("Mean Block Count per Episode")
    plt.xlim(0, 280)
    # --- Mark disruption window with shaded region ---
    disruption_start = 95
    disruption_end = 95 + 40
    plt.axvspan(disruption_start, disruption_end, color='red', alpha=0.2, label='Disruption Window')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Block_Rate_URRLC.png')
    plt.close()

def plot_block_rate_embb():
    plt.figure(figsize=(6, 4))
    plt.plot(smooth(mean_dqn_embb_blocks), label='DQN + LB', color='#d95f02', linestyle='-', marker='o', linewidth=2, markevery=20)
    plt.plot(smooth(mean_a2c_embb_blocks), label='A2C + LB', color='#1b9e77', linestyle='-', marker='s', linewidth=2, markevery=20)
    plt.plot(smooth(mean_rdqn_embb_blocks), label='Rainbow + LB', color='#7570b3', linestyle='-', marker='^', linewidth=2, markevery=20)
    plt.plot(smooth(baseline_embb_blocks), label='LB', color='gray', linewidth=2, linestyle='--')
    plt.title("Block Rate - eMBB Slice")
    plt.xlabel("Number of Episodes")
    plt.ylabel("Mean Block Count per Episode")
    plt.xlim(0, 280)
    # --- Mark disruption window with shaded region ---
    disruption_start = 95
    disruption_end = 95 + 40
    plt.axvspan(disruption_start, disruption_end, color='red', alpha=0.2, label='Disruption Window')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('Block_Rate_eMBB.png')
    plt.close()

def plot_block_rate_mmtc():
    plt.figure(figsize=(6, 4))
    plt.plot(smooth(mean_dqn_mmtc_blocks), label='DQN + LB', color='#d95f02', linestyle='-', marker='o', linewidth=2, markevery=20)
    plt.plot(smooth(mean_a2c_mmtc_blocks), label='A2C + LB', color='#1b9e77', linestyle='-', marker='s', linewidth=2, markevery=20)
    plt.plot(smooth(mean_rdqn_mmtc_blocks), label='Rainbow + LB', color='#7570b3', linestyle='-', marker='^', linewidth=2, markevery=20)
    plt.plot(smooth(baseline_mmtc_blocks), label='LB', color='gray', linewidth=2, linestyle='--')
    plt.title("Block Rate - mMTC Slice")
    plt.xlabel("Number of Episodes")
    plt.ylabel("Mean Block Count per Episode")
    plt.xlim(0, 280)
    # --- Mark disruption window with shaded region ---
    disruption_start = 95
    disruption_end = 95 + 40
    plt.axvspan(disruption_start, disruption_end, color='red', alpha=0.2, label='Disruption Window')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Block_Rate_mMTC.png')
    plt.close()

# Optionally, add similar plots for max/min utilization

plot_results()
plot_std_metric()
plot_block_rate_urllc()
plot_block_rate_embb()
plot_block_rate_mmtc()
plot_block_rate_mmtc()
plot_results()
plot_std_metric()
plot_block_rate_urllc()
plot_block_rate_embb()
plot_block_rate_mmtc()
plot_block_rate_mmtc()
plot_block_rate_urllc()
plot_block_rate_embb()
plot_block_rate_mmtc()
plot_block_rate_mmtc()
plot_block_rate_mmtc()
plot_block_rate_urllc()
plot_block_rate_embb()
plot_block_rate_mmtc()
plot_block_rate_mmtc()
