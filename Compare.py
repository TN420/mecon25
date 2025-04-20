# Compare.py

import numpy as np
import matplotlib.pyplot as plt

# ================================
# Load Results
# ================================

# Load DQN and A2C results from the saved .npz files
dqn_results = np.load('dqn_results.npz')
a2c_results = np.load('a2c_results.npz')

# Extract metrics
dqn_urllc_blocks = dqn_results['urllc_blocks']
a2c_urllc_blocks = a2c_results['urllc_blocks']
dqn_embb_blocks = dqn_results['embb_blocks']
a2c_embb_blocks = a2c_results['embb_blocks']
dqn_urllc_sla = dqn_results['urllc_sla']
a2c_urllc_sla = a2c_results['urllc_sla']
dqn_embb_sla = dqn_results['embb_sla']
a2c_embb_sla = a2c_results['embb_sla']

# ================================
# Smoothing Helper
# ================================

def smooth(data, window=20):
    return np.convolve(data, np.ones(window)/window, mode='valid')

# ================================
# Plot SLA and Block Rate by Slice Type
# ================================

fig, axs = plt.subplots(2, 2, figsize=(14, 8))

# URLLC Block Rate
axs[0, 0].plot(smooth(dqn_urllc_blocks), label="DQN", alpha=0.8)
axs[0, 0].plot(smooth(a2c_urllc_blocks), label="A2C", alpha=0.8)
axs[0, 0].set_title("URLLC Block Rate (Smoothed)")
axs[0, 0].set_ylabel("Block Ratio")
axs[0, 0].legend()
axs[0, 0].grid(True)

# eMBB Block Rate
axs[0, 1].plot(smooth(dqn_embb_blocks), label="DQN", alpha=0.8)
axs[0, 1].plot(smooth(a2c_embb_blocks), label="A2C", alpha=0.8)
axs[0, 1].set_title("eMBB Block Rate (Smoothed)")
axs[0, 1].legend()
axs[0, 1].grid(True)

# URLLC SLA
axs[1, 0].plot(smooth(dqn_urllc_sla), label="DQN", alpha=0.8)
axs[1, 0].plot(smooth(a2c_urllc_sla), label="A2C", alpha=0.8)
axs[1, 0].set_title("URLLC SLA Preservation (Smoothed)")
axs[1, 0].set_ylabel("SLA Ratio")
axs[1, 0].set_ylim(0, 1.05)
axs[1, 0].legend()
axs[1, 0].grid(True)

# eMBB SLA
axs[1, 1].plot(smooth(dqn_embb_sla), label="DQN", alpha=0.8)
axs[1, 1].plot(smooth(a2c_embb_sla), label="A2C", alpha=0.8)
axs[1, 1].set_title("eMBB SLA Preservation (Smoothed)")
axs[1, 1].set_ylim(0, 1.05)
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.savefig("benchmark.png")
plt.show()
