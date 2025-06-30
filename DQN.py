# DQN.py

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import time

# ================================
# Configuration Constants
# ================================

TOTAL_PRBS = 100
URLLC_QUOTA = 30
EMBB_QUOTA = 60
MMTC_QUOTA = 10

TRAFFIC_TYPES = ['URLLC', 'eMBB', 'mMTC']

GAMMA = 0.95
LR = 0.001
BATCH_SIZE = 16
MEMORY_SIZE = 5000
TARGET_UPDATE = 10
EPISODES = 300
STEPS_PER_EPISODE = 50
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 100

# ================================
# Disruption Configuration
# ================================
DISRUPTION_START_EPISODE = 100
DISRUPTION_DURATION = 40
DISRUPTED_PRBS = 1  # Make disruption extremely severe

# ================================
# Environment
# ================================

class RANEnv:
    def __init__(self):
        self.slice_quotas = np.array([URLLC_QUOTA, EMBB_QUOTA, MMTC_QUOTA])
        self.util_threshold = 0.9  # Utilization threshold for positive reward
        self.normal_total_prbs = TOTAL_PRBS
        self.reset()

    def reset(self):
        self.usages = np.zeros(3, dtype=np.float32)  # [urllc, embb, mmtc]
        self.total_prbs = self.normal_total_prbs
        return self._get_state()

    def _get_state(self):
        # State: normalized usages for each slice, and remaining PRBs
        norm_usages = self.usages / self.slice_quotas
        remaining = (self.total_prbs - self.usages.sum()) / TOTAL_PRBS
        return np.concatenate([norm_usages, [remaining]]).astype(np.float32)

    def step(self, action, traffic_type, episode=None, step_num=None):
        # Apply disruption if within the disruption window
        if episode is not None and DISRUPTION_START_EPISODE <= episode < DISRUPTION_START_EPISODE + DISRUPTION_DURATION:
            self.total_prbs = DISRUPTED_PRBS
            # Debug print to confirm disruption is active
            if step_num == 0:
                print(f"*** DISRUPTION ACTIVE (Episode {episode}) - total_prbs={self.total_prbs} ***")
        else:
            self.total_prbs = self.normal_total_prbs

        done = False
        admitted = False
        blocked = False
        sla_violated = False
        slice_idx = TRAFFIC_TYPES.index(traffic_type)
        # Define PRB request ranges for each slice
        if traffic_type == 'URLLC':
            request_prbs = np.random.randint(1, 5)
        elif traffic_type == 'eMBB':
            request_prbs = np.random.randint(5, 50)
        else:  # mMTC
            request_prbs = np.random.randint(1, 3)

        if action == 1:
            # Try to admit the request to the corresponding slice
            if self.usages[slice_idx] + request_prbs <= self.slice_quotas[slice_idx] \
               and self.usages.sum() + request_prbs <= self.total_prbs:
                self.usages[slice_idx] += request_prbs
                admitted = True
            else:
                blocked = True
                sla_violated = True  # Blocked means SLA not met

        norm_usages = self.usages / self.slice_quotas

        # --- Modified reward function ---
        reward = 0.0
        if admitted:
            reward += 1.0  # Strong reward for admitting
        if blocked:
            reward -= 0.5  # Penalty for blocking
        reward -= 0.1 * np.std(norm_usages)  # Smaller penalty for imbalance
        if np.all(norm_usages < self.util_threshold):
            reward += 0.1  # Small bonus for keeping all slices under threshold
        if sla_violated or np.any(norm_usages > 1.0):
            reward -= 0.1  # Penalty for SLA violation/overload

        next_state = self._get_state()
        return next_state, reward, done, admitted, blocked, sla_violated, traffic_type

# ================================
# DQN Network
# ================================

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# ================================
# Standard Replay Buffer
# ================================

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        batch = list(zip(*samples))
        states, actions, rewards, next_states = map(np.array, batch)
        return states, actions, rewards, next_states

    def __len__(self):
        return len(self.buffer)

# ================================
# Training Loop
# ================================

def train_dqn(episodes=EPISODES, run_id=1):
    env = RANEnv()
    state_size = len(env.reset())
    action_size = 2

    policy_net = DQN(state_size, action_size)
    target_net = DQN(state_size, action_size)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)

    reward_history = []
    urllc_block_history = []
    embb_block_history = []
    mmtc_block_history = []
    urllc_sla_pres = []
    embb_sla_pres = []
    mmtc_sla_pres = []
    std_history = []      # Track std deviation of normalized usages
    max_util_history = [] # Track max utilization
    min_util_history = [] # Track min utilization

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        urllc_blocks = 0
        embb_blocks = 0
        mmtc_blocks = 0
        urllc_sla_preserved = 0
        embb_sla_preserved = 0
        mmtc_sla_preserved = 0
        urllc_total_requests = 0
        embb_total_requests = 0
        mmtc_total_requests = 0

        episode_usages = []

        for t in range(STEPS_PER_EPISODE):
            traffic_type = random.choice(TRAFFIC_TYPES)
            urllc_total_requests += traffic_type == 'URLLC'
            embb_total_requests += traffic_type == 'eMBB'
            mmtc_total_requests += traffic_type == 'mMTC'

            epsilon = max(EPSILON_END, EPSILON_START - episode / EPSILON_DECAY)
            if random.random() < epsilon:
                action = random.randint(0, 1)
            else:
                with torch.no_grad():
                    q_vals = policy_net(torch.tensor(state).float().unsqueeze(0))
                    action = q_vals.argmax().item()

            next_state, reward, done, admitted, blocked, sla_violated, t_type = env.step(
                action, traffic_type, episode=episode, step_num=t
            )

            # Track normalized usages for metrics
            norm_usages = env.usages / env.slice_quotas
            episode_usages.append(norm_usages.copy())

            if blocked:
                urllc_blocks += t_type == 'URLLC'
                embb_blocks += t_type == 'eMBB'
                mmtc_blocks += t_type == 'mMTC'
            if not sla_violated:
                urllc_sla_preserved += t_type == 'URLLC'
                embb_sla_preserved += t_type == 'eMBB'
                mmtc_sla_preserved += t_type == 'mMTC'

            memory.push(state, action, reward, next_state)
            state = next_state
            total_reward += reward

            if len(memory) >= BATCH_SIZE:
                s, a, r, ns = memory.sample(BATCH_SIZE)
                s = torch.tensor(s).float()
                a = torch.tensor(a).long()
                r = torch.tensor(r).float()
                ns = torch.tensor(ns).float()

                q_values = policy_net(s).gather(1, a.unsqueeze(1)).squeeze()
                next_q_values = target_net(ns).max(1)[0].detach()
                expected_q = r + GAMMA * next_q_values

                loss = nn.MSELoss()(q_values, expected_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        reward_history.append(total_reward)
        urllc_block_history.append(urllc_blocks)
        embb_block_history.append(embb_blocks)
        mmtc_block_history.append(mmtc_blocks)
        urllc_sla_pres.append(urllc_sla_preserved / urllc_total_requests if urllc_total_requests > 0 else 0)
        embb_sla_pres.append(embb_sla_preserved / embb_total_requests if embb_total_requests > 0 else 0)
        mmtc_sla_pres.append(mmtc_sla_preserved / mmtc_total_requests if mmtc_total_requests > 0 else 0)

        # Compute and store std, max, min utilization for this episode
        episode_usages = np.array(episode_usages)
        std_history.append(np.std(episode_usages, axis=1).mean())
        max_util_history.append(np.max(episode_usages, axis=1).mean())
        min_util_history.append(np.min(episode_usages, axis=1).mean())

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode+1}/{episodes} - Total Reward: {total_reward}")

    os.makedirs("results", exist_ok=True)
    np.savez(f"results/dqn_results_run_{run_id}.npz",
             rewards=reward_history,
             urllc_blocks=urllc_block_history,
             embb_blocks=embb_block_history,
             mmtc_blocks=mmtc_block_history,
             urllc_sla=urllc_sla_pres,
             embb_sla=embb_sla_pres,
             mmtc_sla=mmtc_sla_pres,
             std=std_history,
             max_util=max_util_history,
             min_util=min_util_history)

for run_id in range(1, 2):
    train_dqn(episodes=EPISODES, run_id=run_id)
    train_dqn(episodes=EPISODES, run_id=run_id)
