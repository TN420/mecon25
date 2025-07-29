# A2C.py

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque

# ================================
# Configuration Constants
# ================================

# Environment Constants
TOTAL_PRBS = 100
URLLC_QUOTA = 30
EMBB_QUOTA = 60
MMTC_QUOTA = 10

# Traffic Types
TRAFFIC_TYPES = ['URLLC', 'eMBB', 'mMTC']

# A2C Hyperparameters
GAMMA = 0.95
LR = 0.001
EPISODES = 600
STEPS_PER_EPISODE = 50

# ================================
# Disruption Configuration
# ================================
DISRUPTION_WINDOWS = [
    (100, 30),
    (300, 30),
    (500, 30)
]
DISRUPTED_PRBS = 1  # Make disruption extremely severe

# ================================
# Environment
# ================================

class RANEnv:
    def __init__(self):
        self.slice_quotas = np.array([URLLC_QUOTA, EMBB_QUOTA, MMTC_QUOTA])
        self.util_threshold = 0.9
        self.normal_total_prbs = TOTAL_PRBS
        self.reset()

    def reset(self):
        self.usages = np.zeros(3, dtype=np.float32)  # [urllc, embb, mmtc]
        self.total_prbs = self.normal_total_prbs
        return self._get_state()

    def _get_state(self):
        norm_usages = self.usages / self.slice_quotas
        remaining = (self.total_prbs - self.usages.sum()) / TOTAL_PRBS
        return np.concatenate([norm_usages, [remaining]]).astype(np.float32)

    def step(self, action, traffic_type, episode=None, step_num=None):
        # Apply disruption if within any disruption window
        if episode is not None and any(start <= episode < start + duration for start, duration in DISRUPTION_WINDOWS):
            self.total_prbs = DISRUPTED_PRBS
            if step_num == 0:
                print(f"*** DISRUPTION ACTIVE (Episode {episode}) - total_prbs={self.total_prbs} ***")
        else:
            self.total_prbs = self.normal_total_prbs

        done = False
        admitted = False
        blocked = False
        sla_violated = False
        slice_idx = TRAFFIC_TYPES.index(traffic_type)
        ω_k = 1.0  # weight for slice k
        μ = 0.01   # resource cost coefficient
        C_t = 1.0  # cost per PRB
        β = 0.2    # SLA violation penalty coefficient
        λ_k = 0.5  # penalty for blocking

        if traffic_type == 'URLLC':
            request_prbs = np.random.randint(1, 5)
        elif traffic_type == 'eMBB':
            request_prbs = np.random.randint(5, 50)
        else:  # mMTC
            request_prbs = np.random.randint(1, 3)

        R_k = 0.0
        F_t = 0
        prb_cost = 0.0

        if action == 1:
            prb_cost = request_prbs
            if self.usages[slice_idx] + request_prbs <= self.slice_quotas[slice_idx] \
               and self.usages.sum() + request_prbs <= self.total_prbs:
                self.usages[slice_idx] += request_prbs
                admitted = True
                blocked = False
                sla_violated = False
                R_k = 1.0
            else:
                admitted = False
                blocked = True
                sla_violated = True
                R_k = 0.0
                F_t = 1
        else:
            admitted = False
            blocked = False
            sla_violated = False
            R_k = 0.0

        norm_usages = self.usages / self.slice_quotas

        # --- New reward function ---
        reward = ω_k * R_k - λ_k * blocked - μ * C_t * prb_cost - β * F_t

        next_state = self._get_state()
        return next_state, reward, done, admitted, blocked, sla_violated, traffic_type

# ================================
# A2C Networks
# ================================

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# ================================
# Results Saving Function
# ================================

def save_results(run_id, rewards, urllc_blocks, embb_blocks, mmtc_blocks, urllc_sla, embb_sla, mmtc_sla, stds, max_utils, min_utils):
    # Ensure the results folder exists
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save results with a unique run ID to avoid overwriting
    np.savez(os.path.join(results_dir, f"a2c_results_run_{run_id}.npz"), 
             rewards=rewards, 
             urllc_blocks=urllc_blocks, 
             embb_blocks=embb_blocks, 
             mmtc_blocks=mmtc_blocks,
             urllc_sla=urllc_sla, 
             embb_sla=embb_sla,
             mmtc_sla=mmtc_sla,
             std=stds,
             max_util=max_utils,
             min_util=min_utils)

# ================================
# Training Loop
# ================================

def train_a2c(episodes=EPISODES, run_id=1):
    env = RANEnv()
    state_size = len(env.reset())
    action_size = 2

    actor = Actor(state_size, action_size)
    critic = Critic(state_size)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=LR)

    reward_history = []
    urllc_block_history = []
    embb_block_history = []
    mmtc_block_history = []
    urllc_sla_pres = []
    embb_sla_pres = []
    mmtc_sla_pres = []
    urllc_usage_hist = []
    embb_usage_hist = []
    actor_losses = []
    critic_losses = []
    std_history = []
    max_util_history = []
    min_util_history = []

    for episode in range(episodes):
        state = env.reset()
        episode_return = 0
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
            if traffic_type == 'URLLC':
                urllc_total_requests += 1
            elif traffic_type == 'eMBB':
                embb_total_requests += 1
            else:
                mmtc_total_requests += 1

            state_tensor = torch.tensor(state).float().unsqueeze(0)
            action_probs = actor(state_tensor)
            action = np.random.choice(action_size, p=action_probs.detach().numpy().squeeze())

            next_state, reward, done, admitted, blocked, sla_violated, t_type = env.step(
                action, traffic_type, episode=episode, step_num=t
            )

            # Track normalized usages for metrics
            norm_usages = env.usages / env.slice_quotas
            episode_usages.append(norm_usages.copy())

            if blocked:
                if t_type == 'URLLC': urllc_blocks += 1
                elif t_type == 'eMBB': embb_blocks += 1
                elif t_type == 'mMTC': mmtc_blocks += 1
            if not sla_violated:
                if t_type == 'URLLC': urllc_sla_preserved += 1
                elif t_type == 'eMBB': embb_sla_preserved += 1
                elif t_type == 'mMTC': mmtc_sla_preserved += 1

            state_value = critic(state_tensor)
            next_state_value = critic(torch.tensor(next_state).float().unsqueeze(0))
            advantage = reward + GAMMA * next_state_value - state_value

            actor_loss = -torch.log(action_probs.squeeze(0)[action]) * advantage.detach()
            critic_loss = advantage.pow(2)

            optimizer.zero_grad()
            (actor_loss + critic_loss).mean().backward()
            optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

            state = next_state
            episode_return += reward

        reward_history.append(episode_return)
        urllc_block_history.append(urllc_blocks)
        embb_block_history.append(embb_blocks)
        mmtc_block_history.append(mmtc_blocks)
        urllc_sla_pres.append(urllc_sla_preserved / urllc_total_requests if urllc_total_requests > 0 else 0)
        embb_sla_pres.append(embb_sla_preserved / embb_total_requests if embb_total_requests > 0 else 0)
        mmtc_sla_pres.append(mmtc_sla_preserved / mmtc_total_requests if mmtc_total_requests > 0 else 0)
        urllc_usage_hist.append(env.usages[0])
        embb_usage_hist.append(env.usages[1])

        # Compute and store std, max, min utilization for this episode
        episode_usages = np.array(episode_usages)
        std_history.append(np.std(episode_usages, axis=1).mean())
        max_util_history.append(np.max(episode_usages, axis=1).mean())
        min_util_history.append(np.min(episode_usages, axis=1).mean())

        print(f"Episode {episode+1}/{episodes} - Episode Return: {episode_return}")

    # Save results for the current run with a unique ID
    save_results(run_id, reward_history, urllc_block_history, embb_block_history, mmtc_block_history, urllc_sla_pres, embb_sla_pres, mmtc_sla_pres, std_history, max_util_history, min_util_history)

# ================================
# Run Multiple A2C Simulations
# ================================

for run_id in range(1, 11):  # Run 10 simulations with different IDs
    train_a2c(episodes=EPISODES, run_id=run_id)
