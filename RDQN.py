# RDQN.py

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import time  # Added for timing
import csv  # Added for CSV writing

# ================================
# Configuration Constants
# ================================

TOTAL_PRBS = 100
URLLC_QUOTA = 30
EMBB_QUOTA = 70

TRAFFIC_TYPES = ['URLLC', 'eMBB']

GAMMA = 0.95  # Default, but will be overridden in main loop
LR_VALUES = [0.0005, 0.001, 0.0015]
BATCH_SIZE = 16
MEMORY_SIZE = 5000
TARGET_UPDATE = 10
EPISODES = 300
STEPS_PER_EPISODE = 50
ALPHA = 0.6
BETA_START = 0.4

# ================================
# Environment
# ================================

class RANEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.urllc_usage = 0
        self.embb_usage = 0
        self.total_prbs = TOTAL_PRBS
        self.state = self._get_state()
        return self.state

    def _get_state(self):
        return np.array([
            self.urllc_usage / TOTAL_PRBS,
            self.embb_usage / TOTAL_PRBS,
            1.0,
            1.0
        ], dtype=np.float32)

    def step(self, action, traffic_type):
        reward = 0
        done = False
        admitted = False
        blocked = False
        sla_violated = False

        request_prbs = np.random.randint(1, 5) if traffic_type == 'URLLC' else np.random.randint(5, 50)

        if action == 1:
            if traffic_type == 'URLLC':
                if self.urllc_usage + request_prbs <= URLLC_QUOTA:
                    self.urllc_usage += request_prbs
                    reward = 1
                    admitted = True
                else:
                    reward = -1
                    blocked = True
                    sla_violated = True
            elif traffic_type == 'eMBB':
                if self.embb_usage + request_prbs <= EMBB_QUOTA:
                    self.embb_usage += request_prbs
                    reward = 1
                    admitted = True
                elif self.urllc_usage + self.embb_usage + request_prbs <= TOTAL_PRBS:
                    self.embb_usage += request_prbs
                    reward = -0.5
                    admitted = True
                    sla_violated = True
                else:
                    reward = -1
                    blocked = True
                    sla_violated = True
        else:
            reward = 0.1 if not admitted else -0.2

        next_state = self._get_state()
        return next_state, reward, done, admitted, blocked, sla_violated, traffic_type

# ================================
# Rainbow DQN Network
# ================================

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)

class RainbowDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(RainbowDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.value_stream = nn.Sequential(
            NoisyLinear(128, 64),
            nn.ReLU(),
            NoisyLinear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(128, 64),
            nn.ReLU(),
            NoisyLinear(64, action_size)
        )

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

# ================================
# Prioritized Replay Buffer
# ================================

class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, state, action, reward, next_state):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state)
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=BETA_START):
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** ALPHA
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        states, actions, rewards, next_states = map(np.array, batch)

        return states, actions, rewards, next_states, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-5

    def __len__(self):
        return len(self.buffer)

# ================================
# Training Loop
# ================================

def train_dqn(episodes=EPISODES, run_id=1, lr=0.001):
    start_time = time.time()  # Start timing
    env = RANEnv()
    state_size = len(env.reset())
    action_size = 2

    policy_net = RainbowDQN(state_size, action_size)
    target_net = RainbowDQN(state_size, action_size)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = PrioritizedReplayBuffer(MEMORY_SIZE)

    reward_history = []
    urllc_block_history = []
    embb_block_history = []
    urllc_sla_pres = []
    embb_sla_pres = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        urllc_blocks = 0
        embb_blocks = 0
        urllc_sla_preserved = 0
        embb_sla_preserved = 0
        urllc_total_requests = 0
        embb_total_requests = 0

        beta = min(1.0, BETA_START + episode * (1.0 - BETA_START) / episodes)

        for t in range(STEPS_PER_EPISODE):
            traffic_type = random.choice(TRAFFIC_TYPES)
            urllc_total_requests += traffic_type == 'URLLC'
            embb_total_requests += traffic_type == 'eMBB'

            with torch.no_grad():
                q_vals = policy_net(torch.tensor(state).float().unsqueeze(0))
                action = q_vals.argmax().item()

            next_state, reward, done, admitted, blocked, sla_violated, t_type = env.step(action, traffic_type)

            if blocked:
                urllc_blocks += t_type == 'URLLC'
                embb_blocks += t_type == 'eMBB'
            if not sla_violated:
                urllc_sla_preserved += t_type == 'URLLC'
                embb_sla_preserved += t_type == 'eMBB'

            memory.push(state, action, reward, next_state)
            state = next_state
            total_reward += reward

            if len(memory) >= BATCH_SIZE:
                s, a, r, ns, indices, weights = memory.sample(BATCH_SIZE, beta)
                s = torch.tensor(s).float()
                a = torch.tensor(a).long()
                r = torch.tensor(r).float()
                ns = torch.tensor(ns).float()
                weights = torch.tensor(weights).float()

                q_values = policy_net(s).gather(1, a.unsqueeze(1)).squeeze()
                next_q_values = target_net(ns).max(1)[0].detach()
                expected_q = r + GAMMA * next_q_values

                td_errors = expected_q - q_values
                loss = (td_errors.pow(2) * weights).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                policy_net.reset_noise()
                memory.update_priorities(indices, td_errors.detach().numpy())

        reward_history.append(total_reward)
        urllc_block_history.append(urllc_blocks)
        embb_block_history.append(embb_blocks)
        urllc_sla_pres.append(urllc_sla_preserved / urllc_total_requests if urllc_total_requests > 0 else 0)
        embb_sla_pres.append(embb_sla_preserved / embb_total_requests if embb_total_requests > 0 else 0)

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode+1}/{episodes} - Total Reward: {total_reward}")

    elapsed_time = time.time() - start_time  # End timing
    print(f"RDQN Training Time for Run {run_id}: {elapsed_time:.2f} seconds")

    # Save training time to an RDQN-specific CSV file (lr in filename)
    lr_str = str(lr).replace('.', '_')
    times_csv = f"training_times_rdqn_lr_{lr_str}.csv"
    with open(times_csv, mode="a", newline="") as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["Run_ID", "Training_Time", "Learning_Rate", "Batch_Size", "Memory_Size", "Alpha", "Beta_Start"])
        writer.writerow([run_id, elapsed_time, lr, BATCH_SIZE, MEMORY_SIZE, ALPHA, BETA_START])

    os.makedirs(f"results/results_rdqn_lr_{lr_str}", exist_ok=True)
    np.savez(f"results/results_rdqn_lr_{lr_str}/rdqn_results_run_{run_id}_lr_{lr_str}.npz",
             rewards=reward_history,
             urllc_blocks=urllc_block_history,
             embb_blocks=embb_block_history,
             urllc_sla=urllc_sla_pres,
             embb_sla=embb_sla_pres)

if __name__ == "__main__":
    # gamma_values = [0.9, GAMMA, 0.99]
    for lr in LR_VALUES:
        for run_id in range(1, 6):
            train_dqn(episodes=EPISODES, run_id=run_id, lr=lr)

        # Calculate averages for each lr
        lr_str = str(lr).replace('.', '_')
        rdqn_times = []
        times_csv = f"training_times_rdqn_lr_{lr_str}.csv"
        if os.path.exists(times_csv):
            with open(times_csv, mode="r") as file:
                reader = csv.reader(file)
                for row in reader:
                    if row[0].isdigit():
                        rdqn_times.append(float(row[1]))
        average_time = np.mean(rdqn_times) if rdqn_times else 0
        with open(times_csv, mode="a", newline="") as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(["Run_ID", "Training_Time", "Learning_Rate", "Batch_Size", "Memory_Size", "Alpha", "Beta_Start"])
            writer.writerow(["Average", average_time, lr, BATCH_SIZE, MEMORY_SIZE, ALPHA, BETA_START])
