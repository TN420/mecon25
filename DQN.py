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
import time  # Added for timing
import csv  # Added for CSV writing

# ================================
# Configuration Constants
# ================================

TOTAL_PRBS = 100
URLLC_QUOTA = 30
EMBB_QUOTA = 70

TRAFFIC_TYPES = ['URLLC', 'eMBB']

GAMMA = 0.95
LR_VALUES = [0.0005, 0.001, 0.0015]
BATCH_SIZE = 16
MEMORY_SIZE = 5000
TARGET_UPDATE = 10
EPISODES = 300
STEPS_PER_EPISODE = 50
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 100

# ================================
# Environment
# ================================

class RANEnv:
    def __init__(self):
        # Parameters for reward/state
        self.B = TOTAL_PRBS
        self.L_max = 100  # Example max queue length
        self.mu = 0.1     # Congestion penalty
        self.w = {'URLLC': 2.0, 'eMBB': 1.0}  # Example weights
        self.R = {'URLLC': 2.0, 'eMBB': 1.0}  # Example rewards
        self.lam = {'URLLC': 1.0, 'eMBB': 0.5}  # SLA violation penalty

        self.reset()

    def reset(self):
        self.urllc_usage = 0
        self.embb_usage = 0
        self.Ct = 0.0  # Congestion metric
        self.L_urllc = 0  # URLLC queue length
        self.L_embb = 0   # eMBB queue length
        self.state = self._get_state()
        return self.state

    def _get_state(self):
        # Equation (1)
        return np.array([
            self.urllc_usage / self.B,
            self.embb_usage / self.B,
            self.Ct,
            self.L_urllc / self.L_max,
            self.L_embb / self.L_max
        ], dtype=np.float32)

    def step(self, action, traffic_type):
        # action: 1=accept, 0=reject
        # traffic_type: 'URLLC' or 'eMBB'
        admitted = False
        blocked = False
        sla_violated = False

        # Simulate request PRBs and queue increments
        if traffic_type == 'URLLC':
            request_prbs = np.random.randint(1, 5)
        else:
            request_prbs = np.random.randint(5, 50)

        # Track accepted requests for reward
        accepted = 0

        # Simulate queue arrival
        if traffic_type == 'URLLC':
            self.L_urllc = min(self.L_urllc + 1, self.L_max)
        else:
            self.L_embb = min(self.L_embb + 1, self.L_max)

        if action == 1:
            # Accept request if resources available
            if traffic_type == 'URLLC':
                if self.urllc_usage + request_prbs <= URLLC_QUOTA:
                    self.urllc_usage += request_prbs
                    admitted = True
                    accepted = 1
                    self.L_urllc = max(self.L_urllc - 1, 0)
                else:
                    blocked = True
                    sla_violated = True
            elif traffic_type == 'eMBB':
                if self.embb_usage + request_prbs <= EMBB_QUOTA:
                    self.embb_usage += request_prbs
                    admitted = True
                    accepted = 1
                    self.L_embb = max(self.L_embb - 1, 0)
                elif self.urllc_usage + self.embb_usage + request_prbs <= TOTAL_PRBS:
                    self.embb_usage += request_prbs
                    admitted = True
                    accepted = 1
                    sla_violated = True
                    self.L_embb = max(self.L_embb - 1, 0)
                else:
                    blocked = True
                    sla_violated = True
        # else: reject, nothing changes except queue

        # Update congestion metric (example: normalized total usage)
        self.Ct = (self.urllc_usage + self.embb_usage) / self.B

        # Compute reward using Equation (2)
        reward = (
            self.w[traffic_type] * self.R[traffic_type]
            - (self.lam[traffic_type] if sla_violated else 0)
            - self.mu * self.Ct * accepted
        )

        next_state = self._get_state()
        done = False  # No terminal state in this setup
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

def train_dqn(episodes=EPISODES, run_id=1, lr=0.001):
    start_time = time.time()  # Start timing
    env = RANEnv()
    state_size = len(env.reset())
    action_size = 2

    policy_net = DQN(state_size, action_size)
    target_net = DQN(state_size, action_size)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = ReplayBuffer(MEMORY_SIZE)

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

        for t in range(STEPS_PER_EPISODE):
            traffic_type = random.choice(TRAFFIC_TYPES)
            urllc_total_requests += traffic_type == 'URLLC'
            embb_total_requests += traffic_type == 'eMBB'

            epsilon = max(EPSILON_END, EPSILON_START - episode / EPSILON_DECAY)
            if random.random() < epsilon:
                action = random.randint(0, 1)
            else:
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
        urllc_sla_pres.append(urllc_sla_preserved / urllc_total_requests if urllc_total_requests > 0 else 0)
        embb_sla_pres.append(embb_sla_preserved / embb_total_requests if embb_total_requests > 0 else 0)

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode+1}/{episodes} - Total Reward: {total_reward}")

    elapsed_time = time.time() - start_time  # End timing
    print(f"DQN Training Time for Run {run_id} (lr={lr}): {elapsed_time:.2f} seconds")

    # Save training time to a DQN-specific CSV file (lr in filename)
    lr_str = str(lr).replace('.', '_')
    times_csv = f"training_times_dqn_lr_{lr_str}.csv"
    with open(times_csv, mode="a", newline="") as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # Add headers if the file is empty
            writer.writerow(["Run_ID", "Training_Time", "Learning_Rate", "Batch_Size", "Memory_Size"])
        writer.writerow([run_id, elapsed_time, lr, BATCH_SIZE, MEMORY_SIZE])

    os.makedirs(f"results/results_dqn_lr_{lr_str}", exist_ok=True)
    np.savez(f"results/results_dqn_lr_{lr_str}/dqn_results_run_{run_id}_lr_{lr_str}.npz",
             rewards=reward_history,
             urllc_blocks=urllc_block_history,
             embb_blocks=embb_block_history,
             urllc_sla=urllc_sla_pres,
             embb_sla=embb_sla_pres)

if __name__ == "__main__":
    # gamma_values = [0.9, GAMMA, 0.99]
    for lr in LR_VALUES:
        for run_id in range(1, 6):  # Run [X] simulations with different IDs
            train_dqn(episodes=EPISODES, run_id=run_id, lr=lr)

        # Calculate averages for each lr
        lr_str = str(lr).replace('.', '_')
        dqn_times = []
        times_csv = f"training_times_dqn_lr_{lr_str}.csv"
        if os.path.exists(times_csv):
            with open(times_csv, mode="r") as file:
                reader = csv.reader(file)
                for row in reader:
                    if row[0].isdigit():  # Only consider rows with run IDs
                        dqn_times.append(float(row[1]))

        average_time = np.mean(dqn_times) if dqn_times else 0
        with open(times_csv, mode="a", newline="") as file:
            writer = csv.writer(file)
            if file.tell() == 0:  # Add headers if the file is empty
                writer.writerow(["Run_ID", "Training_Time", "Learning_Rate", "Batch_Size", "Memory_Size"])
            writer.writerow(["Average", average_time, lr, BATCH_SIZE, MEMORY_SIZE])
