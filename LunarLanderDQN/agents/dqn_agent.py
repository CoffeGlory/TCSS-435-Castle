import random
import numpy as np
import torch
import torch.optim as optim

from networks.q_network import QNetwork
from utils.replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, device: torch.device = torch.device("cpu")):        
        self.device = device
        self.action_dim = action_dim

        # Initialize Q-network and target network
        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)

        # Replay buffer
        self.replay_buffer = ReplayBuffer()

        # Training parameters
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.update_target_every = 1000
        self.train_step = 0

    def select_action(self, state: np.ndarray) -> int:
        """Select an action using epsilon-greedy strategy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def update(self):
        """Sample from replay buffer and train Q-network"""
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute current Q-values
        current_q = self.q_network(states).gather(1, actions)

        # Compute target Q-values using target network
        #with torch.no_grad():
        #    next_q = self.target_network(next_states).max(1, keepdim=True)[0]
        #    expected_q = rewards + (1 - dones) * self.gamma * next_q
     
        # Double DQN target Q-values
        with torch.no_grad():
            # Action selection using online network
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            # Action evaluation using target network
            next_q = self.target_network(next_states).gather(1, next_actions)
            expected_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute MSE loss
        loss = torch.nn.functional.mse_loss(current_q, expected_q)

        # Optimize Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically update target network
        self.train_step += 1
        if self.train_step % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    
    def decay_epsilon(self) -> None:
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)