"""Agente DQN para Pendulum-v1 y entornos discretos.

Mantiene una API similar a ``PendulumSarsaAgent``: entrenamiento por episodios,
registro de recompensas, longitudes y pérdidas TD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.dqn import dqnupdate


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Buffer FIFO para experiencia de replay."""

    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = int(capacity)
        self.buffer: List[Transition] = []
        self.position = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        transition = Transition(state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int = 32) -> Tuple[np.ndarray, ...]:
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    """Red MLP para aproximar Q(s, a)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Sequence[int] = (128, 128)) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        dims = [state_dim, *hidden_dims, action_dim]
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PendulumDqnAgent:
    """Agente DQN con replay buffer y red target.

    Parámetros por defecto de la práctica:
      - gamma=0.99
      - lr=1e-3
      - replay capacity=10000
      - batch_size=32
      - epsilon: 0.2 -> 0.01
      - target update cada 100 episodios
    """

    def __init__(
        self,
        env,
        seed: int = 2024,
        gamma: float = 0.99,
        lr: float = 1e-3,
        epsilon_start: float = 0.2,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.9995,
        replay_capacity: int = 10000,
        batch_size: int = 32,
        target_update_episodes: int = 100,
        action_bins: int = 11,
        device: str | None = None,
    ) -> None:
        self.env = env
        self.seed = seed
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_episodes = target_update_episodes

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.state_dim = int(np.prod(self.env.observation_space.shape))

        # Pendulum es continuo: discretizamos acciones.
        if hasattr(self.env.action_space, "n"):
            self.discrete_actions = np.arange(self.env.action_space.n, dtype=np.int64)
            self.continuous_action_values = None
            self.action_dim = self.env.action_space.n
        else:
            low = float(self.env.action_space.low[0])
            high = float(self.env.action_space.high[0])
            self.continuous_action_values = np.linspace(low, high, action_bins, dtype=np.float32)
            self.discrete_actions = np.arange(action_bins, dtype=np.int64)
            self.action_dim = action_bins

        self.q_network = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)

        self.rewards_history: List[float] = []
        self.lengths_history: List[int] = []
        self.loss_history: List[float] = []

    def _state_to_tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _to_env_action(self, action_idx: int):
        if self.continuous_action_values is None:
            return int(action_idx)
        return np.array([self.continuous_action_values[action_idx]], dtype=np.float32)

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if (not greedy) and (np.random.rand() < self.epsilon):
            return int(np.random.choice(self.discrete_actions))

        with torch.no_grad():
            q_values = self.q_network(self._state_to_tensor(state))
            action = int(torch.argmax(q_values, dim=1).item())
        return action

    def learn_step(self) -> float:
        if len(self.replay_buffer) < self.batch_size:
            return float("nan")

        batch = self.replay_buffer.sample(self.batch_size)
        loss = dqnupdate(
            q_network=self.q_network,
            target_network=self.target_network,
            optimizer=self.optimizer,
            batch=batch,
            gamma=self.gamma,
            device=self.device,
        )
        return float(loss)

    def train(self, num_episodes: int = 10000, max_steps: int = 200):
        for episode in range(1, num_episodes + 1):
            state, _ = self.env.reset(seed=self.seed + episode)
            episode_reward = 0.0
            episode_loss = []

            for step in range(1, max_steps + 1):
                action_idx = self.select_action(state)
                env_action = self._to_env_action(action_idx)

                next_state, reward, terminated, truncated, _ = self.env.step(env_action)
                done = terminated or truncated

                self.replay_buffer.add(state, action_idx, reward, next_state, done)
                loss = self.learn_step()
                if not np.isnan(loss):
                    episode_loss.append(loss)

                state = next_state
                episode_reward += reward

                if done:
                    break

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if episode % self.target_update_episodes == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            self.rewards_history.append(float(episode_reward))
            self.lengths_history.append(step)
            self.loss_history.append(float(np.mean(episode_loss)) if episode_loss else np.nan)

        return {
            "rewards": self.rewards_history,
            "lengths": self.lengths_history,
            "losses": self.loss_history,
        }

    def evaluate(self, num_episodes: int = 20, max_steps: int = 200) -> float:
        scores = []
        for episode in range(num_episodes):
            state, _ = self.env.reset(seed=self.seed + 10_000 + episode)
            total_reward = 0.0
            for _ in range(max_steps):
                action_idx = self.select_action(state, greedy=True)
                state, reward, terminated, truncated, _ = self.env.step(self._to_env_action(action_idx))
                total_reward += reward
                if terminated or truncated:
                    break
            scores.append(total_reward)
        return float(np.mean(scores))
