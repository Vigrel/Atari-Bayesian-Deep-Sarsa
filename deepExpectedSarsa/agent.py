from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch import nn


@dataclass
class EnvArgs:
    batch_size: int
    buffer_size: int
    discount_factor: float
    learning_rate: float


class Agent:
    def __init__(self, env: gym.Env, args: EnvArgs) -> None:
        self.env = env
        self.num_actions = env.action_space.n
        self.num_state = len(env.observation_space.sample())
        self.model = self._build_model()
        self.args = args
        self.replay_buffer = deque(maxlen=args.buffer_size)

    def _build_model(self) -> nn.Module:
        raise NotImplementedError("Subclasses must implement _build_model method")

    def select_action(self, state, epsilon: float = 0) -> int:
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)
        q_values = (
            self.model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            .detach()
            .numpy()[0]
        )
        return np.argmax(q_values)

    def sample_experiences(self):
        indices = np.random.randint(len(self.replay_buffer), size=self.args.batch_size)
        batch = [self.replay_buffer[index] for index in indices]
        return [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(6)
        ]

    def play_one_step(self, state, epsilon: float):
        action = self.select_action(state, epsilon)
        next_state, reward, done, truncated, info = self.env.step(action)
        self.replay_buffer.append((state, action, reward, next_state, done, truncated))
        return next_state, reward, done, truncated, info

    def training_step(self, epsilon=None) -> None:
        raise NotImplementedError("Subclasses must implement training_step method")
