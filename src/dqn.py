import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim

from agent import Agent, EnvArgs


# code: https://github.com/ageron/handson-ml3
class DQN(Agent):
    def __init__(self, env: gym.Env, args: EnvArgs) -> None:
        super().__init__(env, args)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.loss_fn = nn.MSELoss()

    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.num_state, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions),
        )

    def training_step(self, epsilon=None) -> None:
        states, actions, rewards, next_states, dones, truncateds = (
            self.sample_experiences()
        )
        next_q_values = (
            self.model(torch.tensor(next_states, dtype=torch.float32)).detach().numpy()
        )

        runs = 1.0 - (dones | truncateds)
        target_q_values = (
            rewards + runs * self.args.discount_factor * np.max(next_q_values, axis=1)
        ).reshape(-1, 1)

        mask = torch.tensor(np.eye(self.num_actions)[actions], dtype=torch.float32)
        all_q_values = self.model(torch.tensor(states, dtype=torch.float32))
        q_values = torch.sum(all_q_values * mask, dim=1, keepdim=True)
        loss = self.loss_fn(torch.tensor(target_q_values).float(), q_values.float())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
