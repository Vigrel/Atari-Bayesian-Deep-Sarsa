import random

import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv

from models.q_network import QNetwork


class Agent:
    def __init__(self, envs: VecEnv, device: torch.device) -> None:
        self.envs = envs
        self.device = device
        self.model = QNetwork(envs.action_space.n).to(device)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.backends.cudnn.deterministic = True

    def random_action(self) -> np.ndarray:
        return self.envs.action_space.sample()

    def best_action(self, state) -> np.ndarray:
        input_tensor = torch.Tensor(state).permute(0, 3, 1, 2)
        q_values = self.model(input_tensor.to(self.device))
        return torch.argmax(q_values, dim=1).cpu().numpy()[0]


if __name__ == "__main__":
    from atari_enviroment import AtariEnviroment

    atari = AtariEnviroment("BreakoutNoFrameskip-v4", 1)
    atari.stack_frames(4)
    obs = atari.env.reset()
    if torch.backends.mps.is_available():
        device_name = "mps"
    elif torch.cuda.is_available():
        device_name = "cuda"
    else:
        device_name = "cpu"
    device = torch.device(device_name)
    agent = Agent(atari.env, device, 8000, "teste")
