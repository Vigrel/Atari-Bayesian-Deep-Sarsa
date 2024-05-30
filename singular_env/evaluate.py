from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from make_env import make_env
from networks.bayesian_q_network import BayesianQNetwork
from networks.q_network import QNetwork
from torch.utils.tensorboard import SummaryWriter


def mid_evaluation(model, env_id, run_name, num_episodes, device):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, run_name, True)])
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < num_episodes:
        q_values = model(torch.Tensor(obs).to(device))
        actions = torch.argmax(q_values, dim=1).cpu().numpy()

        if actions.ndim > 1:
            actions = [actions[0][0]]

        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs
    return np.mean(episodic_returns)


def evaluate(
    model_path: str,
    env_id: str,
    eval_episode: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, run_name, True)])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episode:
        q_values = model(torch.Tensor(obs).to(device))
        actions = torch.argmax(q_values, dim=1).cpu().numpy()

        if actions.ndim > 1:
            actions = [actions[0][0]]

        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns


if __name__ == "__main__":
    device = torch.device("mps")
    episodic_returns = evaluate(
        "./exp21_05/runs/dqn__BreakoutNoFrameskip-v4__42.pth",
        "BreakoutNoFrameskip-v4",
        50,
        "dqn__BreakoutNoFrameskip-v4__42-eval",
        QNetwork,
        device,
    )
    writer = SummaryWriter(f"runs/dqn__BreakoutNoFrameskip-v4__42-eval")

    for idx, episodic_return in enumerate(episodic_returns):
        writer.add_scalar("eval/episodic_return", episodic_return, idx)
    writer.close()
