import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from agent import Agent, EnvArgs
from dqn import DQN
from deep_expected_sarsa import DeepExpectedSarsa
from draw_env import show_one_episode


def experiment(
    file_name: str, agent: Agent, max_steps: int = 200, num_episodes: int = 600
):
    best_reward = float("-inf")
    best_policy = None
    rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        for step in range(max_steps):
            epsilon = max(1 - episode / 500, 0.01)
            obs, _, done, truncated, _ = agent.play_one_step(obs, epsilon)
            if done or truncated:
                break

        rewards.append(step)

        if total_reward > best_reward:
            best_reward = total_reward
            best_policy = agent.select_action

        if episode > 50:
            agent.training_step(hyperparameters.batch_size)

        print(
            f"\rEpisode: {episode + 1}, Steps: {step + 1}, eps: {epsilon:.3f}", end=""
        )

    plt.figure(figsize=(8, 4))
    plt.plot(rewards)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Sum of rewards", fontsize=14)
    plt.grid(True)
    plt.savefig(f"src/imgs/{file_name}.png")

    show_one_episode(file_name, best_policy)


if __name__ == "__main__":
    hyperparameters = EnvArgs(64, 2000, 0.5, 0.01)
    env = gym.make("CartPole-v1")

    env.reset(seed=42)
    torch.manual_seed(42)
    np.random.seed(42)

    experiment("dqn", DQN(env, hyperparameters))
    env.reset(seed=42)
    experiment("dqn_expected_sarsa", DeepExpectedSarsa(env, hyperparameters))
