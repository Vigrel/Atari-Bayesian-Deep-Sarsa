import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from base_agent import Agent
from data_models.experiment_args import ExperimentArgs
from models.q_network import QNetwork
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecEnv
from torch import device, optim
from torch.utils.tensorboard import SummaryWriter


class ExpectedSarsa(Agent):
    def __init__(self, envs: VecEnv, device: device, args: ExperimentArgs) -> None:
        super().__init__(envs, device)
        self.rb = ReplayBuffer(
            args.buffer_size,
            envs.observation_space,
            envs.action_space,
            device,
            self.envs.num_envs,
            True,
            False,
        )
        self.args = args
        self.writer = SummaryWriter(f"src/runs/{args.experiment_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    def linear_schedule(self, t: int) -> float:
        duration = self.args.exploration_fraction * self.args.total_timesteps
        slope = (self.args.end_e - self.args.start_e) / duration
        return max(slope * t + self.args.start_e, self.args.end_e)

    def train(self):

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        target_network = QNetwork(self.envs.action_space.n).to(self.device)
        target_network.load_state_dict(self.model.state_dict())

        start_time = time.time()
        obs = self.envs.reset()

        for global_step in range(self.args.total_timesteps):
            epsilon = self.linear_schedule(global_step)

            if random.random() < epsilon:
                actions = np.array(
                    [self.random_action() for _ in range(self.envs.num_envs)]
                )
            else:
                actions = np.array(
                    [self.best_action(obs) for _ in range(self.envs.num_envs)]
                )
            next_obs, rewards, dones, infos = self.envs.step(actions)
            for idx in np.where(dones == True)[0]:
                self.writer.add_scalar(
                    "charts/episodic_return", rewards[idx], global_step
                )
                self.writer.add_scalar(
                    "charts/episode_length", infos[idx]["frame_number"], global_step
                )
                self.writer.add_scalar("charts/epsilon", epsilon, global_step)
            self.rb.add(obs, next_obs, actions, rewards, dones, infos)

            obs = next_obs

            if global_step > self.args.learning_starts:
                if global_step % self.args.train_frequency == 0:
                    data = self.rb.sample(self.args.batch_size)
                    data_next_obs = torch.Tensor(data.next_observations).permute(
                        0, 3, 1, 2
                    )
                    data_obs = torch.Tensor(data.observations).permute(0, 3, 1, 2)
                    with torch.no_grad():
                        next_q_values = target_network(data_next_obs)
                        expected_next_q_values = torch.mean(next_q_values, dim=1)
                        td_target = (
                            data.rewards.flatten()
                            + self.args.gamma
                            * expected_next_q_values
                            * (1 - data.dones.flatten())
                        )
                    old_val = self.model(data_obs).gather(1, data.actions).squeeze()
                    loss = F.mse_loss(td_target, old_val)

                    if global_step % 100 == 0:
                        self.writer.add_scalar(
                            "losses/td_loss", loss.item(), global_step
                        )
                        self.writer.add_scalar(
                            "losses/q_values", torch.mean(old_val).item(), global_step
                        )

                        steps_per_sec = self.args.train_frequency / (
                            time.time() - start_time
                        )
                        self.writer.add_scalar("charts/SPS", steps_per_sec, global_step)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if global_step % self.args.target_network_frequency == 0:
                    for target_network_param, q_network_param in zip(
                        target_network.parameters(), self.model.parameters()
                    ):
                        target_network_param.data.copy_(
                            self.args.tau * q_network_param.data
                            + (1.0 - self.args.tau) * target_network_param.data
                        )

        if True:
            model_path = (
                f"src/runs/{self.args.experiment_name}/{self.args.experiment_name}.pth"
            )
            torch.save(self.model.state_dict(), model_path)

        self.envs.close()
        self.writer.close()
