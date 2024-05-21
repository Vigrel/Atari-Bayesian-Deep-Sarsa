import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from base_agent import Agent
from data_models.experiment_args import ExperimentArgs, BayesianExperimentArgs
from models.bayesian_q_network import BayesianQNetwork
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecEnv
from torch import device, optim
from torch.utils.tensorboard import SummaryWriter

import logging
logger = logging.getLogger(__name__)


class BayesianExpectedSarsa(Agent):
    def __init__(self, envs: VecEnv, device: device, args: BayesianExperimentArgs) -> None:
        super().__init__(envs, device)
        self.model = BayesianQNetwork(envs.action_space.n).to(device)
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

    def best_action(self, state) -> np.ndarray:
        input_tensor = torch.Tensor(state).permute(0, 3, 1, 2)
        q_values = self.model(input_tensor.to(self.device))
        #logger.info(f'Q_values: {q_values}')
        #logger.info(f'Q_values: {q_values[:,:,0]}')
        return torch.argmax(q_values[:,:,0], dim=1).cpu().numpy()[0]

    def train(self):

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        target_network = BayesianQNetwork(self.envs.action_space.n).to(self.device)
        target_network.load_state_dict(self.model.state_dict())

        prior_network = BayesianQNetwork(self.envs.action_space.n).to(self.device)
        prior_network.load_state_dict(self.model.state_dict())

        start_time = time.time()
        obs = self.envs.reset()

        for global_step in range(self.args.total_timesteps):

            actions = np.array(
                [self.best_action(obs) for _ in range(self.envs.num_envs)]
                )
            #logger.info(f'Actions: {actions}')
            
            next_obs, rewards, dones, infos = self.envs.step(actions)
            #exit()
            for idx in np.where(dones == True)[0]:
                self.writer.add_scalar(
                    "charts/episodic_return", rewards[idx], global_step
                )
                self.writer.add_scalar(
                    "charts/episode_length", infos[idx]["frame_number"], global_step
                )
            
            self.rb.add(obs, next_obs, actions, rewards, dones, infos)

            if global_step == 0:
                input_tensor_test = torch.Tensor(next_obs).permute(0, 3, 1, 2).to(self.device)
                logger.info(f'Next_obs 0 ({input_tensor_test.shape })[{input_tensor_test.type()}]')

            obs = next_obs

            if global_step >= self.args.learning_starts:
                if global_step % self.args.train_frequency == 0:
                    data = self.rb.sample(self.args.batch_size)
                    data_next_obs = torch.Tensor(data.next_observations).permute(
                        0, 3, 1, 2
                    ).to(self.device,torch.float)
                    data_obs = torch.Tensor(data.observations).permute(0, 3, 1, 2).to(self.device,torch.float)

                    #Calculate belman target
                    with torch.no_grad():
                        data_next_obs
                        #logger.info(f'Data_next_obs ({data_next_obs.shape})[{data_next_obs.type()}]')
                        next_q_values = target_network(data_next_obs)[:,:,0]
                        expected_next_q_values = torch.mean(next_q_values, dim=1)
                        td_target = (
                            data.rewards.flatten()
                            + self.args.gamma
                            * expected_next_q_values
                            * (1 - data.dones.flatten())
                        )
                    old_val = self.model(data_obs)
                    mean_val = old_val[:,:,0].gather(1, data.actions).squeeze()
                    logstd_val = old_val[:,:,1].gather(1, data.actions).squeeze()
                    old_val = torch.stack([mean_val, logstd_val],-1)

                    loss = self.model.ELBOloss(old_val, td_target)

                    
                    self.writer.add_scalar(
                        "losses/td_loss", loss.item(), global_step
                    )
                    self.writer.add_scalar(
                        "losses/q_values", torch.mean(old_val[:,0]).item(), global_step
                    )

                    self.writer.add_scalar(
                        "losses/q_values_std", torch.mean(old_val[:,1].exp()).item(), global_step
                    )

                    steps_per_sec = self.args.train_frequency / (
                        time.time() - start_time
                    )
                    self.writer.add_scalar("charts/SPS", steps_per_sec, global_step)

                    prior_network.load_state_dict(self.model.state_dict())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                #if global_step % self.args.target_network_frequency == 0:
                    for target_network_param, q_network_param in zip(
                        target_network.parameters(), self.model.parameters()
                    ):
                        target_network_param.data.copy_(
                            self.args.tau * q_network_param.data
                            + (1.0 - self.args.tau) * target_network_param.data
                        )
                    
                    #Set new prior using old network
                    BayesianQNetwork.update_prior_bnn(self.model, prior_network)
                
                    #Reset replay buffer
                    logger.info(f'Replay Buffer size: {self.rb.size()}')
                    self.rb.reset()
                    #logger.info(f'Replay Buffer size: {self.rb.size()}')

        if True:
            model_path = (
                f"src/runs/{self.args.experiment_name}/{self.args.experiment_name}.pth"
            )
            torch.save(self.model.state_dict(), model_path)

        self.envs.close()
        self.writer.close()
