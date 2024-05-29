import time

import numpy as np
import torch
import torch.optim as optim
from evaluate import mid_evaluation
from networks.bayesian_q_network import BayesianQNetwork


def best_action(model, device, state) -> np.ndarray:
    input_tensor = torch.Tensor(state)
    model_pred = model(input_tensor.to(device))
    q_values = model_pred[:, :, 0] + model_pred[:, :, 1].sqrt() * torch.randn_like(model_pred[:, :, 0])
    return torch.argmax(q_values, dim=1).cpu().numpy()[0]


def bayesian_expected_sarsa(envs, device, writer, args, rb):
    q_network = BayesianQNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)

    target_network = BayesianQNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    prior_network = BayesianQNetwork(envs).to(device)
    prior_network.load_state_dict(q_network.state_dict())

    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)

    for global_step in range(args.total_timesteps):

        actions = np.array(
            [best_action(q_network, device, obs) for _ in range(envs.num_envs)]
        )

        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                writer.add_scalar(
                    "charts/episodic_return", info["episode"]["r"], global_step
                )
                writer.add_scalar(
                    "charts/episode_length", info["episode"]["l"], global_step
                )
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncated):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminated, infos)

        obs = next_obs

        if global_step >= args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                data_next_obs = torch.Tensor(data.next_observations).to(
                    device, torch.float
                )
                data_obs = torch.Tensor(data.observations).to(device, torch.float)

                with torch.no_grad():
                    next_q_values = target_network(data_next_obs)[:, :, 0]
                    expected_next_q_values = torch.mean(next_q_values, dim=1)
                    td_target = (
                        data.rewards.flatten()
                        + args.gamma
                        * expected_next_q_values
                        * (1 - data.dones.flatten())
                    )
                old_val = q_network(data_obs)
                mean_val = old_val[:, :, 0].gather(1, data.actions).squeeze()
                logstd_val = old_val[:, :, 1].gather(1, data.actions).squeeze()
                old_val = torch.stack([mean_val, logstd_val], -1)

                loss = q_network.ELBOloss(old_val, td_target)

                writer.add_scalar("losses/td_loss", loss.item(), global_step)
                writer.add_scalar(
                    "losses/q_values", torch.mean(old_val[:, 0]).item(), global_step
                )

                writer.add_scalar(
                    "losses/q_values_std",
                    torch.mean(old_val[:, 1].exp()).item(),
                    global_step,
                )

                steps_per_sec = args.train_frequency / (time.time() - start_time)
                writer.add_scalar("charts/SPS", steps_per_sec, global_step)

                prior_network.load_state_dict(q_network.state_dict())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for target_network_param, q_network_param in zip(
                    target_network.parameters(), q_network.parameters()
                ):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data
                        + (1.0 - args.tau) * target_network_param.data
                    )

                BayesianQNetwork.update_prior_bnn(q_network, prior_network)
                rb.reset()

            
            if global_step % args.eval_frequency == 0:
                mean_return = mid_evaluation(
                    q_network, envs, args.eval_episodes, device
                )
                writer.add_scalar(
                    "eval/mean_episodic_return",
                    mean_return,
                    global_step,
                )

    return q_network
