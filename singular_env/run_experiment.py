import random

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from evaluate import evaluate
from make_env import make_env


def run_experiment(algorithm, args):
    run_name = f"{algorithm.__name__}__{args.env_id}__{args.seed}"
    if args.bayesian_log:
        run_name = (
            f"{algorithm.__name__}__{args.bayesian_log}__{args.env_id}__{args.seed}"
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if torch.backends.mps.is_available():
        device_name = "mps"
    elif torch.cuda.is_available():
        device_name = "cuda"
    else:
        device_name = "cpu"

    device = torch.device(device_name)

    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, run_name)])

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    q_net = algorithm(envs, device, writer, args, rb)

    model_path = f"runs/{run_name}.pth"
    torch.save(q_net.state_dict(), model_path)

    episodic_returns = evaluate(
        model_path,
        args.env_id,
        eval_episode=10,
        run_name=f"{run_name}-eval",
        Model=type(q_net),
        device=device,
    )

    for idx, episodic_return in enumerate(episodic_returns):
        writer.add_scalar("eval/episodic_return", episodic_return, idx)

    envs.close()
    writer.close()
