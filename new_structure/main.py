import torch
from atari_enviroment import AtariEnviroment
from experiments.expected_sarsa import ExpectedSarsa
from data_models.experiment_args import ExperimentArgs

envs_id = [
    "BreakoutNoFrameskip-v4",
    "SpaceInvadersNoFrameskip-v4",
    "TennisNoFrameskip-v4",
]
num_envs = 4
stack_frames = 4


experiments = [ExpectedSarsa]
experiments_args = ExperimentArgs("ExpectedSarsa")
device = torch.device("mps")


if __name__ == "__main__":
    for env_id in envs_id:
        atari = AtariEnviroment(env_id, num_envs)
        atari.stack_frames(stack_frames)
        experiment = ExpectedSarsa(atari.env, device, experiments_args)
        experiment.train()
