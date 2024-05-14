import torch
from atari_enviroment import AtariEnviroment
from experiments.bayesian_expected_sarsa import BayesianExpectedSarsa
from data_models.experiment_args import ExperimentArgs, BayesianExperimentArgs

import logging
logger = logging.getLogger(__name__)

envs_id = [
    'BreakoutNoFrameskip-v4',
    #'SpaceInvadersNoFrameskip-v4',
    #'TennisNoFrameskip-v4'
]
num_envs = 4
stack_frames = 4


experiments = [BayesianExpectedSarsa]
experiments_args = BayesianExperimentArgs("BayesianExpectedSarsa")
if torch.backends.mps.is_available():
    device_name = "mps"
elif torch.cuda.is_available():
    device_name = "cuda"
else:
    device_name = "cpu"
device = torch.device(device_name)


if __name__ == "__main__":
    logging.basicConfig(filename='experiment.log', level=logging.INFO)
    logger.info('Started')

    for env_id in envs_id:
        atari = AtariEnviroment(env_id, num_envs)
        atari.stack_frames(stack_frames)
        experiment = BayesianExpectedSarsa(atari.env, device, experiments_args)
        experiment.train()

    logger.info('Finished')
