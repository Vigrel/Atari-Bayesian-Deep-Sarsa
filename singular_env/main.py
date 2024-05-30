from algorithms.bayesian_expected_sarsa import bayesian_expected_sarsa
from algorithms.dqn import dqn
from algorithms.ddqn import ddqn
from algorithms.expected_sarsa import expected_sarsa
from experiment_args import ExperimentArgs
from run_experiment import run_experiment

if __name__ == "__main__":
    # algorithms = [dqn, ddqn, expected_sarsa, bayesian_expected_sarsa]
    # envs = [
    #     "BreakoutNoFrameskip-v4",
    #     "SpaceInvadersNoFrameskip-v4",
    #     "FreewayNoFrameskip-v4",
    # ]

    # for alg in algorithms:
    #     for env in envs:
    #         args = ExperimentArgs(env)
    #         if alg.__name__ == "bayesian_expected_sarsa":
    #             args = ExperimentArgs(
    #                 env,
    #                 train_frequency=256,
    #                 batch_size=256,
    #                 buffer_size=256,
    #                 target_network_frequency=256,
    #             )
    #         run_experiment(alg, args)
    ##################################################################
    pow2 = [2**j for j in range(9, 11)]
    pow2 = [32, 256]
    for num in pow2:
        args = ExperimentArgs(
            "SpaceInvadersNoFrameskip-v4",
            train_frequency=num,
            batch_size=num,
            buffer_size=num,
            target_network_frequency=num,
            bayesian_log=num,
        )
        run_experiment(bayesian_expected_sarsa, args)
    exit()
