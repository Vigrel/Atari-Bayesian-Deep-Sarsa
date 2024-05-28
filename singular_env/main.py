from algorithms.bayesian_expected_sarsa import bayesian_expected_sarsa
from algorithms.dqn import dqn
from algorithms.ddqn import ddqn
from algorithms.expected_sarsa import expected_sarsa
from experiment_args import ExperimentArgs
from run_experiment import run_experiment

if __name__ == "__main__":
    algorithms = [dqn, ddqn, expected_sarsa, bayesian_expected_sarsa]
    envs = [
        "BreakoutNoFrameskip-v4",
        "SpaceInvadersNoFrameskip-v4",
        "FreewayNoFrameskip-v4",
    ]

    for alg in algorithms:
        for env in envs:
            args = ExperimentArgs(env)
            if alg.__name__ == "bayesian_expected_sarsa":
                args = ExperimentArgs(
                    env,
                    train_frequency=64,
                    batch_size=64,
                    buffer_size=200,
                    target_network_frequency=64,
                )
            print()
            print(alg.__name__, env)
            print()
            run_experiment(alg, args)
    exit()
