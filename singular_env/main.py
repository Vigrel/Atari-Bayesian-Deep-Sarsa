from algorithms.bayesian_expected_sarsa import bayesian_expected_sarsa
from algorithms.dqn import dqn
from algorithms.ddqn import ddqn
from algorithms.expected_sarsa import expected_sarsa
from experiment_args import ExperimentArgs
from run_experiment import run_experiment

if __name__ == "__main__":
    algorithms = [dqn, ddqn, expected_sarsa, bayesian_expected_sarsa]
    algorithms = [bayesian_expected_sarsa]
    envs = [
        "BreakoutNoFrameskip-v4",
        "SpaceInvadersNoFrameskip-v4",
        "FreewayNoFrameskip-v4",
    ]
    envs = [
        "BreakoutNoFrameskip-v4",
    ]

    for alg in algorithms:
        for env in envs:
            args = ExperimentArgs(env, total_timesteps=500000, learning_rate=100)
            if alg.__name__ == "bayesian_expected_sarsa":
                update_interval = 256
                args = ExperimentArgs(
                    env,
                    train_frequency=update_interval,
                    batch_size=update_interval,
                    buffer_size=update_interval,
                    target_network_frequency=update_interval,
                    learning_starts=update_interval,
                    total_timesteps=500000,
                    seed=41
                )
            print()
            print(alg.__name__, env)
            print()
            run_experiment(alg, args)
    exit()
