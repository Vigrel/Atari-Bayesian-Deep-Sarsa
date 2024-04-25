from dataclasses import dataclass


@dataclass
class ExperimentArgs:
    experiment_name: str
    buffer_size: int = 80000
    learning_rate: float = 1e-4
    total_timesteps: int = 1000000
    start_e: int = 1
    end_e: int = 0.01
    exploration_fraction: float = 0.1
    learning_starts: int = 100
    train_frequency: int = 4
    gamma: float = 0.99
    target_network_frequency: int = 1000
    tau: float = 1.0
    batch_size: int = 32
