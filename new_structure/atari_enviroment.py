from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


class AtariEnviroment:
    def __init__(self, env_id: str, n_envs: int, seed: int = 42) -> None:
        self.env = make_atari_env(env_id, n_envs, seed)

    def set_video_record(self, folder_name: str) -> None:
        self.env = RecordVideo(self.env, folder_name)

    def stack_frames(self, n_stacks: int) -> None:
        self.env = VecFrameStack(self.env, n_stack=n_stacks)


if __name__ == "__main__":
    atari = AtariEnviroment("BreakoutNoFrameskip-v4", 1)
    atari.stack_frames(4)
