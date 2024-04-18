import matplotlib.animation
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np


# code: https://github.com/ageron/handson-ml3
def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return (patch,)


def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis("off")
    anim = matplotlib.animation.FuncAnimation(
        fig,
        update_scene,
        fargs=(frames, patch),
        frames=len(frames),
        repeat=repeat,
        interval=interval,
    )
    plt.close()
    return anim


def show_one_episode(file_name, policy, n_max_steps=200, seed=42):
    frames = []
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)
    for _ in range(n_max_steps):
        frames.append(env.render())
        action = policy(obs)
        obs, _, done, truncated, _ = env.step(action)
        if done or truncated:
            break
    env.close()
    anim = plot_animation(frames)
    anim.save(f"src/imgs/{file_name}.gif")
