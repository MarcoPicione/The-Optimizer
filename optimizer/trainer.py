from optimizer import pso_environment_AEC
import optimizer
import numpy as np
from stable_baselines3 import PPO
import supersuit as ss
from stable_baselines3.ppo import MlpPolicy
import time
from matplotlib import pyplot as plt
from optimizer import callback
from tqdm import tqdm


def train(env_fn, steps: int = 1e4, seed: int = 0, **env_kwargs):
    tqdm.rich = False
    env = env_fn.parallel_env(**env_kwargs)
    env.reset(seed=seed)
    print(f"Starting training on {str(env.metadata['name'])}.")
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs = 1, num_cpus=1, base_class="stable_baselines3")

    model = PPO(
        MlpPolicy,
        env,
        verbose=2,
        learning_rate=1e-2,
        n_steps=2048,
        batch_size=2048,
        n_epochs = 100,
    )

    model.learn(total_timesteps=steps, progress_bar=True, callback=callback.CustomCallback())
    model.save("model")
    print("Model has been saved.")
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")
    env.close()
