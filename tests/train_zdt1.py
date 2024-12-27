from optimizer import pso_environment_AEC
import optimizer
import numpy as np
from stable_baselines3 import PPO, TD3
import supersuit as ss
from stable_baselines3.ppo import MlpPolicy, MultiInputPolicy
import time
from matplotlib import pyplot as plt
from optimizer import callback
import os
import torch as th
from stable_baselines3.common.vec_env import VecMonitor
import pdb
from optimizer.trainer import train
from optimizer.tester import explainability

num_agents = 50
num_iterations = 100
num_params = 30

lb = [0.] * num_params
ub = [1.] * num_params

optimizer.Logger.setLevel('INFO')

def zdt1_objective1(x):
    return x[0]

def zdt1_objective2(x):
    f1 = x[0]
    g = 1 + 9.0 / (len(x)-1) * sum(x[1:])
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h
    return f2


optimizer.FileManager.working_dir = "tmp/periodic_problem/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = False


objective = optimizer.ElementWiseObjective([zdt1_objective1, zdt1_objective2])

def main():

    pso = optimizer.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                        num_particles=num_agents,
                        inertia_weight=0.4, cognitive_coefficient=1.5, social_coefficient=2, initial_particles_position='random', exploring_particles=False,
                        rl_model=None, topology='round_robin')

    env_fn = pso_environment_AEC
    scaler = 150
    env_kwargs = {'pso' : pso,
                'pso_iterations' : num_iterations,
                'metric_reward' : 1,
                'metric_reward_hv_diff': 0, #130 max
                'evaluation_penalty' : -0.1, #-300/num_iterations,
                'not_dominated_reward' : 0,#600/num_iterations,
                'render_mode' : 'None'
                    }

    name = f"zdt1_ag_{num_agents}_iter_{num_iterations}_mr_{env_kwargs['metric_reward']}_mdr_{env_kwargs['metric_reward_hv_diff']}_p_{env_kwargs['evaluation_penalty']}_ndr_{env_kwargs['not_dominated_reward']}"
    train(env_fn, steps=500000, seed=0, name = name, **env_kwargs)

    #TEST

    rl_model = f"./{name}_model"
    explainability(rl_model, 5)

if __name__ == "__main__":
    main()