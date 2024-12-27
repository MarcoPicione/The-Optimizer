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
from optimizer.tester import explainability, test_model
import multiprocessing

optimizer.Logger.setLevel('INFO')

def objective1(x):
    return np.cos(x[0])*np.sin(x[1])*x[2] + 10

def objective2(x):
    return np.cos(x[0]-2)*np.sin(x[1])*x[2]


optimizer.FileManager.working_dir = "tmp/myproblem/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = False


objective = optimizer.ElementWiseObjective([objective1, objective2])

def main():

    num_agents = 50
    num_iterations = 100
    num_params = 3

    lb = [-10.] * num_params
    ub = [10.] * num_params

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
                'evaluation_penalty' : -1, #-300/num_iterations,
                'not_dominated_reward' : 0,#600/num_iterations,
                'render_mode' : 'None'
                    }
    steps = 200000 
    name = f"myproblem_pretrain_ag_{num_agents}_iter_{num_iterations}_mr_{env_kwargs['metric_reward']}_mdr_{env_kwargs['metric_reward_hv_diff']}_p_{env_kwargs['evaluation_penalty']}_ndr_{env_kwargs['not_dominated_reward']}_steps_{steps}"
    pre_trained = "./models/myproblem_pretrained_model/model10"
    # pre_trained = None
    train(env_fn, steps=steps, seed=0, name = name, pre_trained_model=pre_trained, **env_kwargs)

    #TEST

    rl_model = f"./{name}_model"
    explainability(rl_model, 50)

    mopso_parameters = {'lower_bounds': lb,
                'upper_bounds'        : ub,
                'num_particles'       : num_agents,
                'topology'            : 'round_robin',
                'exploring_particles' : False,   
                'radius_scaler'       : 0.03 
                }
    ref_point=[600,600]
    seeds=range(50,150)
    path="./results_myproblem"
    models_to_test = ['pso','pso_trained_policy', 'pso_random_policy']

    # results = test_model(objective, mopso_parameters, num_iterations, rl_model, ref_point, seeds, path, plot_paretos_enabled = False, models_to_test = models_to_test, n_processes=20, verbose = 2)


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()