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
from optimizer.tester import test_model, explainability
from joblib import Parallel, delayed
from multiprocessing import Pool
import multiprocessing
from multiprocessing import get_context


def mp3_objective1(x):
    return np.cos(x[0])*np.sin(x[1])*x[0]*x[1]*x[2] + 10

def mp3_objective2(x):
    return np.cos(x[0]-2)*np.sin(x[1])*(x[0]- 2)*x[1]*x[2]

def mp2_objective1(x):
    return np.sin(x[0])*np.sin(x[1])*x[2] + 10

def mp2_objective2(x):
    return np.cos(x[0]-2)*np.cos(x[1])*x[2]

def mp1_objective1(x):
    return np.cos(x[0])*np.sin(x[1])*x[2] + 10

def mp1_objective2(x):
    return np.cos(x[0]-2)*np.sin(x[1])*x[2]

def objective1(x):
    return np.cos(x[0])*np.sin(x[1])*x[2] + 10

def objective2(x):
    return np.cos(x[0]-2)*np.sin(x[1])*x[2]

optimizer.Logger.setLevel('DEBUG')

test_problem = "mp4"
directory = f"./models/tune_radius/"
save_directory = directory + "results_" + test_problem + "/"

num_agents = 50
num_iterations = 100

num_params_mp = 3

lb_mp = [-10.] * num_params_mp
ub_mp = [10.] * num_params_mp
training_steps=200000

def evaluate_radius(radius):

    model_name = f"mp_radius_{radius}_2_-3_steps_{training_steps}"
    path = directory + model_name

    print('########################################################')
    print(f"Testing radius {radius}")
    print('########################################################')

    if not os.path.isfile(path + "_model"):
        objective_mp = optimizer.ElementWiseObjective([objective1, objective2])
        pso = optimizer.MOPSO(objective=objective_mp, lower_bounds=lb_mp, upper_bounds=ub_mp,
                        num_particles=num_agents,
                        inertia_weight=0.4, cognitive_coefficient=1.5, social_coefficient=2, initial_particles_position='random', exploring_particles=False,
                        rl_model=None, topology='round_robin', radius_scaler=radius)

        env_fn = pso_environment_AEC
        env_kwargs = {'pso' : pso,
                'pso_iterations' : num_iterations,
                'metric_reward' : 2,
                'metric_reward_hv_diff': 0,
                'evaluation_penalty' : -3,
                'not_dominated_reward' : 0,
                'render_mode' : 'None'
                }

        # pre_trained = "./models/myproblem_pretrained_model/model100"
        pre_trained = "./models/myproblem_pretrained_model/myproblem_pretrain_ag_50_iter_100_mr_5_mdr_0_p_0_ndr_5_steps_50000_model"

        train(env_fn, steps=training_steps, seed=0, name = path, pre_trained_model=pre_trained, **env_kwargs)

    return None

def main():
    optimizer.FileManager.working_dir = "tmp/periodic_problem/"
    optimizer.FileManager.loading_enabled = False
    optimizer.FileManager.saving_enabled = False

    radiuses = np.linspace(0.001, 0.1, 100)[5:27]#[0:16]#[0:23]#[0:31]
    radiuses = np.linspace(0.004, 0.026, 30)#[5:27]#[0:16]#[0:23]#[0:31]
    radiuses = np.linspace(0.007, 0.023, 30)#[5:27]#[0:16]#[0:23]#[0:31]
    #radiuses=radiuses[::-1]
    print(f"Radiuses {radiuses}")
    num_rad = len(radiuses)
    hv_means_trained = np.zeros(num_rad)
    hv_stds_trained = np.zeros(num_rad)
    hv_means_random = np.zeros(num_rad)
    hv_stds_random = np.zeros(num_rad)
    evaluations_means_trained = np.zeros(num_rad)
    evaluations_stds_trained = np.zeros(num_rad)
    evaluations_means_random = np.zeros(num_rad)
    evaluations_stds_random = np.zeros(num_rad)

    evaluations_not_0_0_means_trained = np.zeros(num_rad)
    evaluations_not_0_0_stds_trained = np.zeros(num_rad)
    evaluations_not_0_0_means_random = np.zeros(num_rad)
    evaluations_not_0_0_stds_random = np.zeros(num_rad)

    res_objs = []
    with Pool(20) as p:
            res_objs = p.map(evaluate_radius, radiuses)
    # res_objs = []
    # for radius in radiuses:
    #     res_objs.append(evaluate_radius(radius))

    #TEST
    res_objs = []
    for radius in radiuses:
        mopso_parameters = {'lower_bounds': lb_mp,
                    'upper_bounds'        : ub_mp,
                    'num_particles'       : num_agents,
                    'topology'            : 'round_robin',
                    'exploring_particles' : False,
                    'max_iterations_without_improvement' : 100,   
                    'radius_scaler'       : radius 
                    }

        model_name = f"mp_radius_{radius}_2_-3_steps_{training_steps}"
        path = directory + model_name
        rl_model = path + "_model"
        explainability(rl_model, [15,50])
        ref_point_mp2=[11, 1]

        seeds=range(50,100)
        save_path = save_directory + model_name + "other_objective"
        models_to_test = ['pso_trained_policy', 'pso_random_policy']
        objective_mp2 = optimizer.ElementWiseObjective([mp2_objective1, mp2_objective2])

        evaluations_max = int(2000)
        results = test_model(objective_mp2, mopso_parameters, num_iterations, rl_model, ref_point_mp2, seeds, save_path, n_processes=20, plot_paretos_enabled = False, models_to_test = models_to_test, evaluations_max=evaluations_max, verbose = 2)
        res_objs.append(results)

    for r, res in enumerate(res_objs):  
        hv_means_trained[r] = res.get_metric_means('hyper_volume')['pso_trained_policy']
        hv_means_random[r] = res.get_metric_means('hyper_volume')['pso_random_policy']
        hv_stds_trained[r] = res.get_metric_stds('hyper_volume')['pso_trained_policy']
        hv_stds_random[r] = res.get_metric_stds('hyper_volume')['pso_random_policy']

        evaluations_means_trained[r] = res.get_metric_means('evaluations')['pso_trained_policy']
        evaluations_means_random[r] = res.get_metric_means('evaluations')['pso_random_policy']
        evaluations_stds_trained[r] = res.get_metric_stds('evaluations')['pso_trained_policy']
        evaluations_stds_random[r] = res.get_metric_stds('evaluations')['pso_random_policy']
        
        evaluations_not_0_0_means_trained[r] = res.get_metric_means('evaluations_not_0_0')['pso_trained_policy']
        evaluations_not_0_0_means_random[r] = res.get_metric_means('evaluations_not_0_0')['pso_random_policy']
        evaluations_not_0_0_stds_trained[r] = res.get_metric_stds('evaluations_not_0_0')['pso_trained_policy']
        evaluations_not_0_0_stds_random[r] = res.get_metric_stds('evaluations_not_0_0')['pso_random_policy']
        
        np.save(f"{save_directory}_hv_means_trained_max_evaluations_{evaluations_max}.npy", hv_means_trained)
        np.save(f"{save_directory}_hv_stds_trained_max_evaluations_{evaluations_max}.npy", hv_stds_trained)
        np.save(f"{save_directory}_evaluations_means_trained_max_evaluations_{evaluations_max}.npy", evaluations_means_trained)
        np.save(f"{save_directory}_evaluations_stds_trained_max_evaluations_{evaluations_max}.npy", evaluations_stds_trained)
        np.save(f"{save_directory}_evaluations_not_0_0_means_trained_max_evaluations_{evaluations_max}.npy", evaluations_not_0_0_means_trained)
        np.save(f"{save_directory}_evaluations_not_0_0_stds_trained_max_evaluations_{evaluations_max}.npy", evaluations_not_0_0_stds_trained)

        np.save(f"{save_directory}_hv_means_random_max_evaluations_{evaluations_max}.npy", hv_means_random)
        np.save(f"{save_directory}_hv_stds_random_max_evaluations_{evaluations_max}.npy", hv_stds_random)
        np.save(f"{save_directory}_evaluations_means_random_max_evaluations_{evaluations_max}.npy", evaluations_means_random)
        np.save(f"{save_directory}_evaluations_stds_random_max_evaluations_{evaluations_max}.npy", evaluations_stds_random)
        np.save(f"{save_directory}_evaluations_not_0_0_means_random_max_evaluations_{evaluations_max}.npy", evaluations_not_0_0_means_random)
        np.save(f"{save_directory}_evaluations_not_0_0_stds_random_max_evaluations_{evaluations_max}.npy", evaluations_not_0_0_stds_random)

    lw = 4
    ls = 20
    fs = 22
    leg_fs = 16
    
    fig, ax = plt.subplots(figsize=(20,10))
    ax.errorbar(radiuses, hv_means_trained, hv_stds_trained, label='Trained policy', linewidth=lw)
    ax.errorbar(radiuses, hv_means_random, hv_stds_random, label='Random policy', linewidth=lw)

    ax.spines['top'].set_linewidth(lw)
    ax.spines['right'].set_linewidth(lw)
    ax.spines['left'].set_linewidth(lw)
    ax.spines['bottom'].set_linewidth(lw)
    ax.tick_params(axis='both', which='major', labelsize=ls, width=2)
    plt.xticks(ax.get_xticks()[1:-1], weight = 'bold')
    plt.yticks(ax.get_yticks()[1:-1], weight = 'bold')
    plt.legend(prop={'weight':'bold', 'size': leg_fs}, scatterpoints=1, markerscale=2, fontsize=fs)
    plt.xlabel('Radius scaler', fontweight='bold', fontsize=fs)
    plt.ylabel('Mean hyper volume', fontweight='bold', fontsize=fs)
    plt.savefig(f"radius_tuning_hv{test_problem}.png")

    plt.close()
    # ax.errorbar(radiuses, evaluations_means_trained, evaluations_stds_trained, label='Trained policy')
    # ax.errorbar(radiuses, evaluations_means_random, evaluations_stds_random, label='Random policy')
    # ax.set_xlabel('Radius\' scaler')
    # ax.set_ylabel('Mean number of evaluations')
    # ax.legend()
    fig, ax = plt.subplots(figsize=(20,10))
    ax.errorbar(radiuses, evaluations_not_0_0_means_trained, evaluations_not_0_0_stds_trained, label='Trained policy', linewidth=lw)
    ax.errorbar(radiuses, evaluations_not_0_0_means_random, evaluations_not_0_0_stds_random, label='Random policy', linewidth=lw)

    ax.spines['top'].set_linewidth(lw)
    ax.spines['right'].set_linewidth(lw)
    ax.spines['left'].set_linewidth(lw)
    ax.spines['bottom'].set_linewidth(lw)
    ax.tick_params(axis='both', which='major', labelsize=ls, width=2)
    plt.xticks(ax.get_xticks()[1:-1], weight = 'bold')
    plt.yticks(ax.get_yticks()[1:-1], weight = 'bold')
    plt.legend(prop={'weight':'bold', 'size': leg_fs}, scatterpoints=1, markerscale=2, fontsize=fs, loc='upper left')
    plt.xlabel('Radius scaler', fontweight='bold', fontsize=fs)
    plt.ylabel('Mean number of not forced evaluations', fontweight='bold', fontsize=fs)

    plt.savefig(f"radius_tuning_eval{test_problem}.png")

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
