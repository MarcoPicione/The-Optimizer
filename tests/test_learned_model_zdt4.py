import optimizer
import numpy as np
from optimizer.tester import explainability, test_model
from matplotlib import pyplot as plt
import multiprocessing

def zdt4_objective1(x):
    return x[0]

def zdt4_objective2(x):
    f1 = x[0]
    g = 1.0 + 10 * (len(x) - 1) + sum([i**2 - 10 * np.cos(4 * np.pi * i) for i in x[1:]])
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h
    return f2

optimizer.Logger.setLevel('INFO')    

def main():

    num_agents = 50
    num_iterations = 100
    num_params = 10

    lb = [0.] + [-5.] * (num_params - 1)
    ub = [1.] + [5.] * (num_params - 1)

    optimizer.FileManager.working_dir = "tmp/myproblem/"
    optimizer.FileManager.loading_enabled = False
    optimizer.FileManager.saving_enabled = False


    objective = optimizer.ElementWiseObjective([zdt4_objective1, zdt4_objective2])

    #TEST
    env_kwargs = {'metric_reward' : 1,
                'metric_reward_hv_diff': 0, #130 max
                'evaluation_penalty' : -1, #-300/num_iterations,
                'not_dominated_reward' : 0,#600/num_iterations,
                'render_mode' : 'None'
                    }
    name = f"myproblem_pretrain_ag_{num_agents}_iter_{num_iterations}_mr_{env_kwargs['metric_reward']}_mdr_{env_kwargs['metric_reward_hv_diff']}_p_{env_kwargs['evaluation_penalty']}_ndr_{env_kwargs['not_dominated_reward']}_long"
    name = f"myproblem_ag_50_iter_100_mr_1_mdr_0_p_0_ndr_1_model"

    rl_model = f"./models/final_model/myproblem_pretrain_ag_50_iter_100_mr_2_mdr_0_p_-3_ndr_0_steps_500000_model"

    mopso_parameters = {'lower_bounds': lb,
                'upper_bounds'        : ub,
                'num_particles'       : num_agents,
                'topology'            : 'round_robin',
                'exploring_particles' : False,   
                'radius_scaler'       : 0.01 
                }
    ref_point=[50,50]
    seeds=range(50,150)

    path=f"./{name}"
    models_to_test = ['pso', 'pso_trained_policy', 'pso_random_policy']
    num_iterations = 100
    evaluations_max = np.inf #5000
    time_limit=np.inf #60
    results = test_model(objective, mopso_parameters, num_iterations, rl_model, ref_point, seeds, path, plot_paretos_enabled = False, models_to_test = models_to_test,n_processes=20, evaluations_max=evaluations_max,time_limit=time_limit, verbose = 2)

if __name__ == '__main__':
    # multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
