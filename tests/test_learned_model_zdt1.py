import optimizer
import numpy as np
from optimizer.tester import explainability, test_model
from matplotlib import pyplot as plt
import multiprocessing

def zdt1_objective1(x):
    return x[0]

def zdt1_objective2(x):
    f1 = x[0]
    g = 1 + 9.0 / (len(x)-1) * sum(x[1:])
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h
    return f2

def main():

    num_agents = 50
    num_iterations = 100
    num_params = 30

    lb = [0.] * num_params
    ub = [1.] * num_params

    optimizer.FileManager.working_dir = "tmp/myproblem/"
    optimizer.FileManager.loading_enabled = False
    optimizer.FileManager.saving_enabled = False


    objective = optimizer.ElementWiseObjective([zdt1_objective1, zdt1_objective2])

    #TEST
    env_kwargs = {'metric_reward' : 5,
                'metric_reward_hv_diff': 0, #130 max
                'evaluation_penalty' : 0, #-300/num_iterations,
                'not_dominated_reward' : 5,#600/num_iterations,
                'render_mode' : 'None'
                    }
    steps=10000
    name = f"myproblem_pretrain_ag_{num_agents}_iter_{num_iterations}_mr_{env_kwargs['metric_reward']}_mdr_{env_kwargs['metric_reward_hv_diff']}_p_{env_kwargs['evaluation_penalty']}_ndr_{env_kwargs['not_dominated_reward']}_steps_{steps}"

    rl_model = f"./models/{name}/{name}_model"
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

    path=f"./{name}"
    models_to_test = ['pso', 'pso_trained_policy', 'pso_random_policy']
    num_iterations = 100
    results = test_model(objective, mopso_parameters, num_iterations, rl_model, ref_point, seeds, path, plot_paretos_enabled = True, models_to_test = models_to_test, verbose = 2)

if __name__ == '__main__':
    # multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
