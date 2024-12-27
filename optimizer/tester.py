import optimizer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import matplotlib.colors as mcolors
from pymoo.indicators.hv import HV
import time
from tqdm import tqdm
import json
from stable_baselines3 import PPO
import plotly.graph_objs as go
import plotly.io as pio
from copy import deepcopy
import os
import matplotlib.patches as mpatches
import multiprocessing
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map
import copy
import matplotlib.patches as patches


VALID_MODELS = ['pso', 'pso_trained_policy', 'pso_random_policy', 'pso_explainable_policy']

class results_container:
    def __init__(self, res):
        self.res = self.check_dict(res)
        models = self.res[next(iter(res))]['models']
        self.models_keys = list(models.keys())
        self.metrics_keys = list(models[next(iter(models))].keys())
        self.metrics_keys.remove('pareto_front')
        self.metrics_keys.remove('stopped_on_time')
        self.num_metrics = len(self.metrics_keys)
        self.num_models = len(self.models_keys)
        self.num_seeds = len(self.res.keys())

        self.means = None
        self.stds = None

        self.metric_to_index_map = {}
        for i, met in enumerate(self.metrics_keys):
            self.metric_to_index_map[met] = i

    def check_dict(self, res):
        if type(res) is not dict:
            if type(res) is not str:
                 raise ValueError(f"Results must be a dictionary or a string")
            print(f"Loading file {res}")
            f = open(res) 
            res = json.load(f)
        return res

    def print_results(self):
        print(f"Number of seeds: {len(list(self.res.keys()))}")
        for i, mod in enumerate(self.models_keys):
            print(f"Model {mod}:")
            for j, met in enumerate(self.metrics_keys):
                print(f"\t{met}: {self.get_metric_means(met)[mod]} +- {self.get_metric_stds(met)[mod]}")

    def save_results(self, name):
        res = dict()
        for i, mod in enumerate(self.models_keys):
            metrics = {}
            for met in list(self.metrics_keys):
                metrics[met] = (self.get_metric_means(met)[mod], self.get_metric_stds(met)[mod])
            res[mod] = metrics.copy()
        
        out_file = open(f"{name}_means.json", "w")
        json.dump(res, out_file, indent = 6)
        out_file.close()

        paretos={}
        for k in self.models_keys:
            paretos[k]={[]}
        for k in self.models_keys:
            paretos[k]={[]}
        
        
    def calculate_metrics_momenta(self):
        metrics = np.zeros((self.num_seeds, self.num_models, self.num_metrics))
        for s, seed in enumerate(self.res.keys()):
            for i, mod in enumerate(self.models_keys):
                for j, met in enumerate(self.metrics_keys):
                    metrics[s][i][j] = self.res[seed]['models'][mod][met]

        # means has models on rows and metrics on cols
        self.means = np.mean(metrics, axis = 0)
        self.stds = np.std(metrics, axis = 0)
        return self.means, self.stds

    def get_metric_momentum(self, metric, matrix):
        metric_id = self.metric_to_index_map[metric]
        metric_means = {}
        for i, mod in enumerate(self.models_keys):
            metric_means[mod] = matrix[i][metric_id]
        return metric_means
    
    def get_metric_means(self, metric):
        if self.means is None:
            self.calculate_metrics_momenta()
        return self.get_metric_momentum(metric, self.means)
    
    def get_metric_stds(self, metric):
        if self.stds is None:
            self.calculate_metrics_momenta()
        return self.get_metric_momentum(metric, self.stds)

    def plot_paretos(self, known_pareto = None):
        for r in self.res.keys():
            result = self.res[r]
            paretos = {}
            for mod in result['models'].keys():
                model = result['models'][mod]
                pareto = model['pareto_front']
                axs = []
                num_objectives = len(pareto[0])
                for obj in range(num_objectives):
                    axs.append([fitness[obj] for fitness in pareto])
                paretos[mod] = axs
            if num_objectives == 2: plot_pareto_2d(paretos, result['seed'], known_pareto)
            elif num_objectives == 3: plot_pareto_3d(paretos, result['seed'], known_pareto)
            else: print(f"No implementation of plot fuction for {num_objectives} objectives")

def clean_on_time(res):
    to_be_removed = True
    while(to_be_removed):
        for i, r in enumerate(res):
            to_be_removed = False
            for k in r['models'].keys():
                if r['models'][k]['stopped_on_time']:
                    to_be_removed = True
            if to_be_removed:        
                res.pop(i)
                break
    return res

def test_model(objective, mopso_parameters, num_iterations, rl_model, ref_point, seeds, save_path, n_processes=1, plot_paretos_enabled = False, print_results_enabled = True, save_results_enabled = True, known_pareto=None, time_limit = np.inf, evaluations_max=np.inf, models_to_test = VALID_MODELS, verbose = 0):
    
    file = f"{save_path}_results_{evaluations_max}.json"
    # if not os.path.isfile(file):
    if n_processes > 1:
        args = []
        for s in seeds:
            args.append((copy.deepcopy(objective), mopso_parameters, num_iterations, rl_model, ref_point, s, time_limit, evaluations_max, models_to_test, verbose))
        with Pool(n_processes) as pool:
            res = list(tqdm(pool.starmap(test_seed, args), total=len(args)))
    else:
        res = []
        for s in seeds:
            res.append(test_seed(copy.deepcopy(objective), mopso_parameters, num_iterations, rl_model, ref_point, s, time_limit, evaluations_max, models_to_test, verbose))
    
    res=clean_on_time(res)

    results=dict(zip(seeds, res))
    out_file = open(file, "w")
    json.dump(results, out_file, indent = 6)
    out_file.close()

    # else:
    #     print(f"Loading radius {mopso_parameters['radius_scaler']}")
    #     f_obj = open(file,)
    #     results = json.load(f_obj)

    results_obj = results_container(results)
    if print_results_enabled: results_obj.print_results()
    if save_results_enabled: results_obj.save_results(save_path)
    if plot_paretos_enabled: results_obj.plot_paretos(known_pareto)

    return results_obj

def test_seed(objective, mopso_parameters, num_iterations, rl_model, ref_point, seed, time_limit = np.inf, evaluations_max=np.inf, models_to_test = VALID_MODELS, verbose = 0):
    global VALID_MODELS
    print(f"Testing models: {models_to_test} with scaler {mopso_parameters['radius_scaler']}")
    res =    {'seed'   : seed,
              'models' : {}
             }
    ind = HV(ref_point=ref_point)
    
    if verbose > 0 : print(f"SEED {seed}")

    #Optimizers
    optimizers = []

    if VALID_MODELS[0] in models_to_test:
        if verbose > 1 : print("Starting MOPSO withot RL")
        optimizer.Randomizer.rng = np.random.default_rng(seed)
        pso = optimizer.MOPSO(objective=objective, 
                            lower_bounds=mopso_parameters['lower_bounds'], upper_bounds=mopso_parameters['upper_bounds'], num_particles=mopso_parameters['num_particles'],
                            inertia_weight=0.6, cognitive_coefficient=1, social_coefficient=2, topology = mopso_parameters['topology'],
                            initial_particles_position='random', exploring_particles = mopso_parameters['exploring_particles'],
                            rl_model=None, radius_scaler=mopso_parameters['radius_scaler'])

        start_time_pso = time.time()                
        pso.optimize(num_iterations=num_iterations, max_iterations_without_improvement=mopso_parameters['max_iterations_without_improvement'] if mopso_parameters['exploring_particles'] else 0, time_limit=time_limit, evaluations_max=evaluations_max)
        end_time_pso = time.time()
        optimizers.append(pso)

        evaluations_pso = int(sum(pso.evaluations))
        evaluations_pso_nzz_taken = int(sum(pso.evaluations_nzz_taken))
        evaluations_pso_nzz_not_taken = int(sum(pso.evaluations_nzz_not_taken))
        pareto_pso = [p.fitness.tolist() for p in pso.pareto_front]
        hv_pso = ind(np.array(pareto_pso))
        time_pso = end_time_pso - start_time_pso

        res['models']['pso'] = {'evaluations'  : evaluations_pso,
                                'evaluations_nzz_taken' : evaluations_pso_nzz_taken,
                                'evaluations_nzz_not_taken' : evaluations_pso_nzz_not_taken,
                                'pareto_front' : pareto_pso,
                                'pareto_front_len' : len(pareto_pso),
                                'hyper_volume' : hv_pso,
                                'time'         : time_pso,
                                'stopped_on_time'         : pso.stopped_on_time,
                            }
        
    if VALID_MODELS[1] in models_to_test:
        if verbose > 1 : print("Starting MOPSO with trained policy")
        optimizer.Randomizer.rng = np.random.default_rng(seed)
        pso_trained_policy = optimizer.MOPSO(objective=objective, 
                            lower_bounds=mopso_parameters['lower_bounds'], upper_bounds=mopso_parameters['upper_bounds'], num_particles=mopso_parameters['num_particles'],
                            inertia_weight=0.6, cognitive_coefficient=1, social_coefficient=2, topology = mopso_parameters['topology'],
                            initial_particles_position='random', exploring_particles = mopso_parameters['exploring_particles'],
                            rl_model = rl_model, radius_scaler=mopso_parameters['radius_scaler'])

        start_time_pso_trained_policy = time.time() 
        pso_trained_policy.optimize(num_iterations=num_iterations, max_iterations_without_improvement=mopso_parameters['max_iterations_without_improvement'] if mopso_parameters['exploring_particles'] else 0, time_limit=time_limit, evaluations_max=evaluations_max)
        end_time_pso_trained_policy = time.time()
        optimizers.append(pso_trained_policy)
        evaluations_pso_trained_policy = int(sum(pso_trained_policy.evaluations))
        evaluations_pso_trained_policy_nzz_taken = int(sum(pso_trained_policy.evaluations_nzz_taken))
        evaluations_pso_trained_policy_nzz_not_taken = int(sum(pso_trained_policy.evaluations_nzz_not_taken))
        pareto_pso_trained_policy = [p.fitness.tolist() for p in pso_trained_policy.pareto_front]
        hv_pso_trained_policy= ind(np.array(pareto_pso_trained_policy))
        time_pso_trained_policy = end_time_pso_trained_policy - start_time_pso_trained_policy

        res['models']['pso_trained_policy'] = {'evaluations'  : evaluations_pso_trained_policy,
                                            'evaluations_nzz_taken' : evaluations_pso_trained_policy_nzz_taken,
                                            'evaluations_nzz_not_taken' : evaluations_pso_trained_policy_nzz_not_taken,
                                            'pareto_front' : pareto_pso_trained_policy,
                                            'pareto_front_len' : len(pareto_pso_trained_policy),
                                            'hyper_volume' : hv_pso_trained_policy,
                                            'time'         : time_pso_trained_policy,
                                            'stopped_on_time'         : pso_trained_policy.stopped_on_time,
                                            }
        
    if VALID_MODELS[2] in models_to_test:
        if verbose > 1 : print("Starting MOPSO with random policy")
        optimizer.Randomizer.rng = np.random.default_rng(seed)
        pso_random_policy = optimizer.MOPSO(objective=objective, 
                            lower_bounds=mopso_parameters['lower_bounds'], upper_bounds=mopso_parameters['upper_bounds'], num_particles=mopso_parameters['num_particles'],
                            inertia_weight=0.6, cognitive_coefficient=1, social_coefficient=2, topology = mopso_parameters['topology'],
                            initial_particles_position='random', exploring_particles = mopso_parameters['exploring_particles'],
                            rl_model='random', radius_scaler=mopso_parameters['radius_scaler'])

        start_time_pso_random_policy = time.time() 
        pso_random_policy.optimize(num_iterations=num_iterations, max_iterations_without_improvement=mopso_parameters['max_iterations_without_improvement'] if mopso_parameters['exploring_particles'] else 0, time_limit=time_limit, evaluations_max=evaluations_max)
        end_time_pso_random_policy = time.time()
        optimizers.append(pso_random_policy)
        evaluations_pso_random_policy = int(sum(pso_random_policy.evaluations))
        evaluations_pso_random_policy_nzz_taken = int(sum(pso_random_policy.evaluations_nzz_taken))
        evaluations_pso_random_policy_nzz_not_taken = int(sum(pso_random_policy.evaluations_nzz_not_taken))
        pareto_pso_random_policy = [p.fitness.tolist() for p in pso_random_policy.pareto_front]
        hv_pso_random_policy= ind(np.array(pareto_pso_random_policy))
        time_random_policy = end_time_pso_random_policy - start_time_pso_random_policy

        res['models']['pso_random_policy'] = {'evaluations'  : evaluations_pso_random_policy,
                                            'evaluations_nzz_taken' : evaluations_pso_random_policy_nzz_taken,
                                            'evaluations_nzz_not_taken' : evaluations_pso_random_policy_nzz_not_taken,
                                            'pareto_front' : pareto_pso_random_policy,
                                            'pareto_front_len' : len(pareto_pso_random_policy),
                                            'hyper_volume' : hv_pso_random_policy,
                                            'time'         : time_random_policy,
                                            'stopped_on_time'         : pso_random_policy.stopped_on_time,
                                            }
    if VALID_MODELS[3] in models_to_test:
        if verbose > 1 : print("Starting MOPSO with explainable policy")
        optimizer.Randomizer.rng = np.random.default_rng(seed)
        pso_explainable_policy = optimizer.MOPSO(objective=objective, 
                            lower_bounds=mopso_parameters['lower_bounds'], upper_bounds=mopso_parameters['upper_bounds'], num_particles=mopso_parameters['num_particles'],
                            inertia_weight=0.6, cognitive_coefficient=1, social_coefficient=2, topology = mopso_parameters['topology'],
                            initial_particles_position='random', exploring_particles = mopso_parameters['exploring_particles'],
                            rl_model='explainable', radius_scaler=mopso_parameters['radius_scaler'])

        start_time_pso_explainable_policy = time.time() 
        pso_explainable_policy.optimize(num_iterations=num_iterations, max_iterations_without_improvement=mopso_parameters['max_iterations_without_improvement'] if mopso_parameters['exploring_particles'] else 0, time_limit=time_limit, evaluations_max=evaluations_max)
        end_time_pso_explainable_policy = time.time()
        optimizers.append(pso_explainable_policy)

        evaluations_pso_explainable_policy = int(sum(pso_explainable_policy.evaluations))
        evaluations_pso_explainable_policy_nzz_taken = int(sum(pso_explainable_policy.evaluations_nzz_taken))
        evaluations_pso_explainable_policy_nzz_not_taken = int(sum(pso_explainable_policy.evaluations_nzz_not_taken))
        pareto_pso_explainable_policy = [p.fitness.tolist() for p in pso_explainable_policy.pareto_front]
        hv_pso_explainable_policy= ind(np.array(pareto_pso_explainable_policy))
        time_explainable_policy = end_time_pso_explainable_policy - start_time_pso_explainable_policy

        res['models']['pso_explainable_policy'] = {'evaluations'  : evaluations_pso_explainable_policy,
                                                'evaluations_nzz_taken' : evaluations_pso_explainable_policy_nzz_taken,
                                                'evaluations_nzz_not_taken' : evaluations_pso_explainable_policy_nzz_not_taken,
                                                'pareto_front' : pareto_pso_explainable_policy,
                                                'pareto_front_len' : len(pareto_pso_explainable_policy),
                                                'hyper_volume' : hv_pso_explainable_policy,
                                                'time'         : time_explainable_policy,
                                                'stopped_on_time'         : pso_explainable_policy.stopped_on_time,
                                                }
    return res

def plot_pareto_2d(paretos, seed, known_pareto = None):
    markers = list(MarkerStyle('').markers.keys())
    plt.figure()
    if known_pareto is not None:
        plt.scatter(known_pareto[0], known_pareto[1], c='red', s=5, label = 'Known pareto')
    for i, mod in enumerate(paretos.keys()):
        pareto = paretos[mod] 
        plt.scatter(pareto[0], pareto[1], s=5, label = mod, marker=markers[i])

    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title(f"Seed: {seed}")
    plt.legend()
    plt.savefig(f"./tests/paretos/Pareto_front_seed_{seed}.png")
    plt.close()

def plot_pareto_3d(paretos, seed, known_pareto = None):

    colors = ['red', 'blue', 'green', 'yellow']
    model_paretos = {'pareto_obj1' : [],
                      'pareto_obj2' : [],
                      'pareto_obj3' : [],
                      }
    ranges = np.empty((3,2))
    ranges[:, 0] = np.inf
    ranges[:, 1] = -np.inf

    fig = go.Figure()

    # if known_pareto is not None:
    #     plt.scatter(known_pareto[0], known_pareto[1], c='red', s=5, label = 'Known pareto')

    for i, mod in enumerate(paretos.keys()):
        pareto = paretos[mod]
        model_paretos ['pareto_obj1'] = [fitness[0] for fitness in pareto]
        model_paretos ['pareto_obj2'] = [fitness[1] for fitness in pareto]
        model_paretos ['pareto_obj3'] = [fitness[2] for fitness in pareto]
        for j, k in enumerate(model_paretos.keys()):
            min_pareto = np.min(model_paretos[k])
            max_pareto = np.max(model_paretos[k])
            if min_pareto < ranges[j][0]: ranges[j][0] = min_pareto
            if max_pareto > ranges[j][1]: ranges[j][1] = max_pareto
        
        fig.add_trace(go.Scatter3d(
                        x=pareto[0],
                        y=pareto[1],
                        z=pareto[2],
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=colors[i],
                            opacity=0.8
                            )
                        )
                    #   go.Layout(
                    #         scene=dict(
                    #         xaxis=dict(title='Objective 1', range=ranges[0]),
                    #         yaxis=dict(title='Objective 2', range=ranges[1]),
                    #         zaxis=dict(title='Objective 3', range=ranges[2])
                    #         )
                    #     )

                    )

    pio.write_html(fig, file=f"./tests/paretos/Pareto_front_seed_{seed}.html", auto_open=True)

def explainability(rl_model, num_points):
    model = PPO.load(rl_model)

    if type(num_points) is list or tuple:
        max_bad=num_points[0]
        max_good=num_points[1]
    else:
        max_bad=num_points
        max_good=num_points

    res = np.empty((max_good, max_bad))

    for x in range(max_bad):
        for y in range(max_good):
            res[y,x] = model.predict([x, y], deterministic=True)[0].tolist()

    # res[0][0] = 1.

    lw = 2
    ls = 16
    fs = 18
    leg_fs = 14

    cmap = mcolors.ListedColormap(['red', 'limegreen'])

    # Crea i livelli del contorno
    levels = [-1, 0, 1]

    fig, ax = plt.subplots(figsize=(8,8))
    # Plot dei contorni con hatches (linee diagonali)
    plt.contourf(res, levels=levels, cmap=cmap, alpha=0.5)
    contours = plt.contour(res, levels=levels, colors='black')

    # Aggiungi le hatches
    plt.contourf(res, levels=levels, colors='none', hatches=['/', '\\'], alpha=0)
    # ax.set_xticks(np.arange(0, num_points - 1, 5))
    # ax.set_yticks(np.arange(0, num_points - 1, 5))

    plt.xlabel("Dominated points",fontweight='bold', fontsize=fs)
    plt.ylabel("Archive points",fontweight='bold', fontsize=fs)
    rect1 = patches.Patch(facecolor='red', edgecolor=None, hatch='/', linewidth=lw)
    rect2 = patches.Patch(facecolor='limegreen', edgecolor=None, hatch='\\', linewidth=lw)
    
    labels=['Skipped', 'Evaluated']

    plt.legend(prop={'weight':'bold', 'size': leg_fs}, scatterpoints=1, markerscale=2, fontsize=fs, handles=[rect1, rect2], labels=labels, loc='upper right')
    ax.spines['top'].set_linewidth(lw)
    ax.spines['right'].set_linewidth(lw)
    ax.spines['left'].set_linewidth(lw)
    ax.spines['bottom'].set_linewidth(lw)
    ax.tick_params(axis='both', which='major', labelsize=ls, width=2)
    plt.xticks(ax.get_xticks()[:-1], weight = 'bold')
    plt.yticks(ax.get_yticks()[:-1], weight = 'bold')

    head_tail = os.path.split(rl_model)
    plt.savefig(f"{head_tail[0]}/explainability_{head_tail[1]}.png")
    plt.close()