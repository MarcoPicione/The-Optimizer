import optimizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from pymoo.indicators.hv import HV
import copy
from optimizer.util import get_dominated
from multiprocessing import Pool
import multiprocessing
import pickle
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


import warnings
warnings.filterwarnings("error")

num_agents = 50
num_iterations = 100
num_params = 3

lb = [-10.] * num_params
ub = [10.] * num_params

optimizer.Logger.setLevel('INFO')

optimizer.FileManager.working_dir = "tmp/myproblem/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = False

def objective1(x):
    return np.cos(x[0])*np.sin(x[1])*x[0]*x[1]*x[2] + 10

def objective2(x):
    return np.cos(x[0]-2)*np.sin(x[1])*(x[0]- 2)*x[1]*x[2]

objective = optimizer.ElementWiseObjective([objective1, objective2])

pso = optimizer.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                    num_particles=num_agents,
                    inertia_weight=0.4, cognitive_coefficient=1.5, social_coefficient=2,
                    initial_particles_position='random', exploring_particles=False, topology='round_robin')

def find_pareto(_):
    pso_tmp = copy.deepcopy(pso)
    pso_tmp.optimize(num_iterations)
    return copy.deepcopy(pso_tmp.pareto_front)

lw = 4
ls = 20
fs = 22
leg_fs = 16

def main():

    file = "test_problem_paretos"    
    if not os.path.isfile(file):
        with Pool(20) as p:
            res = p.map(find_pareto, range(100))
        
        paretos = []
        for i, PF in enumerate(res):
            pareto_x = [particle.fitness[0] for particle in PF]
            pareto_y = [particle.fitness[1] for particle in PF]
            paretos.append([pareto_x, pareto_y])
        
        with open(file, "wb") as output_file:
            pickle.dump(paretos, output_file)
    else:
        with open(file, "rb") as input_file:
            paretos = pickle.load(input_file)

    ref_point=[700,500]
    ind = HV(ref_point=ref_point)
    paretos_flat = [[],[]]

    fig, ax = plt.subplots(figsize=(10,10))

    hvs = []
    for PF in paretos:
        hvs.append(ind(np.array([[PF[0][i], PF[1][i]] for i in range(len(PF[0]))])))
        plt.scatter(PF[0], PF[1], color='blue', alpha=0.01, s=70)
        paretos_flat[0]+=PF[0]
        paretos_flat[1]+=PF[1]
          
    particles = np.array([np.array([paretos_flat[0][i], paretos_flat[1][i]]) for i in range(len(paretos_flat[0]))])
    dominated = get_dominated(particles, 0)
    true_pareto = [particles[i] for i in range(len(particles)) if not dominated[i]]
    true_x = [particle[0] for particle in true_pareto]
    true_y = [particle[1] for particle in true_pareto]
    
    plt.scatter(ref_point[0], ref_point[1], color = 'black',s=200)
    # plt.scatter(true_x, true_y, color = 'red',s=200, label="Best Pareto front")

    ax.spines['top'].set_linewidth(lw)
    ax.spines['right'].set_linewidth(lw)
    ax.spines['left'].set_linewidth(lw)
    ax.spines['bottom'].set_linewidth(lw)
    ax.tick_params(axis='both', which='major', labelsize=ls, width=2)
    plt.xticks(ax.get_xticks(), weight = 'bold')
    plt.yticks(ax.get_yticks(), weight = 'bold')
    plt.title('Test problem', fontweight='bold', fontsize=fs + 2, pad = 10)
    plt.xlabel('Objective 1', fontweight='bold', fontsize=fs)
    plt.ylabel('Objective 2', fontweight='bold', fontsize=fs, labelpad = -3)
    handles, labels = plt.gca().get_legend_handles_labels()
    full_circle = Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='Cerchio Blu')
    blue_patch = Patch(color='blue', label='Pareto front')
    handles.append(blue_patch)
    handles.append(full_circle)
    labels.append('Pareto front')
    labels.append('Reference point')
    plt.legend(handles, labels, prop={'weight':'bold', 'size': leg_fs}, scatterpoints=1, markerscale=2, fontsize=fs, loc='lower left')

    plt.vlines(ref_point[0], ymin=-820, ymax=ref_point[1], color = 'black', linestyles="--", linewidth = lw)
    plt.hlines(ref_point[1], xmin=-750, xmax=ref_point[0], color = 'black', linestyles="--", linewidth = lw)
    plt.savefig("dense_test_problem.png")


    print(f"Mean HV: {np.mean(hvs)} +- {np.std(hvs)}")
    print(f"Optimal HV: {ind(np.array([[true_x[i], true_y[i]]for i in range(len(true_x))]))}")

    if not os.path.exists('tmp'):
        os.makedirs('tmp')

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()