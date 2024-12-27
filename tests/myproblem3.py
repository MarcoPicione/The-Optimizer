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

import warnings
warnings.filterwarnings("error")

num_agents = 50
num_iterations = 100
num_params = 3

lb = [-10.] * num_params
ub = [10.] * num_params

optimizer.Logger.setLevel('INFO')
# optimizer.Randomizer.rng = np.random.default_rng(42)


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
ref_point = [600, 750]  
def mean_hv(_):
    ind = HV(ref_point=ref_point)
    pso_tmp = copy.deepcopy(pso)
    pso_tmp.optimize(num_iterations)
    pareto_x = [particle.fitness[0] for particle in pso_tmp.pareto_front]
    pareto_y = [particle.fitness[1] for particle in pso_tmp.pareto_front]
    hv_pso = ind(np.array([[pareto_x[i], pareto_y[i]] for i in range(len(pareto_x))]))
    return hv_pso

def best_hv_set(_):
    pso_tmp = copy.deepcopy(pso)
    pso_tmp.optimize(num_iterations)
    return copy.deepcopy(pso_tmp.pareto_front)

lw = 4
ls = 20
fs = 22
leg_fs = 16

def main():

    # run the optimization algorithm several times

    # if False:
    #     with Pool(20) as p:
    #         res = p.map(mean_hv, range(100))

    #     

    # file = "MyProblem3PF.npy"    
    # if not os.path.isfile(file):
    #     with Pool(20) as p:
    #         res = p.map(best_hv_set, range(100))
    
    #     particles = []
    #     for elem in res:
    #         particles += elem

    #     particle_fitnesses = np.array([particle.fitness for particle in particles])
    #     dominated = get_dominated(particle_fitnesses, 0)
    #     PF = [particles[i] for i in range(len(particles)) if not dominated[i]]
    #     print(f"Len: {len(PF)}")
    #     pareto_x = [particle.fitness[0] for particle in PF]
    #     pareto_y = [particle.fitness[1] for particle in PF]
    #     np.save(file, [pareto_x, pareto_y])

    file = "MyProblem3PF_overlap"    
    if not os.path.isfile(file):
        with Pool(20) as p:
            res = p.map(best_hv_set, range(100))
        
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

    ref_point=[600,750]
    ind = HV(ref_point=ref_point)
    paretos_flat = [[],[]]

    fig, ax = plt.subplots(figsize=(8,8))
    ax.spines['top'].set_linewidth(lw)
    ax.spines['right'].set_linewidth(lw)
    ax.spines['left'].set_linewidth(lw)
    ax.spines['bottom'].set_linewidth(lw)
    ax.tick_params(axis='both', which='major', labelsize=ls, width=2)
    plt.xticks(ax.get_xticks()[1:-1], weight = 'bold')
    plt.yticks(ax.get_yticks()[1:-1], weight = 'bold')
    plt.title('Validation problem', fontweight='bold', fontsize=fs + 2, pad = 20)
    plt.xlabel('Objective 1', fontweight='bold', fontsize=fs)
    plt.ylabel('Objective 2', fontweight='bold', fontsize=fs)

    hvs = []
    for PF in paretos:
        hvs.append(ind(np.array([[PF[0][i], PF[1][i]] for i in range(len(PF[0]))])))
        plt.scatter(PF[0], PF[1], color='blue', alpha=0.01, s=70, label= 'Pareto front')
        paretos_flat[0]+=PF[0]
        paretos_flat[1]+=PF[1]
          
    particles = np.array([np.array([paretos_flat[0][i], paretos_flat[1][i]]) for i in range(len(paretos_flat[0]))])
    dominated = get_dominated(particles, 0)
    true_pareto = [particles[i] for i in range(len(particles)) if not dominated[i]]
    true_x = [particle[0] for particle in true_pareto]
    true_y = [particle[1] for particle in true_pareto]
    
    plt.scatter(true_x, true_y, color = 'red',s=70, label= 'Optimal pareto')
    plt.scatter(ref_point[0], ref_point[1], color = 'black',s=70, label= 'Reference point')

    plt.legend(prop={'weight':'bold', 'size': leg_fs}, scatterpoints=1, markerscale=2, fontsize=fs)
    plt.savefig("dense_myproblem3.png")

    print(f"Mean HV: {np.mean(hvs)} +- {np.std(hvs)}")

    if not os.path.exists('tmp'):
        os.makedirs('tmp')

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()