import optimizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import copy
from pymoo.indicators.hv import HV
from multiprocessing import Pool


num_agents = 100
num_iterations = 300
num_params = 10

lb = [0.] + [-5.] * (num_params - 1)
ub = [1.] + [5.] * (num_params - 1)

optimizer.Logger.setLevel('ERROR')

def zdt4_objective1(x):
    return x[0]


def zdt4_objective2(x):
    f1 = x[0]
    g = 1.0 + 10 * (len(x) - 1) + sum([i**2 - 10 * np.cos(4 * np.pi * i) for i in x[1:]])
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h
    return f2

max_it = 2

def evaluate_scaler(scaler):
    seeds = list(range(50,150))
    ref_point = [50, 50]
    ind = HV(ref_point=ref_point)
    inertia_weight = 0.4
    cognitive_coefficient = 1
    social_coefficient = 2
    hvs = np.empty(len(seeds))

    for j, s in enumerate(seeds):
        print(f"Starting {scaler} scaler with seed {s}")
        optimizer.Randomizer.rng = np.random.default_rng(s)  
        pso = optimizer.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                            num_particles=num_agents,
                            inertia_weight=inertia_weight, cognitive_coefficient=cognitive_coefficient, social_coefficient=social_coefficient, 
                            initial_particles_position='random',
                            topology = 'round_robin',
                            exploring_particles = True, scaler = scaler)

        # run the optimization algorithm
        pso.optimize(num_iterations, max_it)
        pareto_front = pso.pareto_front
        hv = ind(np.array([p.fitness for p in pareto_front]))
        hvs[j] = hv
        print(hv)

    mean = np.mean(hvs)
    err = np.std(hvs)

    return (mean, err)

optimizer.FileManager.working_dir = "tmp/zdt1/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = False

objective = optimizer.ElementWiseObjective([zdt4_objective1, zdt4_objective2])

# Tune scaler
scalers = np.linspace(0.01, 0.3, 100)

with Pool(10) as p:
    res_objs = p.map(evaluate_scaler, scalers)

means = [x[0] for x in res_objs]
stds = [x[1] for x in res_objs]

np.save(f"./tests/means/means_{max_it}.npy", means)
np.save(f"./tests/stds/stds_{max_it}.npy", stds)

# fig, axs = plt.subplots()
# axs.errorbar(scalers, means, stds, label='Trained policy')
# axs.set_xlabel('Scaler')
# axs.set_ylabel('Mean hypervolume')
# axs.legend()