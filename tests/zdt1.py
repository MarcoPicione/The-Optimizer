import optimizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

import warnings
warnings.filterwarnings("error")

num_agents = 100
num_iterations = 200
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

use_reinforcement_learning = 0

optimizer.Randomizer.rng = np.random.default_rng(42)

optimizer.FileManager.working_dir = "tmp/zdt1/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = True

if use_reinforcement_learning: evaluation_mask = optimizer.FileManager.load_csv("useful_evaluations_0.csv")
objective = optimizer.ElementWiseObjective([zdt1_objective1, zdt1_objective2])

pso = optimizer.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                      num_particles=num_agents,
                      inertia_weight=0.6, cognitive_coefficient=1, social_coefficient=2, initial_particles_position='random', incremental_pareto=True, 
                      use_reinforcement_learning=use_reinforcement_learning, masks = evaluation_mask if use_reinforcement_learning else None)

# run the optimization algorithm
pso.optimize(num_iterations)

print(len(pso.pareto_front))
fig, ax = plt.subplots()

pareto_front = pso.pareto_front
n_pareto_points = len(pareto_front)
pareto_x = [particle.fitness[0] for particle in pareto_front]
pareto_y = [particle.fitness[1] for particle in pareto_front]
real_x = (np.linspace(0, 1, n_pareto_points))
real_y = 1-np.sqrt(real_x)
plt.scatter(real_x, real_y, s=5, c='red')
plt.scatter(pareto_x, pareto_y, s=5)

if not os.path.exists('tmp'):
    os.makedirs('tmp')
plt.savefig('tmp/pf_'+str(use_reinforcement_learning)+'.png')

print("TEST")
# evaluation_mask0 = optimizer.FileManager.load_csv("useful_evaluations_0.csv")
# evaluation_mask1 = optimizer.FileManager.load_csv("useful_evaluations_1.csv")

# for i in range(len(evaluation_mask0)):
#     if not np.allclose(evaluation_mask0[i], evaluation_mask1[i]):
#         print (i)

# opt_0 = np.load('optimization_output_0.npy')
# opt_1 = np.load('optimization_output_1.npy')
# for i in range(len(opt_0)):
#     if not np.allclose(opt_0[i], opt_1[i]):
#         print (i)
