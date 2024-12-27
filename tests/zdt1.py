import optimizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from pymoo.indicators.hv import HV

import warnings
warnings.filterwarnings("error")

num_agents = 6
num_iterations = 10
num_params = 30

lb = [0.] * num_params
ub = [1.] * num_params

optimizer.Logger.setLevel('DEBUG')

def zdt1_objective(x):
    f1 = x[0]
    g = 1 + 9.0 / (len(x)-1) * sum(x[1:])
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h
    return f1, f2

use_reinforcement_learning = 0

optimizer.Randomizer.rng = np.random.default_rng(46)

optimizer.FileManager.working_dir = "tmp/zdt1/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = False

objective = optimizer.ElementWiseObjective(zdt1_objective, 2)

pso = optimizer.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                      num_particles=num_agents,
                      inertia_weight=0.4, cognitive_coefficient=1.5, social_coefficient=2,
                      initial_particles_position='random', exploring_particles=False, topology='round_robin')

# run the optimization algorithm
pso.optimize(num_iterations, max_iterations_without_improvement=5)

print(len(pso.pareto_front))
fig, ax = plt.subplots(figsize=(8,8))

pareto_front = pso.pareto_front
n_pareto_points = len(pareto_front)
pareto_x = [particle.fitness[0] for particle in pareto_front]
pareto_y = [particle.fitness[1] for particle in pareto_front]
real_x = (np.linspace(0, 1, n_pareto_points))
real_y = 1-np.sqrt(real_x)
plt.scatter(real_x, real_y, s=70, c='red', label = 'Known optimal Pareto front')
plt.scatter(pareto_x, pareto_y, s=70, label= 'Pareto front')

ind = HV(ref_point=[5,5])
hv_real = ind(np.array([[real_x[i], real_y[i]] for i in range(len(real_x))]))
hv_pso = ind(np.array([p.fitness.tolist() for p in pso.pareto_front]))
print(hv_real)
print(round(hv_pso / hv_real, 2))

lw = 4
ls = 20
fs = 22
leg_fs = 16
ax.spines['top'].set_linewidth(lw)
ax.spines['right'].set_linewidth(lw)
ax.spines['left'].set_linewidth(lw)
ax.spines['bottom'].set_linewidth(lw)
ax.tick_params(axis='both', which='major', labelsize=ls, width=2)
plt.xticks(ax.get_xticks()[1:-1], weight = 'bold')
plt.yticks(ax.get_yticks()[1:-1], weight = 'bold')
plt.legend(prop={'weight':'bold', 'size': leg_fs}, scatterpoints=1, markerscale=2, fontsize=fs)
plt.title('ZDT 1', fontweight='bold', fontsize=fs + 2)
plt.xlabel('Objective 1', fontweight='bold', fontsize=fs)
plt.ylabel('Objective 2', fontweight='bold', fontsize=fs)

if not os.path.exists('tmp'):
    os.makedirs('tmp')
plt.savefig('tmp/zdt1.png', bbox_inches='tight')
plt.close()

np.save("zdt1.npy",np.array([pareto_x, pareto_y]))
