import optimizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import math

num_agents = 100
num_iterations = 10
num_params = 10

lb = [0] * num_params
ub = [1] * num_params

optimizer.Logger.setLevel('DEBUG')
optimizer.Randomizer.rng = np.random.default_rng(45)

def zdt6_objective1(x):
    return 1 - (np.exp(-4 * x[0]) * np.power(np.sin(6 * np.pi * x[0]), 6))


def zdt6_objective2(x):
    f1 = 1 - (np.exp(-4 * x[0]) * np.power(np.sin(6 * np.pi * x[0]), 6))
    g = 1 + 9 * np.power(sum(x[1:]) / (len(x) - 1), 0.25)
    h = 1.0 - (f1 / g)**2
    f2 = g * h
    return f2

optimizer.FileManager.working_dir = "tmp/zdt6/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = False

if not os.path.exists(optimizer.FileManager.working_dir):
    os.makedirs(optimizer.FileManager.working_dir)

objective = optimizer.ElementWiseObjective([zdt6_objective1, zdt6_objective2])

pso = optimizer.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                      num_particles=num_agents,
                      inertia_weight=5, cognitive_coefficient=0, social_coefficient=0, initial_particles_position='random', topology='round_robin', exploring_particles=True, scaler=0.065)

# run the optimization algorithm
pso.optimize(num_iterations, max_iterations_without_improvement=5)

fig, ax = plt.subplots(figsize=(10,10))
pareto_front = pso.pareto_front
n_pareto_points = len(pareto_front)
pareto_x = [particle.fitness[0] for particle in pareto_front]
pareto_y = [particle.fitness[1] for particle in pareto_front]

real_x = np.linspace(0, 1, 100)
f1 = 1 - (np.exp(-4 * real_x) * np.power(np.sin(6 * np.pi * real_x), 6))
real_y = 1 - (f1 ** 2)

fig, ax = plt.subplots(figsize=(10,10))
plt.scatter(f1, real_y, s=70, c='red', label = 'Known optimal Pareto front')
plt.scatter(pareto_x, pareto_y, s=70, label = 'Pareto front')

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
plt.title('ZDT 6', fontweight='bold', fontsize=fs + 2)
plt.xlabel('Objective 1', fontweight='bold', fontsize=fs)
plt.ylabel('Objective 2', fontweight='bold', fontsize=fs)

ax.add_patch(plt.Circle((pareto_x[0], pareto_y[0]), radius=0.03,fill=False, linewidth = lw, color = '#1f77b4'))


plt.savefig('tmp/zdt6.png')

np.save("zdt6.npy",np.array([pareto_x, pareto_y]))