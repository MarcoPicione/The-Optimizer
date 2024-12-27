import optimizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

num_agents = 200
num_iterations = 2000
num_params = 10

lb = [0.] + [-5.] * (num_params - 1)
ub = [1.] + [5.] * (num_params - 1)

optimizer.Logger.setLevel('INFO')

optimizer.Randomizer.rng = np.random.default_rng(45)

def zdt4_objective1(x):
    return x[0]


def zdt4_objective2(x):
    f1 = x[0]
    g = 1.0 + 10 * (len(x) - 1) + sum([i**2 - 10 * np.cos(4 * np.pi * i) for i in x[1:]])
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h
    return f2

optimizer.FileManager.working_dir = "tmp/zdt4/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = False

if not os.path.exists(optimizer.FileManager.working_dir):
    os.makedirs(optimizer.FileManager.working_dir)

objective = optimizer.ElementWiseObjective([zdt4_objective1, zdt4_objective2])

pso = optimizer.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                      num_particles=num_agents,
                      inertia_weight=0.2, cognitive_coefficient=1, social_coefficient=2,
                      initial_particles_position='random', topology = 'round_robin',
                      exploring_particles=True, scaler=0.065)

# run the optimization algorithm
pso.optimize(num_iterations, max_iterations_without_improvement=6)

fig, ax = plt.subplots(figsize=(8,8))

pareto_front = pso.pareto_front
n_pareto_points = len(pareto_front)
pareto_x = [particle.fitness[0] for particle in pareto_front]
pareto_y = [particle.fitness[1] for particle in pareto_front]

real_x = (np.linspace(0, 1, 100))
real_y = 1 - np.sqrt(real_x)
plt.scatter(real_x, real_y, s=70, c='red', label = 'Known optimal Pareto front')
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
plt.title('ZDT 4', fontweight='bold', fontsize=fs + 2)
plt.xlabel('Objective 1', fontweight='bold', fontsize=fs)
plt.ylabel('Objective 2', fontweight='bold', fontsize=fs, labelpad=-10)

plt.savefig('tmp/zdt4.png', bbox_inches='tight')

np.save("zdt4.npy",np.array([pareto_x, pareto_y]))