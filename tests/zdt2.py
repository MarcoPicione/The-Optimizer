import optimizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

num_agents = 100
num_iterations = 200
num_params = 30

lb = [0.] * num_params
ub = [1.] * num_params


def zdt2_objective1(x):
    return x[0]


def zdt2_objective2(x):
    f1 = x[0]
    g = 1.0 + 9.0 * sum(x[1:]) / (len(x) - 1)
    h = 1.0 - np.power((f1 * 1.0 / g), 2)
    f2 = g * h
    return f2

optimizer.Randomizer.rng = np.random.default_rng(46)
optimizer.FileManager.working_dir = "tmp/zdt2/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = False

objective = optimizer.ElementWiseObjective([zdt2_objective1, zdt2_objective2])

pso = optimizer.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                      num_particles=num_agents,
                      inertia_weight=0.5, cognitive_coefficient=1.5, social_coefficient=2, initial_particles_position='random', topology="round_robin", exploring_particles=True, scaler=0.065)

# run the optimization algorithm
pso.optimize(num_iterations, max_iterations_without_improvement=5)

fig, ax = plt.subplots(figsize=(8,8))

pareto_front = pso.pareto_front
n_pareto_points = len(pareto_front)
pareto_x = [particle.fitness[0] for particle in pareto_front]
pareto_y = [particle.fitness[1] for particle in pareto_front]

real_x = (np.linspace(0, 1, n_pareto_points))
real_y = 1 - np.power(real_x, 2)
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
plt.title('ZDT 2', fontweight='bold', fontsize=fs + 2)
plt.xlabel('Objective 1', fontweight='bold', fontsize=fs)
plt.ylabel('Objective 2', fontweight='bold', fontsize=fs)

if not os.path.exists('tmp'):
    os.makedirs('tmp')
plt.savefig('tmp/zdt2.png')

np.save("zdt2.npy",np.array([pareto_x, pareto_y]))