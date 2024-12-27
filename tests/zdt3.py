import optimizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

num_agents = 100
num_iterations = 100
num_params = 30

lb = [0.] * num_params
ub = [1.] * num_params


def zdt3_objective1(x):
    return x[0]


def zdt3_objective2(x):
    f1 = x[0]
    g = 1.0 + 9.0 * sum(x[1:]) / (len(x) - 1)
    h = (1.0 - np.power(f1 * 1.0 / g, 0.5) -
         (f1 * 1.0 / g) * np.sin(10 * np.pi * f1))
    f2 = g * h
    return f2


optimizer.FileManager.working_dir = "tmp/zdt3/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = False

objective = optimizer.ElementWiseObjective([zdt3_objective1, zdt3_objective2])

pso = optimizer.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                      num_particles=num_agents,
                      inertia_weight=1., cognitive_coefficient=2, social_coefficient=2, initial_particles_position='random', topology = 'round_robin')

# run the optimization algorithm
pso.optimize(num_iterations)

fig, ax = plt.subplots(figsize=(9,8))

pareto_front = pso.pareto_front
n_pareto_points = len(pareto_front)
pareto_x = [particle.fitness[0] for particle in pareto_front]
pareto_y = [particle.fitness[1] for particle in pareto_front]

regions = [[0, 0.0830015349],
           [0.182228780, 0.2577623634],
           [0.4093136748, 0.4538821041],
           [0.6183967944, 0.6525117038],
           [0.8233317983, 0.8518328654]]

pf = []

for r in regions:
    x1 = np.linspace(r[0], r[1], int(n_pareto_points / len(regions)))
    x2 = 1 - np.sqrt(x1) - x1 * np.sin(10 * np.pi * x1)
    pf.append([x1, x2])

real_x = np.concatenate([x for x, _ in pf])
real_y = np.concatenate([y for _, y in pf])

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
plt.title('ZDT 3', fontweight='bold', fontsize=fs + 2)
plt.xlabel('Objective 1', fontweight='bold', fontsize=fs)
plt.ylabel('Objective 2', fontweight='bold', fontsize=fs, labelpad=-15)

if not os.path.exists('tmp'):
    os.makedirs('tmp')
plt.savefig('tmp/zdt3.png')
np.save("zdt3.npy",np.array([pareto_x, pareto_y]))
