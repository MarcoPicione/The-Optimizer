import optimizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

num_agents = 200
num_iterations = 200
num_params = 11

lb = [0] * num_params
ub = [1073741823] + [31] * (num_params - 1)


def zdt5_objective1(x):
    return 1 + u(x[0])


def zdt5_objective2(x):
    f1 = 1 + u(x[0])
    g = sum([v(u(i)) for i in x[1:]])
    h = 1.0 / f1
    f2 = g * h
    return f2

def u(x):
    c = 0
    while x:
        c += 1
        x &= x - 1
    return c


def v(x):
    un = u(x)
    if un < 5:
        return 2 + un
    elif un == 5:
        return 1
    else:
        print('upsi')
        return 0


optimizer.FileManager.working_dir = "tmp/zdt5/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = False

if not os.path.exists(optimizer.FileManager.working_dir):
    os.makedirs(optimizer.FileManager.working_dir)

objective = optimizer.ElementWiseObjective([zdt5_objective1, zdt5_objective2])

pso = optimizer.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                      num_particles=num_agents,
                      inertia_weight=0.9, cognitive_coefficient=2, social_coefficient=2, initial_particles_position='random', topology='round_robin', exploring_particles=False, scaler=0.065)

# run the optimization algorithm
pso.optimize(num_iterations, max_iterations_without_improvement=5)

fig, ax = plt.subplots(figsize=(10,10))

pareto_front = pso.pareto_front
n_pareto_points = len(pareto_front)
pareto_x = [particle.fitness[0] for particle in pareto_front]
pareto_y = [particle.fitness[1] for particle in pareto_front]

real_x = np.array([u(x) for x in np.linspace(0, 1073741823, 100000, dtype=np.int_)])
real_y = [10 / (1 + x) for x in real_x]
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
plt.title('ZDT 5', fontweight='bold', fontsize=fs + 2)
plt.xlabel('Objective 1', fontweight='bold', fontsize=fs)
plt.ylabel('Objective 2', fontweight='bold', fontsize=fs)

plt.savefig('tmp/zdt5.png')

np.save("zdt5.npy",np.array([pareto_x, pareto_y]))