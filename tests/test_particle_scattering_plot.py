import optimizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import copy
from pymoo.indicators.hv import HV
from multiprocessing import Pool

means = []
stds = []
names = []

indexes = [4,5,6]

zdt = 4

mean_max = -np.inf
mean_min = np.inf
for f in os.listdir(f"./tests/means_zdt{zdt}"):
    id = int(f.replace('means_','').replace('.npy',''))
    if id in indexes:
        names.append(id)
        file = np.load(f'./tests/means_zdt{zdt}/' + f)
        means.append(file)
        M = max(file)
        mean_max = M if M > mean_max else mean_max

# means = [np.array(m) / mean_max for m in means]
# for i in means:
#     m = min(i)
#     mean_min = m if m < mean_min else mean_min

# means = [np.array(m) - mean_min for m in means]

for f in os.listdir(f"./tests/stds_zdt{zdt}"):
    if id in indexes:
        stds.append(np.load(f'./tests/stds_zdt{zdt}/' + f))

scalers = np.linspace(0.01, 0.3, 100)    
fig, ax =plt.subplots(figsize=(12, 8))

# for i in np.argsort(names):
#     plt.plot(scalers, means[i], label=f'{names[i]} iterations')
    # plt.errorbar(scalers, means[i], stds[i], label=f'Max iterations =  {names[i]}')

# if zdt == 1:
#     plt.hlines(y=0.8, xmin=scalers[0], xmax=scalers[-1], color='black', linestyle='-', label = 'No exploring particles')

# else:
#     plt.hlines(y=2020, xmin=scalers[0], xmax=scalers[-1], color='black', linestyle='-', label = 'No exploring particles')
# plt.hlines(y=2250, xmin=scalers[0], xmax=scalers[-1], color='black', linestyle='--', label = 'No exploring')

baseline = 0.8 if zdt==1 else 2020
lw = 4
ls = 20
fs = 22
leg_fs = 16

for i in np.argsort(names):
    array = (means[i]-baseline) / stds[i]
    plt.plot(scalers, array, label=f'{names[i]} no improving iterations', linewidth=lw)

plt.hlines(y=0, xmin=scalers[0], xmax=scalers[-1], color='black', linestyle='-', label = 'Baseline', linewidth = lw)
ax.spines['top'].set_linewidth(lw)
ax.spines['right'].set_linewidth(lw)
ax.spines['left'].set_linewidth(lw)
ax.spines['bottom'].set_linewidth(lw)
ax.tick_params(axis='both', which='major', labelsize=ls, width=2)
plt.xticks(ax.get_xticks()[1:-1], weight = 'bold')
plt.yticks(ax.get_yticks()[1:-1], weight = 'bold')
legend = plt.legend(prop={'weight':'bold', 'size': leg_fs}, scatterpoints=1, markerscale=2, fontsize=fs)
plt.title(f'ZDT {zdt}', fontweight='bold', fontsize=fs + 2)
plt.xlabel('Scaler', fontweight='bold', fontsize=fs)
plt.ylabel('Normalized hypervolume', fontweight='bold', fontsize=fs)
legend.get_frame().set_alpha(None)
# if zdt == 4:
#     plt.ylim((1900, 2400))
# plt.xlim((0, 0.15))
plt.savefig(f"plotScaler_zdt{zdt}.png")
plt.close()