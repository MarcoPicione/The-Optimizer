from optimizer.metrics import hyper_volume
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches

pareto_x = (np.linspace(0, 1, 5))
pareto_y = -pareto_x + 1
pareto = np.column_stack((pareto_x, pareto_y))

print("Pareto front: \n", pareto)
ref_point = [1.25, 1.25]
print("Hyper volume ", hyper_volume(pareto, ref_point))

fig, ax = plt.subplots(figsize=(5,5))

for x, y in zip(pareto_x, pareto_y):
    width = ref_point[0] - x
    height = ref_point[1] - y
    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor=None,  hatch='/', facecolor='green', alpha=0.5)
    ax.add_patch(rect)
    ref_point[1] = y

ax.plot([0, 1], [1, 1], linestyle='--', color='black', alpha=1)
ax.plot([1, 1], [0, 1], linestyle='--', color='black', alpha=1)

ax.plot([0, 0.25], [1, 1], color='black', alpha=1, zorder=1)
ax.plot([0.25, 0.5], [0.75, 0.75], color='black', alpha=1, zorder=1)
ax.plot([0.5, 0.75], [0.5, 0.5], color='black', alpha=1, zorder=1)
ax.plot([0.75, 1], [0.25, 0.25], color='black', alpha=1, zorder=1)
ax.plot([1, 1.25], [0, 0], color='black', alpha=1, zorder=1)

ax.plot([1, 1], [0, 0.25], color='black', alpha=1, zorder=1)
ax.plot([0.75, 0.75],[0.25, 0.5], color='black', alpha=1, zorder=1)
ax.plot([0.5, 0.5],[0.5, 0.75], color='black', alpha=1, zorder=1)
ax.plot([0.25, 0.25],[0.75, 1], color='black', alpha=1, zorder=1)
ax.plot([0, 0],[1, 1.25], color='black', alpha=1, zorder=1)

ax.plot([0, 1.25], [1.25, 1.25], color='black', alpha=1, zorder=1)
ax.plot([1.25, 1.25],[0, 1.25], color='black', alpha=1, zorder=1)

plt.scatter(pareto_x, pareto_y, label = "Pareto front", zorder=2, edgecolor='darkblue')
plt.scatter(1.25,1.25, color = 'r', label = "Reference point", zorder=2, edgecolor='darkblue')
plt.scatter(1,1,color='black',label='Nadir point', zorder=2)

ax.set_xlabel("Objective 1", fontweight='bold')
ax.set_ylabel("Objective 2", fontweight='bold')
legend_patch = patches.Patch(facecolor='green', edgecolor=None, hatch='/')
plt.legend()
handles, labels = ax.get_legend_handles_labels()
handles.append(legend_patch)
labels.append('Hyper volume')
ax.legend(handles=handles, labels=labels,prop={'weight':'bold'})

ax.spines['top'].set_linewidth(2)   # Spessore bordo superiore
ax.spines['right'].set_linewidth(2) # Spessore bordo destro
ax.spines['left'].set_linewidth(2)  # Spessore bordo sinistro
ax.spines['bottom'].set_linewidth(2) # Spessore bordo inferiore

ax.tick_params(axis='both', which='major', labelsize=10, width=2)  # Spessore delle linee dei tick
# ax.xaxis.set_tick_params(labelsize=10, fontweight='bold')
# ax.yaxis.set_tick_params(labelsize=10, fontweight='bold')
plt.xticks(ax.get_xticks()[1:-1], weight = 'bold')
plt.yticks(ax.get_yticks()[1:-1], weight = 'bold')

plt.savefig("HV.png")