import optimizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from stable_baselines3 import PPO
from optimizer import pso_environment_AEC
import supersuit as ss


import warnings
warnings.filterwarnings("error")

num_agents = 50
num_iterations = 200
num_params = 2

lb = [-10.] * num_params
ub = [10.] * num_params

optimizer.Logger.setLevel('INFO')

def objective1(x):
    return 3 * np.cos(x[0])

def objective2(x):
    return 3 * np.cos(x[0] + np.pi / 2) + 1

use_reinforcement_learning = 0

optimizer.Randomizer.rng = np.random.default_rng(43)

optimizer.FileManager.working_dir = "tmp/policy_easy/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = False


objective = optimizer.ElementWiseObjective([objective1, objective2])

pso = optimizer.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                      num_particles=num_agents,
                      inertia_weight=0.6, cognitive_coefficient=1, social_coefficient=2, initial_particles_position='random', incremental_pareto=True, 
                      use_reinforcement_learning=use_reinforcement_learning)

# run the optimization algorithm

env_kwargs = {'pso' : pso,
                'pso_iterations' : num_iterations,
                'metric_reward' : 10,
                'evaluation_penalty' : -1,
                'not_dominated_reward' : 0,
                'render_mode' : 'None'
                    }
env = pso_environment_AEC.env(**env_kwargs)
env = pso_environment_AEC.parallel_env(**env_kwargs)

print("Starting MOPSO with RL")
model = PPO.load("model")
rewards = {agent: 0 for agent in env.possible_agents}
observations = env.reset()[0]
print("OBSERVATIONS: ", observations)
num_actions = num_agents

it = 0

while env.agents:
        # this is where you would insert your policy
        actions = model.predict(list(observations.values()), deterministic=True)[0]
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        print("Iteration ", it)
        num_actions += sum(list(actions.values()))
        # print("Iteration ", env.aec_env.env.pso.iteration)
        # print(rewards)
        # print(terminations)
        # input()
        it += 1

env = env.aec_env
print("Tot evaluations: ", num_actions)
print("Fraction evaluations: ", num_actions / (num_agents * (num_iterations + 1)))
fitnesses = np.array([p.fitness for p in env.env.pso.pareto_front])
# env.close()

# for agent in env.agent_iter():
#     obs, reward, termination, truncation, info = env.last()

#     for a in env.agents:
#         rewards[a] += env.rewards[a]

#     if termination or truncation:
#         break
#     else:
#         actions = model.predict(obs, deterministic=True)[0]
#         print(actions)
#         num_actions += np.sum(actions)

#     env.step(actions)
#     print("Iteration ", env.env.pso.iteration)

print("Starting MOPSO without RL")
print("PARETO ", len(pso.pareto_front))
pso.optimize(num_iterations)

plt.figure()
pareto_x = [particle.fitness[0] for particle in pso.pareto_front]
pareto_y = [particle.fitness[1] for particle in pso.pareto_front]
plt.scatter(pareto_x, pareto_y, s=5, c='red', label = "MOPSO")
plt.scatter(fitnesses[:,0],fitnesses[:,1], s=5, label = "Reinforcement Learning")
plt.legend()
plt.savefig("paretoRL.png")