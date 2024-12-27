import optimizer
import numpy as np
from pymoo.indicators.hv import HV
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from optimizer import pso_environment_AEC
import supersuit as ss

num_agents = 30
num_iterations = 2
num_params = 2

lb = [-10.] * num_params
ub = [10.] * num_params


optimizer.Logger.setLevel('INFO')

def objective1(x):
    return 3 * np.cos(x[0])

def objective2(x):
    return 3 * np.cos(x[0] + np.pi / 2) + 1

optimizer.FileManager.working_dir = "tmp/policy_easy/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = False


objective = optimizer.ElementWiseObjective([objective1, objective2], sleep_time = 0.)

# test(objective, num_agents, num_iterations, lb, ub)
# test_time(objective, num_agents, num_iterations, lb, ub, [0.1, 0.5, 1, 2, 5, 10])
# test_random(objective, num_agents, num_iterations, lb, ub, max_time= 5)

pso = optimizer.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                      num_particles=num_agents,
                      inertia_weight=0.6, cognitive_coefficient=0.5, social_coefficient=1, initial_particles_position='random', topology="round_robin")

# run the optimization algorithm
pso.optimize(num_iterations)
ind = HV(ref_point=[5,5])
hv_pso = ind(np.array([p.fitness.tolist() for p in pso.pareto_front]))
plt.figure()
print(round(hv_pso / 54.06236516259962, 2))

pareto_front = pso.pareto_front
n_pareto_points = len(pareto_front)
pareto_x = [particle.fitness[0] for particle in pareto_front]
pareto_y = [particle.fitness[1] for particle in pareto_front]

plt.scatter(pareto_x, pareto_y, s=5)

plt.savefig('Pareto_try.png')

env_kwargs = {'pso' : pso,
                'pso_iterations' : num_iterations,
                'metric_reward' : 1,
                'metric_reward_hv_diff': 0, #130 max
                'evaluation_penalty' : -1, #-300/num_iterations,
                'not_dominated_reward' : 0,#600/num_iterations,
                'render_mode' : 'human'
                    }

env = pso_environment_AEC.env(**env_kwargs)

print(f"Starting training on {str(env.metadata['name'])}.")

# env = ss.pettingzoo_env_to_vec_env_v1(env)
# env = ss.concat_vec_envs_v1(env, 1, num_cpus=2, base_class="stable_baselines3")
model = PPO.load("myproblem_ag_50_iter_100_mr_1_mdr_0_p_0_ndr_1_model")

rewards = {"particle_" + str(i) for i in range(num_agents)}
env.reset()
num_actions = num_agents
for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            # print("Observation ", obs)

            # for a in env.agents:
            #     rewards[agent] += env.rewards[agent]
            if termination or truncation:
                plt.figure()
                fitnesses = np.array([p.fitness for p in env.env.pso.pareto_front])
                plt.scatter(fitnesses[:,0],fitnesses[:,1], s=5)
                n_pareto_points = len(env.env.pso.pareto_front)
                real_x = (np.linspace(0, 1, n_pareto_points))
                real_y = 1-np.sqrt(real_x)
                plt.scatter(real_x, real_y, s=5, c='red')
                plt.savefig("paretoRL.png")
                break
            else:
                actions = model.predict(obs, deterministic=True)[0]
                print(actions)
                num_actions += np.sum(actions)
                # print("Action ", act)

            env.step(actions)
            env.render()
            print("Iteration ", env.env.pso.iteration)
print("Tot evaluations: ", num_actions)
env.close()