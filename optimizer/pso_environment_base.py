import numpy as np
from gymnasium.spaces import Discrete, Box
import copy
from optimizer import Randomizer
from optimizer.reinforcement_learning_utils import observe_list, find_new_bad_points
from pymoo.indicators.hv import HV
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import optimizer


class pso_environment_base:
    def __init__(self, pso, pso_iterations, metric_reward, metric_reward_hv_diff, evaluation_penalty, not_dominated_reward, radius_scaler = 0.03, render_mode = None):
        
        self.possible_pso = pso
        self.pso_iterations = pso_iterations
        self.num_agents = self.possible_pso.num_particles
        self.metric_reward = metric_reward
        self.metric_reward_hv_diff = metric_reward_hv_diff
        self.evaluation_penalty = evaluation_penalty
        self.not_dominated_reward = not_dominated_reward

        self.last_dones = [False for _ in range(self.num_agents)]
        self.last_obs = [None for _ in range(self.num_agents)]
        self.last_rewards = [np.float64(0) for _ in range(self.num_agents)]

        self.render_mode = render_mode
        self._seed()

        # state stuff
        self.ref_point = [15,5] #[13,5]#[600,750]#[14.15,4.21] # #[674.28, 462.14]
        self.ind = HV(ref_point=self.ref_point)
        upper_bounds = np.array(self.possible_pso.upper_bounds)
        lower_bounds = np.array(self.possible_pso.lower_bounds)
        self.num_parameters = len(lower_bounds)
        self.max_dist = np.linalg.norm(upper_bounds - lower_bounds)
        self.radius = radius_scaler * self.max_dist
        self.hv_max = 171 #1607014
        self.num_episode = -1
        self.observed_states = []
        self.reset()
        
        self.get_spaces()
        print("Created environment")

    def get_spaces(self):
        # Define the action and observation spaces for all of the agents
        len_obs = 2
        low = np.array([0.] * len_obs)
        # high = np.array([self.num_agents * self.pso_iterations] * 2 + [self.num_agents] * 3)
        high = np.array([self.pso_iterations + 1] * len_obs)
        obs_space = Box(low = low, high = high, shape = (len_obs,), dtype=np.float32)
        act_space = Discrete(2)

        self.observation_space = [obs_space for i in range(self.num_agents)]
        self.action_space = [act_space for i in range(self.num_agents)]

    def _seed(self, seed=None):
        self.np_random = Randomizer.rng

    def reset(self):
        # seed = np.random.randint(10000)

        # optimizer.Randomizer.rng = np.random.default_rng(seed)
        # pso = copy.deepcopy(self.possible_pso)
        # pso.optimize(self.pso_iterations)
        # self.hv_max = self.ind(np.array([p.fitness for p in pso.pareto_front]))
        #self.hv_max = 1175602 #1367769.4990155988#11433589.0972680415
        # self.hv_max = 145.50476407132953
        # self.hv_max = 1415137.5935035301
        # self.hv_max = 1611731.049297563
        # optimizer.Randomizer.rng = np.random.default_rng(seed)
        self.pso = copy.deepcopy(self.possible_pso)

        # Set up the reward
        self.last_rewards = [np.float64(0) for _ in range(self.num_agents)]
        self.action_list = []
        self.good_points = []
        self.bad_points = []

        # Evaluate all particles to begin with
        self.pso.step()
        self.pso.iteration -= 1
        print(f"{self.pso.iteration}, {self.possible_pso.iteration}")
        self.hv = self.ind(np.array([p.fitness for p in self.pso.pareto_front]))
        self.prev_hv = self.hv
        self.starting_hv = self.hv
        print(f"STARTING HV {self.hv}")
        
        # Get observation
        obs_list = self.observe_list()
        self.last_obs = obs_list
        self.last_dones = [False for _ in range(self.num_agents)]
        self.invalid_actions = [[] for _ in range(self.num_agents)]
        
        return obs_list

    def step(self, action, agent_id, is_last):
        # save the action to give the reward
        self.action_list.append(action)
        p = self.pso.particles[agent_id]
        
        # Execute actions
        p.evaluated = action
        optimization_output = self.pso.objective.evaluate(np.array([p.position]))[0] if action else [np.inf] * len(p.fitness)
        p.set_fitness(optimization_output)

        # Give negative reward if evaluated
        # agent_obs = self.last_obs[agent_id]
        # if agent_obs[0] != 0 or agent_obs[1] != 0:
        self.last_rewards[agent_id] = 0#self.evaluation_penalty * action

        if is_last:
            # Update pareto
            pareto_front_old = copy.deepcopy(self.pso.pareto_front)
            dominated, crowding_distances = self.pso.update_pareto_front(self.action_list)

            # Assign reward if not dominated
            counter_evaluated = 0
            counter_possible_evaluations = 0
            for id in range(self.num_agents):
                if self.last_obs[id]!=[0,0]:
                    counter_possible_evaluations += 1
                    counter_evaluated+=1 if self.action_list[id] else 0
                self.last_rewards[id] += self.not_dominated_reward if not dominated[id] else 0
                self.last_rewards[id] += self.evaluation_penalty * self.action_list[id]
            
            if(counter_evaluated != 0): 
                print(round(counter_evaluated / counter_possible_evaluations,2))
            else:
                print(f"No particle evaluated among {counter_possible_evaluations}")
                # if sum(self.action_list) < 0.1 * self.num_agents:
                #     self.last_rewards[id] += -1000

            # Assign reward if hv improves
            # self.hv = self.ind(np.array([p.fitness for p in self.pso.pareto_front]))
            # for id in range(self.num_agents):
            #     # self.last_rewards[id] += self.metric_reward_hv_diff * (self.hv - self.prev_hv)
            #     # self.last_rewards[id] += self.metric_reward_hv_diff * np.exp(500 * (self.hv / 54.06236516259962 -1 )) - self.metric_reward_hv_diff * np.exp(500 * (self.prev_hv / 54.06236516259962 -1 ))
            #     # self.last_rewards[id] += self.metric_reward_hv_diff * np.exp(500 * (self.hv / self.hv_max -1 )) - self.metric_reward_hv_diff * np.exp(500 * (self.prev_hv / self.hv_max -1 ))
            #     self.last_rewards[id] += self.metric_reward_hv_diff * (np.exp(5 * (self.hv / self.hv_max)) - np.exp(5*(self.prev_hv / self.hv_max)))
            # self.prev_hv = self.hv

            # If a particle is dominated and was evaluated add it to bad points list.      
            self.bad_points = self.pso.bad_points #+= find_new_bad_points(self.pso.particles, dominated, pareto_front_old, dominated_in_pareto, self.action_list)

            # Update velocities and positions
            for particle in self.pso.particles:
                particle.update_velocity(self.pso.pareto_front,
                                            crowding_distances,
                                            self.pso.inertia_weight,
                                            self.pso.cognitive_coefficient,
                                            self.pso.social_coefficient)
                particle.update_position(self.pso.lower_bounds, self.pso.upper_bounds)

            # If is last iteration assign Hyper volume reward to all agents
            if self.pso.iteration == self.pso_iterations - 1:
                self.hv = self.ind(np.array([p.fitness for p in self.pso.pareto_front]))
                # x = 1 - (self.starting_hv / self.hv)
                # if self.hv > self.hv_max:
                #     print(f"HV self {self.hv}")
                #     self.hv_max = self.hv
                #     print(f"HV MAX {self.hv_max}")
                # if self.num_episode > 1:
                x = self.hv/self.hv_max 
                # else:
                #     print("FIRST EPISODE") 
                #     x=1
                print(f"final hv {x}")
                for id in range(self.num_agents):
                    self.last_rewards[id] += self.metric_reward * np.exp(5 * x)

            # End of pso iteration
            self.pso.iteration += 1
            self.action_list = []
            self.invalid_actions = [[] for _ in range(self.num_agents)]

            # Generate new observations
            obs_list = self.observe_list()
            self.last_obs = obs_list

            for particle in self.pso.particles:
                particle.evaluated = -1

            # plt.figure()
            # pareto_x = [particle.fitness[0] for particle in self.pso.pareto_front]
            # pareto_y = [particle.fitness[1] for particle in self.pso.pareto_front]
            # plt.scatter(pareto_x, pareto_y, s=5)
            # plt.savefig("Pareto_try")

        return self.observe(agent_id)

    def observe(self, agent_id):
        return np.array(self.last_obs[agent_id], dtype=np.float32)

    def observe_list(self):
        observations = observe_list(self.pso,
                            np.array([p.position for p in self.pso.pareto_front]),
                            np.array(self.bad_points),
                            self.radius,
                            self.max_dist,
                            self.pso_iterations
                            )
        for i, obs in enumerate(observations):
            if obs[0] > 0: self.invalid_actions[i].append(0)
            # if obs not in self.observed_states:
            #     self.observed_states.append(obs)
        return observations
    
    def action_masks(self):
        return [action not in self.invalid_actions for action in self.possible_actions]
                                     
    def render(self):
        # To be fixed
        if self.num_parameters == 2:
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.scatter([p.position[0] for p in self.pso.particles],[p.position[1] for p in self.pso.particles], color = 'black', marker = '.', s = 200,  zorder=3)
            plt.scatter([p[0] for p in self.bad_points], [p[1] for p in self.bad_points], color = 'blue', marker = 'X', s = 100, zorder=2)
            plt.scatter([p.position[0] for p in self.pso.pareto_front], [p.position[1] for p in self.pso.pareto_front], color = 'green', marker = '*', s = 150,  zorder=1)

            # rect = patches.Rectangle((-4 * np.pi + np.pi / 2, -10), np.pi / 2, 20, linewidth=1, edgecolor=None, facecolor='green', alpha=0.2)
            # ax.add_patch(rect)

            # rect = patches.Rectangle((-2 * np.pi + np.pi / 2, -10), np.pi / 2, 20, linewidth=1, edgecolor=None, facecolor='green', alpha=0.2)
            # ax.add_patch(rect)

            # rect = patches.Rectangle((np.pi / 2, -10), np.pi / 2, 20, linewidth=1, edgecolor=None, facecolor='green', alpha=0.2)
            # ax.add_patch(rect)

            # rect = patches.Rectangle((2 * np.pi + np.pi / 2, -10), np.pi / 2, 20, linewidth=1, edgecolor=None, facecolor='green', alpha=0.2)
            # ax.add_patch(rect)

            # rect = patches.Rectangle((4 * np.pi + np.pi / 2, -10), np.pi / 2, 20, linewidth=1, edgecolor=None, facecolor='green', alpha=0.2)
            # ax.add_patch(rect)

            for p in self.pso.particles:
                # c = plt.Circle(p.position, self.radius, color = 'black', fill = False, linewidth = 2, zorder=3)
                new_position=p.position + p.velocity
                # plt.plot([p.position[0], new_position[0]], [p.position[1], new_position[1]], color='black',  lw=2)
                plt.annotate('', xy=(new_position[0], new_position[1]), xytext=(p.position[0], p.position[1]),arrowprops=dict(arrowstyle="->", lw=2, color='black', alpha=1), zorder=0)

                # ax.add_patch(c)
            plt.xlim(-10,10)
            plt.ylim(-10,10)
            lw = 4
            ls = 20
            fs = 22
            leg_fs = 16
            ax.spines['top'].set_linewidth(lw)
            ax.spines['right'].set_linewidth(lw)
            ax.spines['left'].set_linewidth(lw)
            ax.spines['bottom'].set_linewidth(lw)
            ax.tick_params(axis='both', which='major', labelsize=ls, width=2)
            plt.xticks(ax.get_xticks(), weight = 'bold')
            plt.yticks(ax.get_yticks(), weight = 'bold')
            # plt.xlabel('Objective 1', fontweight='bold', fontsize=fs)
            # plt.ylabel('Objective 2', fontweight='bold', fontsize=fs)
            plt.show()
            plt.savefig(f"Iteration {self.pso.iteration}.png")
            plt.close()
        else:
            print("No implementation found for render mode")        