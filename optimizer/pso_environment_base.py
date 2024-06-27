import numpy as np
from gymnasium.spaces import Discrete, Box, Dict
import copy
from .metrics import hyper_volume
from optimizer import Randomizer
import time
from pymoo.indicators.hv import HV
import math
from copy import deepcopy
from numba import njit, jit, prange
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import optimizer

class pso_environment_base:
    def __init__(self, pso, pso_iterations, metric_reward, evaluation_penalty, not_dominated_reward, render_mode = None):
        
        self.possible_pso = pso
        self.pso_iterations = pso_iterations
        self.num_agents = self.possible_pso.num_particles
        self.metric_reward = metric_reward
        self.evaluation_penalty = evaluation_penalty
        self.not_dominated_reward = not_dominated_reward

        self.last_dones = [False for _ in range(self.num_agents)]
        self.last_obs = [None for _ in range(self.num_agents)]
        self.last_rewards = [np.float64(0) for _ in range(self.num_agents)]

        self.render_mode = render_mode
        self.get_spaces()
        self._seed()

        # state stuff
        self.ref_point = [5, 5]
        self.angle = 20
        self.ind = HV(ref_point=self.ref_point)
        upper_bounds = np.array(self.possible_pso.upper_bounds)
        lower_bounds = np.array(self.possible_pso.lower_bounds)
        self.max_dist = np.linalg.norm(upper_bounds - lower_bounds)
        self.radius = 0.03 * self.max_dist
        print("Max ", self.max_dist)

    def get_spaces(self):
        """Define the action and observation spaces for all of the agents."""

        len_obs = 6
        # low = np.array([0.] * len_obs)
        # high = np.array([np.inf] * len_obs)

        # obs_space = Dict({
        #     # 'distance_from_good_points': Box(low = 0, high = np.inf, shape = (1,), dtype=np.float32),
        #     # 'num_good_points': Discrete(self.num_agents * self.pso_iterations),
        #     # 'distance_from_bad_points': Box(low = 0, high = np.inf, shape = (1,), dtype=np.float32),
        #     # 'num_bad_points': Discrete(self.num_agents * self.pso_iterations),
        #     # 'iter_from_best': Discrete(self.pso_iterations),
        #     'points_in_sphere' : Discrete(self.pso_iterations * self.num_agents)
        #     # 'num_skips': Discrete(self.pso_iterations)
        # })

        obs_space = Box(low = 0, high = self.num_agents * self.pso_iterations, shape = (2,), dtype=np.float32)

        

        # obs_space = Box(
        #     low = low,
        #     high = high,
        #     shape=(len_obs, ),
        #     dtype=np.float32,
        # )

        act_space = Discrete(2)

        self.observation_space = [obs_space for i in range(self.num_agents)]
        self.action_space = [act_space for i in range(self.num_agents)]

    def _seed(self, seed=None):
        self.np_random = Randomizer.rng
        # seed = Randomizer.get_state()[1][0]
        # return [seed]

    def reset(self):
        optimizer.Randomizer.rng = np.random.default_rng(43)
        self.pso = copy.deepcopy(self.possible_pso)
        self.timestep = 0
        self.action_list = []
        self.good_points = []
        self.bad_points = []
        self.invalid_actions = [[] for _ in range(self.num_agents)]

        # Evaluate all particles to begin with
        self.pso.step()
        # mask = np.full(self.num_agents, True, dtype=bool)
        # optimization_output = self.pso.objective.evaluate(
        #     np.array([particle.position for particle in self.pso.particles]), mask)
        # [particle.set_fitness(optimization_output[p_id])
        #     for p_id, particle in enumerate(self.pso.particles)]
        
        # self.pso.update_pareto_front()
        # for particle in self.pso.particles:
        #         particle.update_velocity(self.pso.pareto_front,
        #                                     self.pso.inertia_weight,
        #                                     self.pso.cognitive_coefficient,
        #                                     self.pso.social_coefficient)
        #         particle.update_position(self.pso.lower_bounds, self.pso.upper_bounds)

        self.prev_hv = self.ind(np.array([p.fitness for p in self.pso.pareto_front]))
        # print(self.prev_hv)      
        # obs_list = self.build_state()

        # self.rewards = [0 for a in range(self.num_agents)]

        # self.terminations = {a: False for a in self.agents}
        # self.truncations = {a: False for a in self.agents}
        # # Get dummy infos. Necessary for proper parallel_to_aec conversion
        # infos = {a: {} for a in self.agents}

        # Get observation
        obs_list = self.observe_list()
        self.last_obs = obs_list
        self.last_rewards = [np.float64(0) for _ in range(self.num_agents)]
        self.last_dones = [False for _ in range(self.num_agents)]

        return obs_list[0]

    def step(self, action, agent_id, is_last):
        self.action_list.append(action)
        p = self.pso.particles[agent_id]
        
        # Execute actions
        p.num_skips = 0 if action else p.num_skips + 1
        optimization_output = self.pso.objective.evaluate(np.array([p.position]))[0] if action else [np.inf] * len(p.fitness)
        improving_evaluations = p.set_fitness(optimization_output)

        # Negative reward if evaluated
        self.last_rewards[agent_id] = self.evaluation_penalty * action
        # Negative reward
        if len(self.invalid_actions[agent_id]) > 0 and not action:
            self.last_rewards[agent_id] += -10000 

        if is_last:
            # Update pareto
            dominated = self.pso.update_pareto_front()
            # If a particle is dominated and was evaluated add it to bad points list.
            for id in range(self.num_agents):
                if dominated[id] and self.action_list[id]:
                    self.bad_points.append(self.pso.particles[id].position.copy())
            #     else:
            #         self.good_points.append(deepcopy(self.pso.particles[i].position))

            # Update velocities and positions
            for particle in self.pso.particles:
                particle.update_velocity(self.pso.pareto_front,
                                            self.pso.inertia_weight,
                                            self.pso.cognitive_coefficient,
                                            self.pso.social_coefficient)
                particle.update_position(self.pso.lower_bounds, self.pso.upper_bounds)
            
            # Assign rewards
            
            # hv = self.ind(np.array([p.fitness for p in self.pso.pareto_front]))
            # diff_hv = hv - self.prev_hv
            # self.prev_hv = hv

            for id in range(self.num_agents):
                p = self.pso.particles[id]
                # positive_reward = diff_hv

                # self.last_rewards[id] += self.metric_reward * positive_reward
                self.last_rewards[id] += self.not_dominated_reward if not dominated[id] else -5 #self.dominated_penalty
                self.last_rewards[id] += self.not_dominated_reward if not dominated[id] else 0


                # print(self.last_rewards[id])
                # self.rewards[id] = self.last_rewards[id]

            self.pso.iteration += 1
            self.action_list = []

            # Generate new observations
            obs_list = self.observe_list()
            self.last_obs = obs_list

        return self.observe(agent_id)

    def observe(self, agent_id):
        return np.array(self.last_obs[agent_id], dtype=np.float32) #self.last_obs[agent_id] 

    def observe_list(self):
        observe_list = [[] for i in range(self.num_agents)]
        # Crowding distance
        # crowding_distances = list(self.pso.calculate_crowding_distance(self.pso.particles).values())
        # print(crowding_distances)
        # crowding_distances[0] = 0.1
        # crowding_distances[-1] = 0.1

        # Normalized distance
        # lower_bounds = self.pso.lower_bounds
        # upper_bounds = self.pso.upper_bounds
        # volume = 1
        # for i in range(len(lower_bounds)): volume *= (upper_bounds[i] - lower_bounds[i])

        # volume = volume ** (1 / self.pso.num_params)

        # positions = [p.position for p in self.pso.particles]
        # best_fitnesses = [p.best_fitness for p in self.pso.particles]
        
        for i, particle in enumerate(self.pso.particles):

            # distance = np.linalg.norm(positions - positions[i], axis=1)
            # mean_distance = np.sum(distance) / (self.num_agents - 1) / self.distance_normalization
            # best_position = 
            # distance_best = np.linalg.norm(best_position - particle.position) / self.distance_normalization
            # progress = self.pso.iteration / self.pso_iterations

            # mod_velocity = np.linalg.norm(particle.velocity)
            # if mod_velocity == 0: mod_velocity = 1
            # distance_good_points, num_good_points = distance_from_cluster(particle.velocity, mod_velocity, particle.previous_position, np.array([p.position for p in self.pso.pareto_front]), self.angle, self.max_dist) if len(self.pso.pareto_front) > 0 else (1, 0)
            # distance_bad_points, num_bad_points = distance_from_cluster(particle.velocity, mod_velocity, particle.previous_position, np.array(self.bad_points), self.angle, self.max_dist) if len(self.bad_points) > 0 else (1, 0)
            # distance_bad_points = distance_from_cluster(particle.velocity, particle.position, self.bad_points) if len(self.good_points) > 0 else 1
            # if distance_good_points > 1e8:
            #         print("SHIT")
            
            # distance_good_points = distance_good_points / mod_velocity
            # distance_bad_points = distance_bad_points / mod_velocity
            # if np.linalg.norm(particle.velocity) == 0: print("UpSI")
            # distance_bad_points = distance_bad_points / self.distance_normalization

            points_in_sphere = sphere(particle.position, self.radius, np.array(self.bad_points)) if len(self.bad_points) > 0 else 0
            if points_in_sphere == 0:
                self.invalid_actions[i] = [0]
            else:
                self.invalid_actions[i] = []

            particle_observation = [
                        points_in_sphere,
                        particle.iteration_from_best_position,
                        # # mean_distance,
                        # distance_good_points,
                        # num_good_points,
                        # distance_bad_points,
                        # num_bad_points,
                        # # distance_best,
                        # particle.iteration_from_best_position,
                        # particle.num_skips,
                        # # progress
                    ]
            # particle_observation = {
            #     # 'distance_from_good_points': (distance_good_points, ),
            #     # 'num_good_points':num_good_points,
            #     # 'distance_from_bad_points': (distance_bad_points, ),
            #     # 'num_bad_points': num_bad_points,
            #     'points_in_sphere' : points_in_sphere,
            #     # 'iter_from_best': particle.iteration_from_best_position,
            #     # 'num_skips': particle.num_skips,
            # }

            # if distance_good_points > 1e8:
            #         print("SHIT")
            # if distance_bad_points > 1e8:
            #         print("SHIT")
            
            observe_list[i] = particle_observation

        return observe_list
    
    def action_masks(self):
        return [[action not in self.invalid_actions[i] for i, action in enumerate(self.possible_actions)]]
    
    def render(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.scatter([p.position[0] for p in self.pso.particles],[p.position[1] for p in self.pso.particles], color = 'black', marker = '.', s = 100)
        plt.scatter([p[0] for p in self.bad_points], [p[1] for p in self.bad_points], color = 'blue', marker = 'x', s = 50)
        plt.scatter([p.position[0] for p in self.pso.pareto_front], [p.position[1] for p in self.pso.pareto_front], color = 'green', marker = '*', s = 50)
        rect = patches.Rectangle((-2, -10), 4, 20, linewidth=1, edgecolor=None, facecolor='green', alpha=0.2)
        ax.add_patch(rect)
        for p in self.pso.particles:
            c = plt.Circle(p.position, self.radius, color = 'black', fill = False)
            # plt.plot([p.position[0], new_position[0]], [p.position[1], new_position[1]], '--')
            ax.add_patch(c)
        plt.xlim(-10,10)
        plt.ylim(-10,10)
        plt.show()
        plt.close()
        


@njit
def distance_from_cluster(v, mod_v, pos, points, angle_deg, max_dist):
    angle_rad = angle_deg * np.pi / 180
    mask = np.full(len(points), False)
    num_points_inside = 0
    num_points = len(points)
    for i in prange(num_points):
        u = points[i] - pos
        mod_u = np.linalg.norm(u)
        if mod_u != 0: # if the point is in the pareto is also inside the cone
            angle = math.acos(round(np.dot(v, u) / (mod_v * mod_u), 2))
        else:
            angle = 0
        if angle < angle_rad:
            mask[i] = True
            num_points_inside = num_points_inside + 1
    
    if  num_points_inside > 1:
        # mean_position = [points[i] for i in prange(num_points) if mask[i]][np.random.randint(num_points_inside)]
        mean_position = np.empty(len(pos))
        for j in prange(len(points[0])):
            for i in prange(num_points):
                mean_position[j] = mean_position[j] + points[i][j]
            mean_position[j] = mean_position[j] / num_points                     
        return np.linalg.norm(pos - mean_position), num_points_inside
    
    elif num_points_inside == 1:
        for i in prange(num_points):
            if mask[i]: 
                return np.linalg.norm(pos - points[i]), 1
            
    else:
        return max_dist, 0
    
@njit
def sphere(position, radius, points):
    mask = np.full(len(points), False)
    for i, p in enumerate(points): 
        if np.linalg.norm(position - p) < radius:
            mask[i] = True
    return np.sum(mask)



    
               