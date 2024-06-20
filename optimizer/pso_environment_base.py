import numpy as np
from gymnasium.spaces import Discrete, Box
import copy
from .metrics import hyper_volume
from optimizer import Randomizer
import time
from pymoo.indicators.hv import HV
import math
from copy import deepcopy
from numba import njit, jit, prange

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
        self.distance_normalization = np.linalg.norm(upper_bounds - lower_bounds)

    def get_spaces(self):
        """Define the action and observation spaces for all of the agents."""

        len_obs = 6
        low = np.array([0.] * len_obs)
        high = np.array([np.inf] * len_obs)

        obs_space = Box(
            low = low,
            high = high,
            shape=(len_obs, ),
            dtype=np.float32,
        )

        act_space = Discrete(2)

        self.observation_space = [obs_space for i in range(self.num_agents)]
        self.action_space = [act_space for i in range(self.num_agents)]

    def _seed(self, seed=None):
        self.np_random = Randomizer.rng
        # seed = Randomizer.get_state()[1][0]
        # return [seed]

    def reset(self):
        self.pso = copy.deepcopy(self.possible_pso)
        self.timestep = 0
        self.action_list = []
        self.good_points = []
        self.bad_points = []

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

        self.rewards = [0 for a in range(self.num_agents)]

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

        if is_last:
            # Update pareto, velocities and positions
            dominated = self.pso.update_pareto_front()
            for i in range(self.num_agents):
                if dominated[i]:
                    self.bad_points.append(self.pso.particles[i].position.copy())
            #     else:
            #         self.good_points.append(deepcopy(self.pso.particles[i].position))

            for particle in self.pso.particles:
                particle.update_velocity(self.pso.pareto_front,
                                            self.pso.inertia_weight,
                                            self.pso.cognitive_coefficient,
                                            self.pso.social_coefficient)
                particle.update_position(self.pso.lower_bounds, self.pso.upper_bounds)
            
            # Other stuff
            obs_list = self.observe_list()
            self.last_obs = obs_list

            # if self.pso.iteration == self.pso_iterations:
            # print("Mopso iteration ", self.pso.iteration)
            # print("Pareto dim ", len(self.pso.pareto_front))
            
            # start = time.time()
            hv = self.ind(np.array([p.fitness for p in self.pso.pareto_front]))#hyper_volume([p.fitness for p in self.pso.pareto_front], self.ref_point)
            diff_hv = hv - self.prev_hv
            self.prev_hv = hv
            # print(hv)

            # end = time.time()
            # print(end - start)
            for id in range(self.num_agents):
                p = self.pso.particles[id]
                # print(self.metric_reward * hv)
                # print(self.evaluation_penalty * sum(self.action_list))
                positive_reward = diff_hv
                negative_reward = self.action_list[id]

                # print("Rewards: ", positive_reward, " ", negative_reward)
                # self.last_rewards[id] = self.metric_reward * positive_reward + self.evaluation_penalty * negative_reward
                self.last_rewards[id] = self.evaluation_penalty * negative_reward
                self.last_rewards[id] += self.not_dominated_reward if not dominated[id] else 0

                # print("Reward")
                # print(self.metric_reward * positive_reward)
                # print(self.evaluation_penalty * negative_reward)
                # print(self.not_dominated_reward if not dominated[id] else 0)

                self.rewards[id] = self.last_rewards[id]

            self.pso.iteration += 1
            self.action_list = []

            # rewards = np.array(self.rewards)

        return self.observe(agent_id)

    def observe(self, agent_id):
        return np.array(self.last_obs[agent_id], dtype=np.float32)

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

        positions = [p.position for p in self.pso.particles]
        # best_fitnesses = [p.best_fitness for p in self.pso.particles]
        
        for i, particle in enumerate(self.pso.particles):

            # distance = np.linalg.norm(positions - positions[i], axis=1)
            # mean_distance = np.sum(distance) / (self.num_agents - 1) / self.distance_normalization
            # best_position = 
            # distance_best = np.linalg.norm(best_position - particle.position) / self.distance_normalization
            # progress = self.pso.iteration / self.pso_iterations

            mod_velocity = np.linalg.norm(particle.velocity)
            distance_good_points, num_good_points = distance_from_cluster(particle.velocity, mod_velocity, particle.previous_position, np.array([p.position for p in self.pso.pareto_front]), self.angle) if len(self.pso.pareto_front) > 0 else (1, 0)
            distance_bad_points, num_bad_points = distance_from_cluster(particle.velocity, mod_velocity, particle.previous_position, np.array(self.bad_points), self.angle) if len(self.bad_points) > 0 else (1, 0)
            # distance_bad_points = distance_from_cluster(particle.velocity, particle.position, self.bad_points) if len(self.good_points) > 0 else 1
            distance_good_points = distance_good_points / mod_velocity
            distance_bad_points = distance_bad_points / mod_velocity
            # if np.linalg.norm(particle.velocity) == 0: print("UpSI")
            # distance_bad_points = distance_bad_points / self.distance_normalization


            particle_observation = [
                        # mean_distance,
                        distance_good_points,
                        num_good_points,
                        distance_bad_points,
                        num_bad_points,
                        # distance_best,
                        particle.iteration_from_best_position,
                        particle.num_skips,
                        # progress
                    ]
            observe_list[i] = particle_observation

        return observe_list

@njit
def distance_from_cluster(v, mod_v, pos, points, angle_deg):
    angle_rad = angle_deg * np.pi / 180
    mask = np.full(len(points), False)
    num_points_inside = 0
    num_points = len(points)
    for i in prange(num_points):
        u = points[i] - pos
        mod_u = np.linalg.norm(u)
        if mod_u == 0: continue

        angle = math.acos(np.dot(v, u) / (mod_v * mod_u))
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
        return 1, 0
               