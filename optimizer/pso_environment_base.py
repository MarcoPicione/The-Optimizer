import numpy as np
from gymnasium.spaces import Discrete, Box
import copy
from .metrics import hyper_volume
from optimizer import Randomizer
import time
from pymoo.indicators.hv import HV
import math
from copy import deepcopy

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
        self.theta = 20
        self.ind = HV(ref_point=self.ref_point)
        upper_bounds = np.array(self.possible_pso.upper_bounds)
        lower_bounds = np.array(self.possible_pso.lower_bounds)
        self.distance_normalization = np.linalg.norm(upper_bounds - lower_bounds)

    def get_spaces(self):
        """Define the action and observation spaces for all of the agents."""

        low = np.array([0.] * 4)
        high = np.array([np.inf] * 4)

        obs_space = Box(
            low = low,
            high = high,
            shape=(4, ),
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
        mask = np.full(self.num_agents, True, dtype=bool)
        optimization_output = self.pso.objective.evaluate(
            np.array([particle.position for particle in self.pso.particles]), mask)
        [particle.set_fitness(optimization_output[p_id])
            for p_id, particle in enumerate(self.pso.particles)]
        
        self.pso.update_pareto_front()
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

        self.last_rewards = [np.float64(0) for _ in range(self.num_agents)]
        self.last_dones = [False for _ in range(self.num_agents)]
        self.last_obs = obs_list

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
                    self.bad_points.append(deepcopy(self.pso.particles[i].position))
                else:
                    self.good_points.append(deepcopy(self.pso.particles[i].position))

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
                negative_reward = sum(self.action_list) / self.num_agents

                # print("Rewards: ", positive_reward, " ", negative_reward)
                self.last_rewards[id] = self.metric_reward * positive_reward + self.evaluation_penalty * negative_reward
                self.last_rewards[id] += self.not_dominated_reward if not dominated[i] else 0
                self.rewards[id] = self.last_rewards[id]

            self.pso.iteration += 1
            self.action_list = []

            # rewards = np.array(self.rewards)

        return self.observe(agent_id)

    def observe(self, agent_id):
        return np.array(self.last_obs[agent_id], dtype=np.float32)

    def observe_list(self):
        observe_list = []

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

            distance_good_points = self.distance_from_cluster(particle, self.good_points)
            distance_bad_points = self.distance_from_cluster(particle, self.bad_points)


            particle_observation = [
                        # mean_distance,
                        distance_good_points,
                        distance_bad_points,
                        # distance_best,
                        particle.iteration_from_best_position,
                        particle.num_skips,
                        # progress
                    ]
            observe_list.append(particle_observation)

        return observe_list

    def distance_from_cluster(self, particle, points):
        v = particle.velocity
        position = np.array(particle.position)
        saved_points = []
        for point in points:
            u = np.array(point) - position
            # print(np.dot(v,u) / (np.linalg.norm(v) * np.linalg.norm(u)))
            # print(v)
            # print(u)
            # print(np.dot(v,u))

            angle_rad = math.acos(round(np.dot(v,u) / (np.linalg.norm(v) * np.linalg.norm(u)), 2))
            angle_deg = angle_rad * 180 / np.pi
            if angle_deg < self.theta:
                saved_points.append(np.array(point))
        
        if len(saved_points) > 1:
            mean_position = np.mean(saved_points, axis = 0)
            return np.linalg.norm(position - mean_position)
        elif len(saved_points) == 1:
            return np.linalg.norm(position - saved_points[0])
        else:
            return 1
        