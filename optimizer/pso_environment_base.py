import numpy as np
from gymnasium.spaces import Discrete, Box
import copy
from .metrics import hyper_volume, stupid_hv
from optimizer import Randomizer
import time

class pso_environment_base:
    def __init__(self, pso, num_iterations, metric_reward, evaluation_penalty, render_mode = None):
        
        self.possible_pso = pso
        self.num_iterations = num_iterations
        self.num_agents = self.possible_pso.num_particles
        self.metric_reward = metric_reward
        self.evaluation_penalty = evaluation_penalty
        self.ref_point = [5,5]


        self.last_dones = [False for _ in range(self.num_agents)]
        self.last_obs = [None for _ in range(self.num_agents)]
        self.last_rewards = [np.float64(0) for _ in range(self.num_agents)]

        self.render_mode = render_mode
        self.get_spaces()
        self._seed()

    def get_spaces(self):
        """Define the action and observation spaces for all of the agents."""

        low = np.array([-np.inf, 0., 0.])
        high = np.array([np.inf, np.inf, np.inf])

        obs_space = Box(
            low = low,
            high = high,
            shape=(3, ),
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

        # Evaluate all particles to begin with
        mask = np.full(self.num_agents, True, dtype=bool)
        optimization_output = self.pso.objective.evaluate(
            np.array([particle.position for particle in self.pso.particles]), mask)
        [particle.set_fitness(optimization_output[p_id])
            for p_id, particle in enumerate(self.pso.particles)]
              
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
        optimization_output = self.pso.objective.evaluate(np.array([p.position]))[0] if action else p.best_fitness
        improving_evaluations = p.set_fitness(optimization_output)

        if is_last:
            # Update pareto, velocities and positions
            self.pso.update_pareto_front()
            for particle in self.pso.particles:
                particle.update_velocity(self.pso.pareto_front,
                                            self.pso.inertia_weight,
                                            self.pso.cognitive_coefficient,
                                            self.pso.social_coefficient)
                particle.update_position(self.pso.lower_bounds, self.pso.upper_bounds)
            
            # Other stuff
            obs_list = self.observe_list()
            self.last_obs = obs_list

            # if self.pso.iteration == self.num_iterations:
            # print("Mopso iteration ", self.pso.iteration)
            # print("Pareto dim ", len(self.pso.pareto_front))
            
            # start = time.time()
            hv = hyper_volume([p.fitness for p in self.pso.pareto_front], self.ref_point)
            # print(hv)
            # end = time.time()
            # print(end - start)
            for id in range(self.num_agents):
                p = self.pso.particles[id]
                # print(self.metric_reward * hv)
                # print(self.evaluation_penalty * sum(self.action_list))
                self.last_rewards[id] = self.metric_reward * hv + self.evaluation_penalty * sum(self.action_list) / self.num_agents #Is the shape right? Weight to reward
                self.rewards[id] += self.last_rewards[id]

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
        lower_bounds = self.pso.lower_bounds
        upper_bounds = self.pso.upper_bounds
        volume = 1
        for i in range(len(lower_bounds)): volume *= (upper_bounds[i] - lower_bounds[i])

        positions = [p.position for p in self.pso.particles]
        
        for i, particle in enumerate(self.pso.particles):

            distance = np.linalg.norm(positions - positions[i], axis=1)
            mean_distance = np.sum(distance) / (self.num_agents - 1) / volume
    
            distance_best = np.linalg.norm(particle.best_position - particle.position) / volume
            particle_observation = [
                        mean_distance,
                        distance_best,
                        particle.num_skips
                    ]
            observe_list.append(particle_observation)

        return observe_list