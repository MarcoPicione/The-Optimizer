import functools
from random import seed
from copy import copy
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .metrics import hyper_volume


class pso_environment(ParallelEnv):
    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "name": "pso_environment",
    }

    def __init__(self, pso, num_iterations, render_mode = None):
        self.num_iterations = num_iterations
        self.possible_pso = pso
        self.timestep = None
        # Space for floats
        box_space = spaces.Box(low = -np.inf, high = np.inf, shape=(2,))
        # Space for discrete
        # discrete_space = spaces.Discrete(self.possible_pso.objective.num_objectives)
        # Observation and action spaces
        self.possible_agents = ["particle_" + str(i) for i in range(pso.num_particles)]
        self.observation_spaces = {a: spaces.Dict({
                    "observation": box_space,
                    # "action_mask": None,
                })
                for a in self.possible_agents}
        self.action_spaces = {a:  spaces.Discrete(2) for a in self.possible_agents}
        self.render_mode = render_mode

    def build_state(self):
        crowding_distances = list(self.pso.calculate_crowding_distance(self.pso.particles).values())
        crowding_distances[0] = 0.1
        crowding_distances[-1] = 0.1
        print("Crowding: ", crowding_distances)
        num_skips = [p.num_skips for p in self.pso.particles]
        return {a: (crowding_distances[id], num_skips[id]) for id, a in enumerate(self.agents)}

    def reset(self, seed=None, options=None):
        self.pso = copy(self.possible_pso)
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        # Evaluate all particles to begin with
        mask = np.full(self.num_agents, True, dtype=bool)
        optimization_output = self.pso.objective.evaluate(
            np.array([particle.position for particle in self.pso.particles]), mask)
        [particle.set_fitness(optimization_output[p_id])
            for p_id, particle in enumerate(self.pso.particles)]
        
        print("Best fitnesses post reset:")
        for p in self.pso.particles: print(p.best_fitness)
        
        observations = self.build_state()

        self.rewards = {a: 0 for a in self.agents}

        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):
        step_reward = 100
        best_local_reward = 1
        evaluation_penalty = -1
        self.ref_point = [1,1]
        print("STEP")
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}
        
        # Execute actions
        mask = np.array(list(actions.values()), dtype=bool)
        print("Mask ", mask)
        for id, m in enumerate(mask):
            self.pso.particles[id].num_skips = 0 if m else self.pso.particles[id].num_skips + 1
        improving_evaluations = np.array(self.pso.step(mask = mask))

        # Check termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        for a in self.agents:
            rewards[a] += step_reward * hyper_volume([p.fitness for p in self.pso.pareto_front], self.ref_point) #Is the shape right? Weight to reward
            rewards[a] += evaluation_penalty if actions[a] else 0

        # When all the optimizer iterations are completed, the episode terminates
        if self.pso.iteration == self.num_iterations:
            terminations = {a: True for a in self.agents}
            self.agents = []

        # If a particle founds a best local fitness gets some reward
        # for id, a in enumerate(self.agents):
        #     if improving_evaluations[id]: rewards[a] += best_local_reward

        # If a particle founds a best fitness gets some reward
        # ????

        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in self.agents}
        # if self.timestep > 100:
        #     rewards = {"prisoner": 0, "guard": 0}
        #     truncations = {"prisoner": True, "guard": True}
        #     self.agents = []
        # self.timestep += 1

        # Get observations
        observations = self.build_state()

        # Get dummy infos (not used in this example)
        infos = {a: {} for a in self.agents}

        self.rewards = copy(rewards)
        self.terminations = copy(terminations)
        self.truncations = copy(truncations)

        print("Observations: ", observations)
        return observations, rewards, terminations, truncations, infos

    def render(self):
        pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return self.observation_spaces[agent]
        # return Box(low = -np.inf, high = np.inf, shape=(2,))

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]
        # return Discrete(1)
    
    def observe(self, agent):
        return np.array(self.observations[agent])
    
    def seed(seed=None):
        seed(seed)

    def close(self):
        pass

    def state(self):
        pass

    def state_space(self):
        pass