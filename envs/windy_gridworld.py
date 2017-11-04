# Windy Gridworld: 
# https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/windy_gridworld.py

import numpy as np
from gym.envs.toy_text import discrete
from gym import spaces
# import pdb

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class WindyGridworldEnv(discrete.DiscreteEnv):

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta, winds):
        new_position = np.array(current) + np.array(delta) + np.array([-1, 0]) * winds[tuple(current)]
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        is_done = tuple(new_position) == (3, 7)
        return [(1.0, new_state, -1.0, is_done)]

    def __init__(self):
        self.shape = (7, 10)

        nS = np.prod(self.shape)
        nA = 4

        # Wind strength
        winds = np.zeros(self.shape)
        winds[:,[3,4,5,8]] = 1
        winds[:,[6,7]] = 2

        # Calculate transition probabilities
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = { a : [] for a in range(nA) }
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0], winds)
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1], winds)
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0], winds)
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1], winds)

        # We always start in state (3, 0)
        isd = np.zeros(nS)
        isd[np.ravel_multi_index((3,0), self.shape)] = 1.0

        super(WindyGridworldEnv, self).__init__(nS, nA, P, isd)

# stochastic rewards, either -12 or 10
class StochasticWindyGridworldEnv(discrete.DiscreteEnv):
    
    def __init__(self, stochasticity = 0.1):
        self.shape = (7, 10)
        self.stochasticity = stochasticity
    
        # Wind strength
        self.winds = np.zeros(self.shape)
        self.winds[:, [3,4,5,8]] = 1
        self.winds[:, [6,7]] = 2
        
        self.nS = 70
        self.nA = 4

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)
        
    def reset(self):
        self.done = False
        self.s = [3, 0]
        return 30
    
    def step(self, action):
        if np.random.random(1) < self.stochasticity:
            s_n = self._get_neighbors()
        else:
            if action == 0:
                s_n = self._calculate_transition_prob(self.s, [-1, 0], self.winds)
            elif action == 1:
                s_n = self._calculate_transition_prob(self.s, [0, 1], self.winds)
            elif action == 2:
                s_n = self._calculate_transition_prob(self.s, [1, 0], self.winds)
            elif action == 3:
                s_n = self._calculate_transition_prob(self.s, [0, -1], self.winds)
        self.reward = -1 # np.random.choice((-12.0, 10.0), 1)
        if (tuple(s_n) == (3, 7)):
            self.done = True
        self.s = s_n
        s_n = np.ravel_multi_index(tuple(s_n), self.shape)
        return s_n, self.reward, self.done, None
    
    def _get_neighbors(self):
        up = np.array(self.s) - np.array([1, 0])
        down = np.array(self.s) + np.array([1, 0])
        left = np.array(self.s) - np.array([0, 1])
        right = np.array(self.s) + np.array([0, 1])
        diag1 = np.array(self.s) + np.array([1, - 1])
        diag2 = np.array(self.s) + np.array([- 1, 1])
        diag3 = np.array(self.s) + np.array([1, 1])
        diag4 = np.array(self.s) - np.array([1, 1])
        neighbors = np.array([up, down, left, right, diag1, diag2, diag3, diag4])
        rand_int = np.random.randint(low = 0, high = 8, size = 1)
        new_position = neighbors[rand_int]
        new_position = self._limit_coordinates(new_position[0]).astype(int)
        return new_position
        
    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta, winds):
        new_position = np.array(current) + np.array(delta) + np.array([-1, 0]) * winds[tuple(current)]
        new_position = self._limit_coordinates(new_position).astype(int)
        # new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        return new_position
