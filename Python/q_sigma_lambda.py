"""
Implementation of the Q (sigma, lambda) algorithm
- Value-based reinforcement learning control algorithm.
- Subsumes Sarsa, Expected Sarsa, Q-Learning.
- Multi-step algorithm using eligibility traces.
"""

import numpy as np
import tilecoding

def getPolicy(Q, epsilon):
    """
    Get probabilities of epsilon-greedy policy with respect to Q. 
    
    Parameters
    ----------
    Q: numpy array (float vector)
        Action value function
    epsilon: float(1) in [0, 1]
        Choice to sample a random action.
    
    Return
    ------
    A policy of same length as Q specifying the action probabilities.
    """
    greedy_action = np.argmax(Q)
    policy = np.repeat(epsilon / len(Q), len(Q))
    policy[greedy_action] += 1 - epsilon
    return policy
    
def sampleAction(policy):
    """
    Sample action from policy.
    
    Parameters
    ----------
    policy: numpy array
        The policy, a valid probability distribution.
        
    Return
    ------
    An action, a single integer value.
    """
    return np.random.choice(np.arange(0, len(policy)), p = policy)

"""
Tabular Q(sigma, lambda)
"""
def qSigmaLambda(env, n_episodes = 100, Lambda = 0, sigma = 1, beta = 0, 
                 epsilon = 0.1, alpha = 0.1, gamma = 1, 
                 target_policy = "greedy", printing = False):  
    """
    Value-based reinforcement learning control algorithm.
    Subsumes Sarsa, Expected Sarsa, Q-Learning.
    Multi-step algorithm using eligibility traces.
    
    Parameters
    ----------
    env: gym environment
        Gym environment specified using gym.make().
    n_episodes: int
        Number of episodes.
    Lambda: float(1) in [0, 1]
        Multi-step bootsrapping.
    sigma: float(1) in [0, 1]
        Weighting between Sarsa and Tree-Backup targets.
    beta: float(1) in [0, 1]
        Weighting between accumulating and replacing eligibiliy traces. 
        Use beta = 0 for accumulate traces, beta = 1 for replace traces.
    epsilon: float(1) in [0, 1]
        Choice of random action in epsilon-greedy behavior policy.
    alpha: float(1)
        Learning rate.
    gamma: float(1)
        Discount factor.
    target_policy: character(1)
        Use "greedy" for a Q-Learning target (greedy target policy) or 
        "epsilon-greedy" for on-policy learning (target policy = behavior policy).
    printing: boolean(1)
        Print out number of steps per episode?
        
    Return
    ------
    Q: Numpy array
        Action value function.
    episode_steps: Numpy array
        Steps per episode.
    episode_rewards: Numpy array
        Rewards per episode.  
    """

    if target_policy == "greedy":
        epsilon_target = 0

    Q = np.zeros((env.observation_space.n, env.action_space.n))
    episode_steps = np.zeros(n_episodes)
    rewards = np.zeros(n_episodes)
        
    for i in range(n_episodes):
        done = False
        j = 0
        reward_sum = 0
        # at begin of episode: initialize eligibility to 0
        E = np.zeros_like(Q)
        # get initial state
        s = env.reset()
        # get initial action from behavior policy
        policy = getPolicy(Q[s, ], epsilon)
        a = sampleAction(policy)
        
        while done == False:
            j += 1
            # take action, observe next state and reward
            s_n, r, done, _ = env.step(a)
            reward_sum += r
            # sample next action according to new state
            policy = getPolicy(Q[s_n, ], epsilon)
            a_n = sampleAction(policy)
            
            if target_policy == "greedy":
                policy = getPolicy(Q[s_n, ], epsilon_target)
            
            # compute td target and td error
            sarsa_target = Q[s_n, a_n]
            exp_sarsa_target = np.dot(policy, Q[s_n, ])
            td_target = r + gamma * (sigma * sarsa_target + 
                                     (1 - sigma) * exp_sarsa_target)
            td_error = td_target - Q[s, a]
            
            # update eligibility for visited state, action pair
            E[s, a] = E[s, a] * (1 - beta) + 1
            
            # update Q for all states based on their eligibility
            Q += alpha * E * td_error
            E = gamma * Lambda * E * (sigma + policy[a_n] * (1 - sigma))
            
            # set s to s_n, a to a_n
            s = s_n
            a = a_n
            
            if done:
                if printing:
                    print("Episode " + str(i + 1) + " finished after " + 
                          str(j + 1) + " time steps " + "obtaining " + 
                          str(reward_sum) + " rewards.")                    
                episode_steps[i] = j
                rewards[i] = reward_sum
                break
    return Q, episode_steps, rewards

"""
Q(sigma, lambda) implementation for Mountain Car
- Linear function approximation using grid tiling
"""

# possible actions
ACTIONS = np.arange(3)

# bounds for position and velocity
POSITION_MIN = - 1.2
POSITION_MAX = 0.5
VELOCITY_MIN = - 0.07
VELOCITY_MAX = 0.07

def getValue(state, weights, hash_table, n_tilings):
    Q = np.zeros(3)
    # for each action
    # get tile indices and compute Q value as sum of all active tiles' weights
    for i in ACTIONS:
        active_tiles = getActiveTiles(state[0], state[1], i, hash_table, n_tilings)
        Q[i] = np.sum(weights[active_tiles])
    return Q

def preprocessState(state, n_tilings):
    position = state[0]
    velocity = state[1]
    # scale state (position, velocity)
    position_scale = n_tilings / (POSITION_MAX - POSITION_MIN)
    velocity_scale = n_tilings / (VELOCITY_MAX - VELOCITY_MIN)
    position = position_scale * position # - POSITION_MIN)
    velocity = velocity_scale * velocity #- VELOCITY_MIN)
    return np.array((position, velocity))

def getActiveTiles(position, velocity, action, hash_table, n_tilings):
    active_tiles = tilecoding.tiles(hash_table, n_tilings, 
                                    [position, velocity], [action])
    return active_tiles

# get number of steps to reach the goal under current state value function
def costToGo(state):
    costs = []
    for action in ACTIONS:
        costs.append(getValue(state, action))
    return - np.max(costs)

def qSigmaLambdaMC(env,  n_episodes = 100, Lambda = 0, sigma = 1, 
                   beta = 0, epsilon = 0.1, alpha = 0.1, gamma = 1, 
                   target_policy = "greedy", printing = False, 
                   cliff = False, n_tilings = 8, max_size = 4096): 
    
    # adjust learning rate to number of tilings
    alpha = alpha / n_tilings
    hash_table = tilecoding.IHT(max_size)
    
    if target_policy == "greedy":
        epsilon_target = 0
    weights = np.zeros(max_size)
    episode_steps = np.zeros(n_episodes)
    rewards = np.zeros(n_episodes)
        
    for i in range(n_episodes):
        done = False
        j = 0
        reward_sum = 0
        # at begin of episode: initialize eligibility for each weight to 0
        E = np.zeros_like(weights)
        # get initial state and scale this state
        s = env.reset()
        s = preprocessState(s, n_tilings)

        # get action values
        Q = getValue(s, weights, hash_table, n_tilings)
        # get action probabilities (epsilon-greedy behavior policy)
        policy = getPolicy(Q, epsilon)
        # sample action from policy
        a = sampleAction(policy)
        
        while done == False:
            j += 1
            # take action, observe next state and reward
            s_n, r, done, _ = env.step(a)
            
            # only for Mountain Cliff, negative reward of -100 when falling of the cliff
            if cliff:
                if s_n <= POSITION_MIN:
                    s_n = env.reset()
                    r = - 100
                    
            reward_sum += r
            
            # sample next action according to new state
            s_n = preprocessState(s_n, n_tilings)
            Q_n = getValue(s_n, weights, hash_table, n_tilings)
            policy = getPolicy(Q_n, epsilon)
            a_n = sampleAction(policy)
            
            if target_policy == "greedy":
                policy = getPolicy(Q_n, epsilon_target)
            
            # compute td target and td error
            sarsa_target = Q_n[a_n]
            exp_sarsa_target = np.dot(policy, Q_n)
            td_target = r + gamma * (sigma * sarsa_target + 
                                     (1 - sigma) * exp_sarsa_target)
            td_error = td_target - Q[a]
            
            # get active tiles
            active_tiles = getActiveTiles(s[0], s[1], a, hash_table, n_tilings)
            # update eligibility for all active tiles
            E[active_tiles] = E[active_tiles] * (1 - beta) + 1
            
            # update weights
            weights += alpha * E * td_error
            
            # reduce eligibility for all weights
            E = gamma * Lambda * E * (sigma + policy[a_n] * (1 - sigma))
            
            # set s to s_n, a to a_n, Q to Q_n
            s = s_n
            a = a_n
            Q = Q_n
            
            if done:
                if printing:
                    print("Episode " + str(i + 1) + " finished after " + 
                          str(j + 1) + " time steps " + "obtaining " + 
                          str(reward_sum) + " rewards.")
                episode_steps[i] = j
                rewards[i] = reward_sum
                break
    return weights, episode_steps, rewards

def running_mean(x, n):
    """
    Compute running mean.
    
    Parameters
    ----------
    x: numpy array or list
        The values to compute the mean of.
    n: int
        Window size of running mean.
    
    Return
    ------
    The running mean of x, a numpy array.
    """
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[n:] - cumsum[:-n]) / n 
