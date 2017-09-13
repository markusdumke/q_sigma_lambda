"""
Implementation of the Q (sigma, lambda) algorithm
- Value-based reinforcement learning control algorithm.
- Subsumes Sarsa, Expected Sarsa, Q-Learning.
- Multi-step algorithm using eligibility traces.
"""

import numpy as np

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
        eligibility = np.zeros_like(Q)
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
            eligibility[s, a] += eligibility[s, a] * (1 - beta) + 1
            
            # update Q for all states based on their eligibility
            Q += alpha * eligibility * td_error
            eligibility = gamma * Lambda * eligibility * (sigma + policy[a_n] * (1 - sigma))
            
            # set s to s_n, a to a_n
            s = s_n
            a = a_n
            
            if done:
                if printing:
                    print("Episode " + str(i + 1) + " finished after " + str(j + 1) + " time steps.")
                episode_steps[i] = j
                rewards[i] = reward_sum
                break
    return Q, episode_steps, rewards


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
