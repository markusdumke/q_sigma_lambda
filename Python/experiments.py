"""
Experiments using Q(sigma, lambda) algorithm
"""

import gym
import numpy as np
import pandas as pd
import os

os.getcwd()
os.chdir("C:/Users/M/Desktop/q_sigma_lambda/Python")

from q_sigma_lambda import qSigmaLambda, qSigmaLambdaMC, running_mean
from envs.windy_gridworld import WindyGridworldEnv
from envs.cliff_walking import CliffWalkingEnv

if __name__ == "__main__":    

    # Experiments Mountain Car
    env = gym.make("MountainCar-v0")
    
    # Test values: Parameters
    n_episodes = 3
    alpha = 0.1 # [0.1, 0.5]
    epsilon = 0.1
    n_runs = 2
    beta = 0 # [0, 1]
    lambdas = [0, 0.5, 0.9]
    sigmas = [0, 0.5, 1]
    
    param_comb = len(sigmas) * len(lambdas)
    rows = n_episodes * n_runs * param_comb
    cols = 9
    df = np.zeros((rows, cols))
    
    for Lambda in range(len(lambdas)):
        print("Lambda:" + str(Lambda))
        for sigma in range(len(sigmas)):
            print("Sigma:" + str(sigma))
            param_comb = Lambda * len(sigmas) + sigma
            for run in range(n_runs):
                _, steps, rewards = qSigmaLambdaMC(env, Lambda = lambdas[Lambda], 
                                                   sigma = sigmas[sigma], 
                                                   n_episodes = n_episodes, 
                                                   alpha = alpha, 
                                                   epsilon = epsilon, 
                                                   beta = beta)
                index = Lambda * len(sigmas) * n_runs + sigma * n_runs + run
                # print(index)
                start = index * n_episodes
                end = index * n_episodes + n_episodes
                # print(index_range)
                df[start:end, 0] = param_comb
                df[start:end, 1] = alpha
                df[start:end, 2] = beta
                df[start:end, 3] = lambdas[Lambda]
                df[start:end, 4] = sigmas[sigma]
                df[start:end, 5] = run
                df[start:end, 6] = np.arange(n_episodes)
                # df[start:end, 7] = steps
                # df[start:end, 8] = rewards
    
    df = pd.DataFrame(df)
    df.columns = ['param_comb', 'alpha', "beta", "Lambda", "sigma", "run", 
                  "episode", "steps", "reward"]
    np.save("mountaincar.npy", df)
    
    
    
    # Test on Windy Gridworld domain
    env = WindyGridworldEnv()
    Q, episode_steps = qSigmaLambda(env, Lambda = 0.5)
    print(episode_steps)
    
    # Tests on Cliff Walking
    env = CliffWalkingEnv()
    
    lambdas = 0
    sigmas = np.arange(0, 1.1, 0.25)
    betas = 0
    n_runs = 2
    n_episodes = 300
    
    res = np.zeros([n_runs * len(sigmas), 3 + n_episodes])
    
    for run in range(n_runs):
        print(run)
        for sigma in range(len(sigmas)):
            index = (run - 1) * len(sigmas) + sigma
            res[index, 0] = index
            res[index, 1] = run
            res[index, 2] = sigmas[sigma]
            _, res[index, 3:(3 + n_episodes)] = qSigmaLambda(env, Lambda = 0.8, 
                   sigma = sigmas[sigma], n_episodes = n_episodes, alpha = 0.5)
    np.save("cliff_results_lambda0.8", res)
    
    # plots episode length over time
    
    # Test on Mountain Car
    # env = gym.make("MountainCar-v0")
    
