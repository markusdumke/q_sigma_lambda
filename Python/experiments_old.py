"""
Experiments using Q(sigma, lambda) algorithm
"""

import gym
import numpy as np
import pandas as pd
import os

os.chdir("C:/Users/M/Desktop/q_sigma_lambda/Python")

from q_sigma_lambda import qSigmaLambda, qSigmaLambdaMC
from envs.windy_gridworld import WindyGridworldEnv
from envs.cliff_walking import CliffWalkingEnv

if __name__ == "__main__":    
    # set seed for reproducibility
    np.random.seed(13092017)
    # Experiments Mountain Car
    env = gym.make("MountainCar-v0").env
    
    # Test values: Parameters
    n_episodes = 500
    alphas = [0.1] # [0.1, 0.5]
    epsilon = 0.1
    n_runs = 10
    beta = 0 # [0, 1]
    lambdas = [0, 0.5, 0.9]
    sigmas = [0, 0.5, 1]
    
    param_comb = len(sigmas) * len(lambdas) * len(alphas)
    rows = n_episodes * n_runs * param_comb
    cols = 9
    df = np.zeros((rows, cols))
    
    for Lambda in range(len(lambdas)):
        print("Lambda:" + str(lambdas[Lambda]))
        for sigma in range(len(sigmas)):
            print("Sigma:" + str(sigmas[sigma]))
            for alpha in range(len(alphas)):
                print("Alpha:" + str(alphas[alpha]))
                param_comb = Lambda * len(sigmas) * len(alphas) + sigma * len(alphas) + alpha
                for run in range(n_runs):
                    _, steps, rewards = qSigmaLambdaMC(env, Lambda = lambdas[Lambda], 
                                                       sigma = sigmas[sigma], 
                                                       n_episodes = n_episodes, 
                                                       alpha = alphas[alpha], 
                                                       epsilon = epsilon, 
                                                       beta = beta)
                    # get index of which row to fill
                    index = Lambda * len(sigmas) * len(alphas) * n_runs + sigma * len(alphas) * n_runs + alpha * n_runs + run
                    # print(index)
                    start = index * n_episodes
                    end = index * n_episodes + n_episodes
                    df[start:end, 0] = param_comb
                    df[start:end, 1] = alphas[alpha]
                    df[start:end, 2] = beta
                    df[start:end, 3] = lambdas[Lambda]
                    df[start:end, 4] = sigmas[sigma]
                    df[start:end, 5] = run
                    df[start:end, 6] = np.arange(n_episodes)
                    df[start:end, 7] = steps
                    df[start:end, 8] = rewards
    
    df = pd.DataFrame(df)
    df.columns = ['param_comb', 'alpha', "beta", "Lambda", "sigma", "run", 
                  "episode", "steps", "reward"]
    df.head(10)
    df.to_csv("mountaincar.csv")
#==============================================================================    
    # dynamic sigma: reduce sigma over time (maybe reduce lambda over time?)
    _, steps, rewards = qSigmaLambdaMC(env, Lambda = 0.9, sigma = 0, 
                                       n_episodes = 300, alpha = 0.1, 
                                       epsilon = 0.1, beta = 0, 
                                       update_sigma = 1, cliff = False)
    
    _, steps2, rewards2 = qSigmaLambdaMC(env, Lambda = 0.9, sigma = 0, 
                                         n_episodes = 300, alpha = 0.1, 
                                         epsilon = 0.1, beta = 0.2, 
                                         update_sigma = 1, cliff = False)
    
    _, steps3, rewards3 = qSigmaLambdaMC(env, Lambda = 0.9, sigma = 0, 
                                         n_episodes = 300, alpha = 0.1, 
                                         epsilon = 0.1, beta = 0.5, 
                                         update_sigma = 1, cliff = False)
    
    _, steps4, rewards4 = qSigmaLambdaMC(env, Lambda = 0.9, sigma = 0, 
                                         n_episodes = 300, alpha = 0.1, 
                                         epsilon = 0.1, beta = 1, 
                                         update_sigma = 1, cliff = False)
    
    window = 25
    steps_avg = running_mean(steps, window) 
    x = np.arange(len(steps_avg))
    plt.plot(x, steps_avg, "r", label = "0")
    steps_avg2 = running_mean(steps2, window)
    plt.plot(x, steps_avg2, "b", label = "0.2")
    steps_avg3 = running_mean(steps3, window)
    plt.plot(x, steps_avg3, "g", label = "0.5")
    steps_avg4 = running_mean(steps4, window)
    plt.plot(x, steps_avg4, "y", label = "1")
    plt.ylim(100, 300)
    plt.xlim(0, 200)
    plt.legend()
    #------------------------------
    # Test on Windy Gridworld domain
#==============================================================================
#     env = WindyGridworldEnv()
#     Q, episode_steps = qSigmaLambda(env, Lambda = 0.5)
#     print(episode_steps)
#==============================================================================
    
    # Tests on Cliff Walking
    env = CliffWalkingEnv()
    
    # Test values: Parameters
    n_episodes = 300
    alphas = [0.5] # [0.1, 0.5]
    epsilon = 0.1
    n_runs = 1
    beta = 0 # [0, 1]
    lambdas = [0]#, 0.5, 0.9]
    sigmas = [0, 0.25, 0.5, 0.75, 1]
    
    param_comb = len(sigmas) * len(lambdas) * len(alphas)
    rows = n_episodes * n_runs * param_comb
    cols = 9
    df = np.zeros((rows, cols))
    
    for Lambda in range(len(lambdas)):
        print("Lambda:" + str(lambdas[Lambda]))
        for sigma in range(len(sigmas)):
            print("Sigma:" + str(sigmas[sigma]))
            for alpha in range(len(alphas)):
                print("Alpha:" + str(alphas[alpha]))
                param_comb = Lambda * len(sigmas) * len(alphas) + sigma * len(alphas) + alpha
                for run in range(n_runs):
                    _, steps, rewards = qSigmaLambda(env, Lambda = lambdas[Lambda], 
                                                       sigma = sigmas[sigma], 
                                                       n_episodes = n_episodes, 
                                                       alpha = alphas[alpha], 
                                                       epsilon = epsilon, 
                                                       beta = beta)
                    # get index of which row to fill
                    index = Lambda * len(sigmas) * len(alphas) * n_runs + sigma * len(alphas) * n_runs + alpha * n_runs + run
                    # print(index)
                    start = index * n_episodes
                    end = index * n_episodes + n_episodes
                    df[start:end, 0] = param_comb
                    df[start:end, 1] = alphas[alpha]
                    df[start:end, 2] = beta
                    df[start:end, 3] = lambdas[Lambda]
                    df[start:end, 4] = sigmas[sigma]
                    df[start:end, 5] = run
                    df[start:end, 6] = np.arange(n_episodes)
                    df[start:end, 7] = steps
                    df[start:end, 8] = rewards
    
    df = pd.DataFrame(df)
    df.columns = ['param_comb', 'alpha', "beta", "Lambda", "sigma", "run", 
                  "episode", "steps", "reward"]
    df.head(10)
    df.to_csv("cliffwalking.csv")
    
    
#==============================================================================
    """ Windy Gridworld """
    
    env = WindyGridworldEnv()
    
    # Hyperparameter values
    n_episodes = 100
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    epsilon = 0.1
    n_runs = 1
    beta = 0 # [0, 1]
    lambdas = [0, 0.2, 0.5]
    sigmas = [0]
    
    param_comb_n = len(sigmas) * len(lambdas) * len(alphas)
    rows = n_episodes * n_runs * param_comb_n
    cols = 9
    df = np.zeros((rows, cols))
    
    for Lambda in range(len(lambdas)):
        for sigma in range(len(sigmas)):
            for alpha in range(len(alphas)):
                param_comb = Lambda * len(sigmas) * len(alphas) + sigma * len(alphas) + alpha
                for run in range(n_runs):
                    _, steps, rewards = qSigmaLambda(env, Lambda = lambdas[Lambda], 
                                                       sigma = sigmas[sigma], 
                                                       n_episodes = n_episodes, 
                                                       alpha = alphas[alpha], 
                                                       epsilon = epsilon, 
                                                       beta = beta)
                    # get index of which row to fill
                    index = Lambda * len(sigmas) * len(alphas) * n_runs + sigma * len(alphas) * n_runs + alpha * n_runs + run
                    # print(index)
                    start = index * n_episodes
                    end = index * n_episodes + n_episodes
                    df[start:end, 0] = param_comb
                    df[start:end, 1] = alphas[alpha]
                    df[start:end, 2] = beta
                    df[start:end, 3] = lambdas[Lambda]
                    df[start:end, 4] = sigmas[sigma]
                    df[start:end, 5] = run
                    df[start:end, 6] = np.arange(n_episodes)
                    df[start:end, 7] = steps
                    df[start:end, 8] = rewards
                # Print out how many percent of the simulation are done
                print(str(np.round((param_comb + 1) / param_comb_n * 100, 2)) + "%")
    
    df = pd.DataFrame(df)
    df.columns = ['param_comb', 'alpha', "beta", "Lambda", "sigma", "run", 
                  "episode", "steps", "reward"]
    df.head(10)
    df.to_csv("windygridworld.csv")
    
    
#==============================================================================
#    #-------------------
#    _, steps, rewards = qSigmaLambdaMC(env, Lambda = 0.9, sigma = 1, 
#                                        n_episodes = 50, alpha = 0.2, 
#                                        epsilon = 0.1, beta = 0, 
#                                        update_sigma = 1, cliff = True)
#    np.mean(rewards)
#==============================================================================
