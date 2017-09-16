""" Experiments Graphics """
import gym
import numpy as np
import pandas as pd
import os
os.chdir("C:/Users/M/Desktop/q_sigma_lambda/Python")

from simulate import simulate, plot
from q_sigma_lambda import qSigmaLambda, qSigmaLambdaMC
from envs.windy_gridworld import WindyGridworldEnv
from envs.cliff_walking import CliffWalkingEnv

""" 
============
Mountain Car 
============
"""
# set seed for reproducibility
np.random.seed(13092017)

# we need gym.make().env to prevent the time step limit of 200 steps
env = gym.make("MountainCar-v0").env

# Test different hyperparameter combinations
df = simulate(env, algorithm = qSigmaLambdaMC, n_episodes = 250, n_runs = 1, 
              epsilon = 0.1, gamma = 1, alphas = [0.1], betas = [0, 1], 
              lambdas = [0, 0.5, 0.9], sigmas = [0, 0.5, 1])
# save results as csv
# df.to_csv("mountaincar22.csv")

# load data from experiments
df = pd.read_csv("mountaincar.csv", index_col = 0)

# average over each run
df = df.groupby(['param_comb', 'alpha', "beta", "Lambda", "sigma", "episode"], as_index = False)['steps', "reward"].mean()
df.head(10)

# Plotting
df = df.rename(columns = {"sigma": "Sigma"})
plot(df, window = 50, title = "Mountain Car", file = "mountaincar2.pdf", 
     col_var = "Lambda", ls_var = "Sigma",
     xlim = [0, 150], ylim = [100, 400])

# ------------------
# Table with mean return for 50, 500 episodes

#==============================================================================
# df2 = df.groupby(['param_comb', 'alpha', "beta", "Lambda", "sigma"], as_index = False).mean()
# df3 = df2[["Lambda", "sigma", "steps"]]
# 
# # latex table code
# df3.to_latex()
# 
#==============================================================================
env = WindyGridworldEnv()
df = simulate(env, algorithm = qSigmaLambda, n_episodes = 200, n_runs = 10, 
              epsilon = 0.1, gamma = 1, alphas = [0.5], betas = [0, 0.5, 1], 
              lambdas = [0.5], sigmas = [0.5])
df = df.groupby(['param_comb', 'alpha', "beta", "Lambda", "sigma", "episode"], as_index = False)['steps', "reward"].mean()
df = df.rename(columns = {"beta": "Beta"})
plot(df, window = 10, title = "Windy Gridworld", file = "windygridworld.pdf", 
     col_var = "Beta", ls_var = "sigma",
     xlim = [0, 20], ylim = [0, 200])

#==============================================================================
env = CliffWalkingEnv()
df = simulate(env, algorithm = qSigmaLambda, n_episodes = 300, n_runs = 100, 
              epsilon = 0.1, gamma = 1, alphas = [0.5], betas = [0, 1], 
              lambdas = [0.5], sigmas = [0, 0.5, 1])
df.to_csv("cliffwalking2.csv")
df = df.groupby(['param_comb', 'alpha', "beta", "Lambda", "sigma", "episode"], as_index = False)['steps', "reward"].mean()
plot(df, window = 50, title = "Cliff Walking", file = "cliffwalking2.pdf", 
     col_var = "sigma", ls_var = "beta",
     xlim = [0, 300], ylim = [- 80, 0], y_var = "reward")



