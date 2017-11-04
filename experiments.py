""" Experiments Graphics """
import gym
import numpy as np
import pandas as pd
#import os
#os.chdir("C:/Users/M/Desktop/q_sigma_lambda")

from simulate import simulate, average_runs, average_episodes
from q_sigma_lambda import qSigma, qSigmaMC
from envs.windy_gridworld import StochasticWindyGridworldEnv

#==============================================================================
# set seed for reproducibility
np.random.seed(4112017)

# Windy gridworld with stochastic state transitions in 10% of the cases
env = StochasticWindyGridworldEnv()

## Simulation
##---------------
## Fixed sigma
#df1 = simulate(env, qSigma, n_episodes = 100, alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#              n_runs = 200, sigmas = [0, 0.5, 1], lambdas = [0, 0.7])
#df1.to_csv("data/gridworld_fixedsigma.csv")
## Dynamic sigma
#df2 = simulate(env, qSigma, n_episodes = 100, alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#              n_runs = 200, sigmas = [1], lambdas = [0, 0.7], update_sigma = 0.99)
#df2.to_csv("data/gridworld_dynasigma.csv")

# Load data
df1 = pd.read_csv("data/gridworld_fixedsigma.csv", index_col = 0)
df2 = pd.read_csv("data/gridworld_dynasigma.csv", index_col = 0)

# Average return over runs and episodes
dfa = average_episodes(df1)
print(dfa.sort_values("Returns"))

# Average return over runs and episodes for dynamic sigma
df2a = average_episodes(df2)
print(df2a.sort_values("Returns"))

# Plot
#-------------------
colors = ["DarkViolet", "DodgerBlue", "GoldenRod"]
linestyles = [':', "-"]
col_var = "Sigma"
ls_var = "Lambda"
col = np.unique(df1[col_var]) 
ls = np.unique(df1[ls_var])
import matplotlib.pyplot as plt
#fig, axes = plt.subplots()
fig = plt.figure()
axes = plt.subplot(111)
dummy_lines = []
for col_idx, col_val in enumerate(col):
    for ls_idx, ls_val in enumerate(ls):
        dfb = dfa[dfa[col_var] == col[col_idx]]
        dfc = dfb[dfb[ls_var] == ls[ls_idx]]
        axes.plot(dfc["Alpha"], dfc["Returns"], c = colors[col_idx], 
                  ls = linestyles[ls_idx], label = col_var + " " + str(col_val))
for ls_idx, ls_val in enumerate(ls):
    dfb = df2a[df2a[ls_var] == ls[ls_idx]]
    axes.plot(dfb["Alpha"], dfb["Returns"], c = "LimeGreen", 
              ls = linestyles[ls_idx], label = "Dynamic σ")
    dummy_lines.append(axes.plot([],[], c="black", ls = linestyles[ls_idx])[0])
lines = axes.get_lines()
box = axes.get_position()
axes.set_position([box.x0, box.y0, box.width * 0.8, box.height])
legend1 = plt.legend([lines[i] for i in [1, 3, 5, 8]], ["σ = 0", "σ = 0.5", "σ = 1", "Dynamic σ"], 
                     bbox_to_anchor=(1, 0.8), loc='center left')
legend2 = plt.legend([dummy_lines[i] for i in [0, 1]], ["λ = 0", "λ = 0.7"], loc='center left',
                     bbox_to_anchor=(1, 0.2))
axes.add_artist(legend1)
plt.title("Windy Gridworld")
plt.ylim(-100, -40)
plt.ylabel("Return")
plt.xlabel("Step size α")
plt.savefig("plots/windygridworld_alpha.pdf", format = "pdf", dpi = 150)
plt.show()

#==============================================================================
""" Mountain Cliff """

# we need gym.make().env to prevent the time step limit of 200 steps
env = gym.make("MountainCliff-v0").env

# Test different hyperparameter combinations
#df1 = simulate(env, algorithm = qSigmaMC, n_episodes = 500, n_runs = 100, 
#              epsilon = 0.1, gamma = 1, alphas = [0.1], betas = [0], 
#              lambdas = [0.8], sigmas = [0, 0.5, 1], 
#              target_policy = ["greedy"])
##df1.to_csv("data/mountaincliff_fixedsigma.csv")
#df2 = simulate(env, algorithm = qSigmaMC, n_episodes = 500, n_runs = 100, 
#              epsilon = 0.1, gamma = 1, alphas = [0.1], betas = [0], 
#              lambdas = [0.8], sigmas = [1], update_sigma = 0.99,
#              target_policy = ["greedy"]) # dynamic sigma
##df2.to_csv("data/mountaincliff_dynasigma.csv")

df1 = pd.read_csv("data/mountaincliff_fixedsigma.csv", index_col = 0)
df2 = pd.read_csv("data/mountaincliff_dynasigma.csv", index_col = 0)

# Average return over runs and episodes
dfa = average_episodes(df1)
print(dfa.sort_values("Returns"))

# Average return over runs and episodes for dynamic sigma
df2a = average_episodes(df2)
print(df2a)

# Average over runs
df1 = average_runs(df1)
df2 = average_runs(df2)

# Plot
#-------------------
window = 100
from simulate import running_mean
fig = plt.figure()
axes = plt.subplot(111)
dummy_lines = []
for col_idx, col_val in enumerate(col):
    dfb = df1[df1[col_var] == col[col_idx]]
    steps_avg = running_mean(dfb.as_matrix(["Returns"]), window)
    x = np.arange(len(steps_avg))
    axes.plot(x, steps_avg, c = colors[col_idx], 
              label = col_var + " " + str(col_val))

steps_avg = running_mean(df2.as_matrix(["Returns"]), window)
axes.plot(x, steps_avg, c = "LimeGreen", label = "Dynamic σ")
plt.ylim(-170, -125)
plt.legend(["σ = 0", "σ = 0.5", "σ = 1", "Dynamic σ"], 
           loc = "lower right")
plt.title("Mountain Cliff")
plt.ylabel("Return")
plt.xlabel("Episode")

if window > 1:
    a = axes.get_xticks().tolist()
    for idx, val in enumerate(a):
        a[idx] = str(int(val)) + ":" + str(int(val + window))
    axes.set_xticklabels(a)
    # remove every second tick label
    for label in axes.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
plt.savefig("plots/mountaincliff.pdf", format = "pdf", dpi = 150)
plt.show()
