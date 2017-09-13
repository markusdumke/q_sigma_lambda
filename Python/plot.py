""" Plotting """

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

os.chdir("C:/Users/M/Desktop/q_sigma_lambda/Python")

from q_sigma_lambda import running_mean

# style of plots ggplot, but with white background defined in file theme_bw.mplstyle
# theme_bw = "C:/Users/M/Desktop/arxiv_paper/python_implementation/theme_bw.mplstyle"
# plt.style.use(theme_bw)
plt.style.use("ggplot_white")

""" 
Mountain Car results 
--------------------
"""
# load data from experiments
df = pd.read_csv("mountaincar.csv", index_col = 0)
# df = np.load("mountaincar.npy")

# average over each run
df = df.groupby(['param_comb', 'alpha', "beta", "Lambda", "sigma", "episode"], as_index = False)['steps', "reward"].mean()
df.head(10)

# Plotting
# one line per value of sigma, averaged over 40 episodes
plt.figure(figsize=(7.5, 5))

# moving average over 50 episodes
window = 50

for i in np.unique(df["param_comb"]):
    df2 = df[df["param_comb"] == i]
    steps_avg = running_mean(df2.as_matrix(["steps"]), window)
    x = np.arange(len(steps_avg)) + window / 2
    plt.plot(x, steps_avg, label = i)
    # plt.plot(df2["episode"], df2["steps"], label = i)
plt.title("Mountain Car", loc = "left")    
plt.ylabel("Steps per episode")
plt.xlabel("Episode")
leg = plt.legend(loc='upper right', bbox_to_anchor=(0.85, 0.9), title = "params")
# leg.get_frame().set_edgecolor('b')
# plt.ylim(0, 500)
plt.xlim(0, 100)
plt.savefig("mountaincar.pdf", format = "pdf")
