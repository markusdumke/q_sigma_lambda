from q_sigma_lambda import qSigmaLambda, qSigmaLambdaMC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Plot style: ggplot, but with white background defined in file theme_bw.mplstyle
theme_bw = "C:/Users/M/Desktop/q_sigma_lambda/Python/ggplot_white.mplstyle"
plt.style.use(theme_bw)
# plt.style.use("ggplot_white")

#==============================================================================
# Simulate data for different hyperparameter combinations and return results
#==============================================================================
def simulate(env, algorithm = qSigmaLambda, n_episodes = 100, n_runs = 1, epsilon = 0.1, gamma = 1, 
             alphas = [0.1], betas = [0], lambdas = [0], sigmas = [1]):

    # number of different parameter combinations
    param_comb_n = len(sigmas) * len(lambdas) * len(alphas) * len(betas)
    
    # number of runs
    rows = n_episodes * n_runs * param_comb_n
    cols = 9
    df = np.zeros((rows, cols))
    
    for Lambda_idx, Lambda in enumerate(lambdas):
        for sigma_idx, sigma in enumerate(sigmas):
            for beta_idx, beta in enumerate(betas):
                for alpha_idx, alpha in enumerate(alphas):
                    param_comb = Lambda_idx * len(sigmas) * len(alphas) * len(betas) + sigma_idx * len(alphas) * len(betas) + beta_idx * len(alphas) + alpha_idx
                    for run in range(n_runs):
                        _, steps, rewards = algorithm(env, Lambda = Lambda, 
                                                           sigma = sigma, 
                                                           n_episodes = n_episodes, 
                                                           alpha = alpha, 
                                                           epsilon = epsilon, 
                                                           beta = beta, 
                                                           gamma = gamma)
                        # get index of which row to fill
                        index = Lambda_idx * len(sigmas) * len(alphas) * len(betas) * n_runs + sigma_idx * len(alphas) * len(betas) * n_runs + beta_idx * len(alphas) * n_runs + alpha_idx * n_runs + run
                        start = index * n_episodes
                        end = index * n_episodes + n_episodes
                        df[start:end, 0] = param_comb
                        df[start:end, 1] = alpha
                        df[start:end, 2] = beta
                        df[start:end, 3] = Lambda
                        df[start:end, 4] = sigma
                        df[start:end, 5] = run
                        df[start:end, 6] = np.arange(n_episodes)
                        df[start:end, 7] = steps
                        df[start:end, 8] = rewards
                    # Print out how many percent of the simulation is done
                    print(str(np.round((param_comb + 1) / param_comb_n * 100, 2)) + "%")
    
    df = pd.DataFrame(df)
    df.columns = ['param_comb', 'alpha', "beta", "Lambda", "sigma", 
                  "run", "episode", "steps", "reward"]
    return df

#==============================================================================
# Plot
#==============================================================================
def plot(df, x_var = "episode", y_var = "steps",  col_var = "Lambda", ls_var = "sigma", 
         colors = ['C0', 'C1', 'C2', '777777', 'FBC15E', '8EBA42', 'FFB5B8'],
         linestyles = ['-', "--", ":"],
         file = "a.pdf", window = 1, ylim = [0, 500], xlim = [0, 200], 
         title = "", xlabel = "Episode", ylabel = "Steps per Episode",
         legend_pos_col = 1, legend_pos_ls = 9):

    # get unique values of color var and linestyle var
    col = np.unique(df[col_var]) 
    ls = np.unique(df[ls_var])

    #fig = plt.figure(figsize = (7.5, 5))
    #ax = fig.add_subplot(1, 1, 1)
    fig, axes = plt.subplots() # figsize = (12, 9))
    # ax = fig.add_subplot(1, 1, 1)
    for col_idx, col_val in enumerate(col):
        for ls_idx, ls_val in enumerate(ls):
            df2 = df[df[col_var] == col[col_idx]]
            df2 = df2[df2[ls_var] == ls[ls_idx]]
            steps_avg = running_mean(df2.as_matrix([y_var]), window)
            # print(steps_avg)
            x = np.arange(len(steps_avg))# + window / 2
            axes.plot(x, steps_avg, c = colors[col_idx], ls = linestyles[ls_idx])
            # plt.plot(df2["episode"], df2["steps"], label = i)
    plt.title(title, loc = "left")    
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.ylim(ylim)
    plt.xlim(xlim)
    #dummy lines with NO entries, just to create custom legends
    dummy_lines_ls = []
    for ls_idx, ls_val in enumerate(ls):
        dummy_lines_ls.append(axes.plot([],[], color = "black", ls = linestyles[ls_idx])[0])
    dummy_lines_col = []
    for col_idx, col_val in enumerate(col):
        dummy_lines_col.append(axes.plot([],[], ls = "-", color = colors[col_idx])[0])
    legend1 = plt.legend([dummy_lines_col[i] for i in range(len(dummy_lines_col))], 
                          ["%.2f" % number for number in col], 
                          loc = legend_pos_col, title = col_var)
    legend2 = plt.legend([dummy_lines_ls[i] for i in range(len(dummy_lines_ls))], 
                          ["%.2f" % number for number in ls], 
                          loc = legend_pos_ls, title = ls_var)
    axes.add_artist(legend1)
    axes.add_artist(legend2)

    # If window size > 1: 
    # change x axis ticks labels to an interval of data included in the window
    if window > 1:
        a = axes.get_xticks().tolist()
        for idx, val in enumerate(a):
            a[idx] = str(int(val)) + ":" + str(int(val + window))
        axes.set_xticklabels(a)
        # remove every second tick label
        for label in axes.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)
    # remove tickmarks for both axes
    plt.tick_params(
        axis = 'both',
        which = 'both',
        bottom = 'off',  
        left = "off",            
        labelbottom = 'on') 
    plt.savefig(file, format = "pdf")
    plt.show()

#==============================================================================
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


# Plot using ggplot
# from ggplot import *

# df["sigma"] = df["sigma"].astype('category')
# df["Lambda"] = df["Lambda"].astype('category')
# ggplot(df, aes(x_var, y_var, color = col_var, linetype = ls_var)) + geom_line() + theme_bw() + ylim(0, 400) + xlim(0, 100)
