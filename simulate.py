from q_sigma_lambda import qSigma
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Plot style: ggplot, but with white background defined in file theme_bw.mplstyle
theme_bw = "C:/Users/M/Desktop/q_sigma_lambda/Python/ggplot_white.mplstyle"
plt.style.use(theme_bw)

#==============================================================================
# Function to simulate data for different hyperparameter combinations and return results
#==============================================================================
def simulate(env, algorithm = qSigma, n_episodes = 100, n_runs = 1, 
             epsilon = 0.1, gamma = 1, alphas = [0.1], betas = [0], 
             lambdas = [0], sigmas = [1], printing = False, 
             target_policy = ["greedy"], update_sigma = 1):

    # number of different parameter combinations
    param_comb_n = len(sigmas) * len(lambdas) * len(alphas) * len(betas) * len(target_policy)
    
    # number of runs
    rows = n_episodes * n_runs * param_comb_n
    cols = 10
    df = np.zeros((rows, cols))
    param_comb = 0
    for Lambda_idx, Lambda in enumerate(lambdas):
        for sigma_idx, sigma in enumerate(sigmas):
            for beta_idx, beta in enumerate(betas):
                for alpha_idx, alpha in enumerate(alphas):
                    for target_idx, target in enumerate(target_policy):
                        for run in range(n_runs):
                            steps, returns, _, _ = algorithm(env, n_episodes = n_episodes, 
                                                          Lambda = Lambda, 
                                                          sigma = sigma, 
                                                          beta = beta, 
                                                          epsilon = epsilon, 
                                                          alpha = alpha, 
                                                          gamma = gamma, 
                                                          target_policy = target_policy, 
                                                          printing = printing, update_sigma = update_sigma)
                            # get index of which row to fill
                            index = param_comb * n_runs + run
                            start = index * n_episodes
                            end = index * n_episodes + n_episodes
                            df[start:end, 0] = update_sigma
                            df[start:end, 1] = alpha
                            df[start:end, 2] = beta
                            df[start:end, 3] = Lambda
                            df[start:end, 4] = sigma
                            df[start:end, 5] = target_idx
                            df[start:end, 6] = run + 1
                            df[start:end, 7] = np.arange(n_episodes) + 1
                            df[start:end, 8] = steps
                            df[start:end, 9] = returns
                            # Print out how many of the simulation is done
                            print(str(np.round((param_comb * n_runs + (run + 1)) / (param_comb_n * n_runs), 2)))
                        param_comb += 1
    
    df = pd.DataFrame(df)
    df.columns = ["update_sigma", 'Alpha', "Beta", "Lambda", "Sigma", 
                  "target_policy", "Run", "Episode", "Steps", "Returns"]
    map_dict = {}
    for i in range(len(target_policy)):
        map_dict[i] = target_policy[i]
    df["target_policy"] = df["target_policy"].map(map_dict)
    return df

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

#==============================================================================
def average_runs(df):
    df = df.groupby(['update_sigma', 'Alpha', "Beta", "Lambda", 
                     "Sigma", "target_policy", "Episode"], as_index = False)["Steps", "Returns"].mean()
    return df

#==============================================================================
def average_episodes(df):
    df = df.groupby(['update_sigma', 'Alpha', "Beta", "Lambda", 
                     "Sigma", "target_policy"], as_index = False)["Steps", "Returns"].mean()
    return df
