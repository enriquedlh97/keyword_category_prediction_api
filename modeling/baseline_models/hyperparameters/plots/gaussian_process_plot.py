import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from matplotlib import gridspec

from bayes_opt import UtilityFunction


def get_principal_component(optimizer, column_names):
    """ Gets principal component from hyperparameter space
    :param optimizer:
    :param column_names:
    :return:
    """

    df_params_obs = pd.DataFrame(columns=column_names)

    for res in optimizer.res:
        df_params_obs = df_params_obs.append(res["params"], ignore_index=True)

    # Standardizing the features
    x_all_features = StandardScaler().fit_transform(df_params_obs.values)

    # Computing PCA
    pca_all_features = PCA(n_components=1)
    principal_components_all_features = pca_all_features.fit_transform(x_all_features)
    principal_df_all_features = pd.DataFrame(data=principal_components_all_features
                                             , columns=['principal component 1'])

    # Total explained variance by 2 principal components
    # pca_all_features.explained_variance_ratio_.sum()

    return principal_df_all_features.values


def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma


def plot_gp(optimizer, axis_color='black', fc_color='silver', ec_color='black', alpha=0.6):
    fig = plt.figure(figsize=(16, 10))
    steps = len(optimizer.space)
    fig.suptitle('Gaussian Process and Utility Function', size=30)

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])

    x_obs = get_principal_component(optimizer, column_names=list(optimizer.res[0]['params'].keys()))
    y_obs = np.array([res["target"] for res in optimizer.res])
    x = np.linspace(x_obs.min(), x_obs.max(), 10000).reshape(-1, 1)

    mu, sigma = posterior(optimizer, x_obs, y_obs, x)

    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label='Observations', color=axis_color)
    axis.plot(x, mu, '--', color=axis_color, label='Prediction')

    axis.fill(np.concatenate([x, x[::-1]]),
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
              alpha=alpha, fc=fc_color, ec=ec_color, label='95% confidence interval')

    #     axis.fill(np.concatenate([x, x[::-1]]),
    #                   np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
    #             alpha=.6, fc='silver', ec='k', label='95% confidence interval')

    axis.set_ylabel('5-fold cross-validation loss', fontdict={'size': 20})
    axis.set_xlabel('Principal component 1 from hyperparameter space', fontdict={'size': 20})

    utility_function = UtilityFunction(kind="ucb", kappa=5, xi=0)
    utility = utility_function.utility(x, optimizer._gp, 0)
    acq.plot(x, utility, label='Utility Function', color='k', linewidth=2)
    acq.plot(x[np.argmax(utility)], np.max(utility), 'o', markersize=11,
             label=u'Next Best Guess', markerfacecolor='k', markeredgecolor='black', markeredgewidth=1)

    acq.set_ylabel('Utility', fontdict={'size': 20})
    acq.set_xlabel('Principal component 1 from hyperparameter space', fontdict={'size': 20})

    axis.set_ylim((None, max(np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]).max(),
                             y_obs.max()) + 6))
    acq.set_ylim((None, np.max(utility).max() + 2.5))

    axis.legend(loc='upper right')
    acq.legend(loc='upper right')

    axis.grid()
    acq.grid()
