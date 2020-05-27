import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from torch.utils.tensorboard import SummaryWriter

def pca_1d(parameter_history, e1, e2, t11, t12, t22, w, s12, q, n_epochs, path, iter_name, writer):

    # Store target vector as last row in parameter_history
    parameter_history[n_epochs][0] = e1
    parameter_history[n_epochs][1] = e2
    parameter_history[n_epochs][2] = t11
    parameter_history[n_epochs][3] = t12
    parameter_history[n_epochs][4] = t22
    parameter_history[n_epochs][5] = w
    parameter_history[n_epochs][6] = s12
    parameter_history[n_epochs][7] = q

    # Standardizing features
    parameter_history = StandardScaler().fit_transform(parameter_history)
    # Create object
    pca = PCA(n_components = 2)

    # Perform pca and store results in pca_parameter_history
    pca_parameter_history = pca.fit_transform(parameter_history)

    colors = cm.rainbow(np.linspace(0, 1, n_epochs))

    fig = plt.figure(figsize = (10, 10))

    for i in range(n_epochs):
        c = colors[i]
        plt.scatter(pca_parameter_history[i][0], pca_parameter_history[i][1], color=c, s = 5)
        plt.scatter(pca_parameter_history[n_epochs][0], pca_parameter_history[n_epochs][1], marker = '*', color = 'red')

    ############### Save plot in folder for Tensorboard visualization ###################

    writer.add_figure('Parameters trajectory' + iter_name, fig)
