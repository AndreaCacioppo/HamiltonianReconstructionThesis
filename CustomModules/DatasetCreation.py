import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random
import itertools
from tensorboardX import SummaryWriter
import math

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d

from CustomModules.Bands import bands_1d

def dataset_creation_1d(N, xAxis, data_noise, e1, e2, t11, t12, t22, w, s12, q, TB_path, name):

    eVal = np.zeros((N, 2))

    # Fill bands
    eVal[:,0], eVal[:,1] = bands_1d(N, xAxis, data_noise, e1, e2, t11, t12, t22, w, s12, q)

    fig1 = plt.figure(figsize = (10, 10))

    # Eigenvalues
    plt.scatter(xAxis, eVal[:,0], s = 2, color = 'red')
    plt.scatter(xAxis, eVal[:,1], s = 2, color = 'blue')

    plt.legend(('eVal1', 'eVal2'), loc='upper right')
    plt.title('Eigenvalues')

    ############### Save plot in folder for Tensorboard visualization ###################
    writer = SummaryWriter(TB_path)
    writer.add_figure(name + "-Electronic bands", fig1)
    writer.close()

    return eVal
