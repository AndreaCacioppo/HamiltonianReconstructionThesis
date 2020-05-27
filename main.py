import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random
from datetime import datetime
import time
import math
import os

import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

import tensorflow as tf

import pandas as pd

from CustomModules import Parameters as PA
from CustomModules.DatasetCreation import dataset_creation_1d
from CustomModules.TightBinding import H_TB_1d
from CustomModules.CustomDataset import custom_dataset_1d
from CustomModules.TrainStep import make_train_step_1d
from CustomModules.LearningRate import exp_lr_scheduler
from CustomModules.PCA import pca_1d
from CustomModules.NeuralNetwork import H_NN_1d

# Train step is performed as follows:
# A value of k is fed to H_TB and H_NN, the output of H_NN is manipulated to obtain an estimate of the Hamiltonian, values of the electronic bands are given in dataset.
# Loss is computed between the Hamiltonian estimated by H_TB and H_NN
# Gradient descent is performed jointly between the two models

# Create path for Tensorboard log files
path = 'Tensorboard_1d/' + time.strftime("%Y_%m_%d-%H_%M_%S-layer")

#################### Create dataset ######################

N = PA.N

eVal = np.zeros((N, 2))
H = np.zeros((N, 2, 2))

# Energy matrix
e1_true = PA.e1_true
e2_true = PA.e2_true

# Overlap matrix real
t11_true = PA.t11_true
t12_true = PA.t12_true
t22_true = PA.t22_true
w_true = PA.w_true

# Overlap matrix imaginary (diagonal terms are zero since H is hermitian)
s12_true = PA.s12_true
q_true = PA.q_true

xAxis = np.zeros((N, 1))

########################### Find the Brillouin zone ############################
freq = min(w_true, q_true)
span = math.ceil(2*math.pi/freq)

spacing = span/N

# Fill x axis in the Brillouin zone
for i in range(N):
    xAxis[i] = i*spacing - span/2.

with SummaryWriter(path) as writer:
    writer.add_scalar('e1-true', e1_true)
    writer.add_scalar('e2-true', e2_true)
    writer.add_scalar('t11-true', t11_true)
    writer.add_scalar('t12-true', t12_true)
    writer.add_scalar('t22-true', t22_true)
    writer.add_scalar('w-true', w_true)
    writer.add_scalar('s12-true', s12_true)
    writer.add_scalar('q-true', q_true)
from sys import argv
import math

fig = plt.figure()

data_noise = PA.data_noise

eVal = dataset_creation_1d(N, xAxis, PA.data_noise, e1_true, e2_true, t11_true, t12_true, t22_true, w_true, s12_true, q_true, path, 'Dataset')

################### Load dataset ######################
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Convert Numpy arrays to PyTorch tensors
x_tensor = torch.from_numpy(xAxis).float() # Values of k
y_tensor = torch.from_numpy(eVal).float() # Eigenvalues

############## Create dictionary for using add_hparams ###############
hparam = {'learning_rate': PA.learning_rates,
        'batch_size': PA.batch_sizes,
        'n_hidden_neurons': PA.n_hidden_neurons,
        'regularization_parameter': PA.regularization_parameters}

metrics = {'accuracy', 'loss'}

############ Loop over parameters written in dictionary ################
with SummaryWriter(path) as writer:
    for lr in hparam['learning_rate']:
        for bs in hparam['batch_size']:
            for hn in hparam['n_hidden_neurons']:
                for rp in hparam['regularization_parameter']:
                    for rt in PA.ratios:
                        for rep in range(PA.n_repetitions):
                            name1 = '%s' %lr
                            name2 = '%s' %bs
                            name3 = '%s' %hn
                            name4 = '%s' %rp
                            name5 = '%s' %PA.data_noise
                            name6 = '%s' %rt
                            name7 = '%s' %(rep + 1)

                            # Quit if batch_size is bigger than data points
                            if bs>N*N:
                                print("Batch size must be smaller than the number of experimental points")
                                quit()

                            iter_name = 'lr' + name1 + '-bs' + name2 + '-hn' + name3 + '-rp' + name4 + '-dn' + name5 + '-rt' + name6 + '-rep' + name7
                            iter_name_ = 'lr' + name1 + '-bs' + name2 + '-hn' + name3 + '-rp' + name4 + '-dn' + name5

                            #Create object for dataset
                            dataset = custom_dataset_1d(x_tensor, y_tensor)

                            perc = 100 #percentage of dataset given to training set
                            tra = round(perc/100*N)
                            val = N - tra

                            train_dataset, val_dataset = random_split(dataset, [tra, val])

                            #split dataset into batches
                            train_loader = DataLoader(dataset = train_dataset, batch_size = bs, shuffle = True, drop_last = True)
                            val_loader = DataLoader(dataset = val_dataset, batch_size = bs)

                            ##################### Model training ######################

                            # Create objects from class H_TB and H_NN
                            y_batch = torch.zeros((1, bs)).to(device) # Only needed to define modelNN
                            modelTB = H_TB_1d(bs)
                            modelNN = H_NN_1d(hn, bs, y_batch)

                            # Enables parallel GPU computation
                            modelTB = nn.DataParallel(modelTB).to(device)
                            modelNN = nn.DataParallel(modelNN).to(device)

                            print('------------------')
                            print("Now using", torch.cuda.device_count(), "GPUs")
                            writer.add_scalar('GPUs', torch.cuda.device_count())

                            # Specify loss function and optimizer to use in training
                            loss_fn = nn.SmoothL1Loss(reduction = 'mean')
                            optimizerTB = optim.Adam(modelTB.parameters(), lr = rt*lr, weight_decay = 0)
                            optimizerNN = optim.Adam(modelNN.parameters(), lr = lr, weight_decay = rp)

                            # Creates the train_step function for the model, loss function and optimizer
                            train_step_1d = make_train_step_1d(modelTB, modelNN, loss_fn, optimizerTB, optimizerNN, bs, device)

                            # n_epochs is stored in CustomModules/Parameters.py
                            n_epochs = PA.number_of_epochs

                            # Create a 2-dim arrays for storing the evolution of the 6 parameters of modelTB
                            # Last index corresponds to the target
                            parameter_history = np.zeros((n_epochs + 1, 8))
                            losses = np.zeros((n_epochs))

                            plt.ion()

                            for epoch in range(n_epochs):
                                # Filling parameter history
                                #parameter_history[epoch, :] = modelTB.module.e1.item(), modelTB.module.e2.item(), modelTB.module.t11.item(), modelTB.module.t12.item(), modelTB.module.t22.item(), modelTB.module.w.item(), modelTB.module.s11.item(), modelTB.module.s12.item(), modelTB.module.s22.item(), modelTB.module.q.item()

                                optimizerNN = exp_lr_scheduler(optimizerNN, epoch, lr, round(n_epochs*PA.perc)) # Decay every 30%

                                for x_batch, y_batch in train_loader:

                                    x_batch = x_batch.to(device)
                                    y_batch = y_batch.to(device)

                                    #One training step for every batch
                                    loss = train_step_1d(x_batch, y_batch, epoch, writer, rep)
                                    writer.add_scalar('loss ' + iter_name, loss, epoch)
                                    losses[epoch] = loss

                                # Save trajectory, this will be passed to PrCoAn()
                                parameter_history[epoch][0] = modelTB.module.e1.item()
                                parameter_history[epoch][1] = modelTB.module.e2.item()
                                parameter_history[epoch][2] = modelTB.module.t11.item()
                                parameter_history[epoch][3] = modelTB.module.t12.item()
                                parameter_history[epoch][4] = modelTB.module.t22.item()
                                parameter_history[epoch][5] = modelTB.module.w.item()
                                parameter_history[epoch][6] = modelTB.module.s12.item()
                                parameter_history[epoch][7] = modelTB.module.q.item()

                                writer.add_scalar('e1-rep' + str(rep+1), modelTB.module.e1, epoch)
                                writer.add_scalar('e2-rep' + str(rep+1), modelTB.module.e2, epoch)
                                writer.add_scalar('t11-rep' + str(rep+1), modelTB.module.t11, epoch)
                                writer.add_scalar('t12-rep' + str(rep+1), modelTB.module.t12, epoch)
                                writer.add_scalar('t22-rep' + str(rep+1), modelTB.module.t22, epoch)
                                writer.add_scalar('w-rep' + str(rep+1), modelTB.module.w, epoch)
                                writer.add_scalar('s12-rep' + str(rep+1), modelTB.module.s12, epoch)
                                writer.add_scalar('q-rep' + str(rep+1), modelTB.module.q, epoch)

                            plt.close()

                            # Plot predicted bands on Tensorboard
                            if PA.perf_bands == True:
                                print('plotting fitted bands...')
                                FiteVal = dataset_creation_1d(N, xAxis, PA.data_noise, modelTB.module.e1.item(), modelTB.module.e2.item(), modelTB.module.t11.item(), modelTB.module.t12.item(), modelTB.module.t22.item(), modelTB.module.w.item(), modelTB.module.s12.item(), modelTB.module.q.item(), path, 'Fitted Bands' + iter_name)

                            ################ At the end of every training performs PCA ###################
                            if PA.perf_pca == True:
                                print('performing PCA...')
                                pca_1d(parameter_history, e1_true, e2_true, t11_true, t12_true, t22_true, w_true, s12_true, q_true, n_epochs, path, iter_name, writer)

                            ############################# Save hyperparameters ##############################
                            print(str(rep+1) + ' - saving hyperparameters...')
                            # Compute accuracy of model knowing the real Hamiltonian, synthetically generated
                            accuracy1 = (((e1_true - modelTB.module.e1.item())**2) +((e2_true - modelTB.module.e2.item())**2) + ((t11_true - modelTB.module.t11.item())**2) + ((t12_true - modelTB.module.t12.item())**2) + ((t22_true - modelTB.module.t22.item())**2) + ((w_true - modelTB.module.w.item())**2) +  + ((s12_true - modelTB.module.s12.item())**2) + ((q_true - modelTB.module.q.item())**2))
                            accuracy2 = (((e2_true - modelTB.module.e1.item())**2) +((e1_true - modelTB.module.e2.item())**2) + ((t22_true - modelTB.module.t11.item())**2) + ((t12_true - modelTB.module.t12.item())**2) + ((t11_true - modelTB.module.t22.item())**2) + ((w_true - modelTB.module.w.item())**2) +  + ((s12_true - modelTB.module.s12.item())**2) + ((q_true - modelTB.module.q.item())**2))

                            # Taking into account invariance under exchange of e1-e2 and t11-t22
                            accuracy = min(accuracy1, accuracy2)

                            # Maps accuracy in the interval [0, 100], 100 means completely accurate
                            accuracy = 100/(1+accuracy)

                            # Saves average loss over the last 10 steps for evaluating the overall performance of the model
                            final_loss = 0
                            for i in range(n_epochs-9, n_epochs-1):
                                final_loss += losses[i]
                            final_loss = final_loss/10.

                            # Save parameters in Tensorboard log file
                            writer.add_hparams({'learning_rate': lr, 'batch_size': bs, 'n_hidden': hn, 'regularization_parameter': rp},
                                       	    {'accuracy': accuracy, 'loss': final_loss})#so to check inverse prop with accuracy

                    # For data visualization run on terminal: tensorboard --logdir path/Tensorboard
