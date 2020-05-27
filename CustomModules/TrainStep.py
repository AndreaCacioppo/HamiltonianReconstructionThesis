import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np

from CustomModules import Parameters as PA
theta_video = PA.theta_video

# Define training step, this takes the 2 outputs and computes loss and gradients
def make_train_step_1d(modelTB, modelNN, loss_fn, optimizerTB, optimizerNN, batch_size, device):
    # Builds function that performs a step in the train loop
    def train_step_1d(x_batch, y_batch, epoch, writer, rep): # y are the electronic bands used to compute H_NeuralNetwork

        # Fill y1 and y2 with H_TightBinding and H_NeuralNetwork
        real_H_TB, imaginary_H_TB = modelTB(x_batch, batch_size)
        real_H_NN, imaginary_H_NN, theta, phi = modelNN(x_batch, batch_size, y_batch)

        real_H_TB = real_H_TB.to(device)
        real_H_NN = real_H_NN.to(device)
        imaginary_H_TB = imaginary_H_TB.to(device)
        imaginary_H_NN = imaginary_H_NN.to(device)

        # Train mode
        modelTB.train()
        modelNN.train()

        # Computes loss. This is done as logarithm of MSE plus a term discouraging H_TB parameters to get to negative values, which aren't physically possible
        loss = loss_fn(real_H_TB, real_H_NN) + loss_fn(imaginary_H_TB, imaginary_H_NN) + (torch.exp(-20*modelTB.module.e1)/100) + (torch.exp(-20*modelTB.module.e2)/100) + (torch.exp(-20*modelTB.module.t11)/100) + (torch.exp(-20*modelTB.module.t12)/100) + (torch.exp(-20*modelTB.module.t22)/100) + (torch.exp(-20*modelTB.module.w)/100) + (torch.exp(-20*modelTB.module.s12)/100) + (torch.exp(-20*modelTB.module.q)/100)

        writer.add_scalar('Real loss' + str(rep+1), loss_fn(real_H_TB, real_H_NN), epoch)
        writer.add_scalar('Imaginary loss' + str(rep+1), loss_fn(imaginary_H_TB, imaginary_H_NN), epoch)

        # Computes gradients
        loss.sum().backward()

        # Plot fitted theta in real time
        if theta_video == True:
            plot_theta_1d(epoch, x_batch, theta)

        # Saves gradients as histograms on Tensorboard
        if PA.save_grad == True:
            writer.add_histogram('Gradients_in_hid1_weight-rep' + str(rep+1), modelNN.module.input_hidden1.weight.grad, epoch)
            writer.add_histogram('Gradients_hid1_hid2_weight-rep' + str(rep+1), modelNN.module.hidden1_hidden2.weight.grad, epoch)
            writer.add_histogram('Gradients_hid2_out_weight-rep' + str(rep+1), modelNN.module.hidden2_output.weight.grad, epoch)

            writer.add_histogram('Gradients_in_hid1_bias-rep' + str(rep+1), modelNN.module.input_hidden1.bias.grad, epoch)
            writer.add_histogram('Gradients_hid1_hid2_bias-rep' + str(rep+1), modelNN.module.hidden1_hidden2.bias.grad, epoch)
            writer.add_histogram('Gradients_hid2_out_bias-rep' + str(rep+1), modelNN.module.hidden2_output.bias.grad, epoch)

            writer.add_histogram('Theta-rep' + str(rep+1), theta, epoch)
            writer.add_histogram('Phi-rep' + str(rep+1), phi, epoch)

            writer.add_scalar('e1_grad-rep' + str(rep+1), modelTB.module.e1.grad, epoch)
            writer.add_scalar('e2_grad-rep' + str(rep+1), modelTB.module.e2.grad, epoch)
            writer.add_scalar('t11_grad-rep' + str(rep+1), modelTB.module.t11.grad, epoch)
            writer.add_scalar('t12_grad-rep' + str(rep+1), modelTB.module.t12.grad, epoch)
            writer.add_scalar('t22_grad-rep' + str(rep+1), modelTB.module.t22.grad, epoch)
            writer.add_scalar('w_grad-rep' + str(rep+1), modelTB.module.w.grad, epoch)
            writer.add_scalar('s12_grad-rep' + str(rep+1), modelTB.module.s12.grad, epoch)
            writer.add_scalar('q_grad-rep' + str(rep+1), modelTB.module.q.grad, epoch)

        # Updates parameters
        optimizerTB.step()
        optimizerNN.step()

        # Zeroes gradients
        optimizerTB.zero_grad()
        optimizerNN.zero_grad()

        # Returns the loss
        return loss.item()

    # train_step will be called inside the train loop
    return train_step_1d
