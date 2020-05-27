import torch
import torch.nn as nn
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class H_NN_1d(nn.Module):
    def __init__(self, hidden_neurons, batch_size, eVal):
        super().__init__()

        # Parameters Neural Network
        self.input_hidden1 = nn.Linear(1, hidden_neurons)
        self.hidden1_hidden2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.hidden2_output = nn.Linear(hidden_neurons, 2)

        self.sigmoid_hidden1 = nn.ReLU()
        self.sigmoid_hidden2 = nn.ReLU()

    def forward(self, x, batch_size, eVal):

        # Outputs Neural Network
        h1a = self.input_hidden1(x)
        h1b = self.sigmoid_hidden1(h1a)
        h2a = self.hidden1_hidden2(h1b)
        h2b = self.sigmoid_hidden2(h2a)
        theta, phi = self.hidden2_output(h2b).permute(1,0)

        real_H = torch.zeros((batch_size, 4)).to(device)
        imaginary_H = torch.zeros((batch_size, 4)).to(device)

        # Build real_H and imaginary_H from orthonormalized vectors

        real_H[:,0] = eVal[:,0]*(torch.cos(theta[:]))**2 + eVal[:,1]*(torch.sin(theta[:]))**2
        real_H[:,1] = 0.5*(eVal[:,0]-eVal[:,1])*(torch.sin(2*theta)*torch.cos(phi))
        real_H[:,2] = 0.5*(eVal[:,0]-eVal[:,1])*(torch.sin(2*theta)*torch.cos(phi))
        real_H[:,3] = eVal[:,0]*(torch.sin(theta[:]))**2 + eVal[:,1]*(torch.cos(theta[:]))**2

        imaginary_H[:,0] = 0.
        imaginary_H[:,1] = 0.5*(eVal[:,0]-eVal[:,1])*(torch.sin(2*theta)*torch.sin(phi))
        imaginary_H[:,2] = 0.5*(eVal[:,1]-eVal[:,0])*(torch.sin(2*theta)*torch.sin(phi))
        imaginary_H[:,3] = 0.

        # Returns real and imaginary part of H (only 12 element)
        return real_H, imaginary_H, theta, phi
