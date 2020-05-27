import torch
import torch.optim as optim
import torch.nn as nn

from CustomModules import Parameters as PA

import numpy as np
import numpy.random as rn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Create class for model TH Hamiltonian
class H_TB_1d(nn.Module):
    def __init__(self, batch_size):
        super().__init__()

        # N.B. torch.randn returns a random number according to gauss(0,1)
        # while torch.rand returns a real number uniformly distributed in (0,1)

        #Parameters H_TightBinding
        self.e1 = nn.Parameter(5*torch.rand(1, requires_grad = True, dtype = torch.float) + PA.e1_guess)
        self.e2 = nn.Parameter(5*torch.rand(1, requires_grad = True, dtype = torch.float) + PA.e2_guess)
        self.t11 = nn.Parameter(5*torch.rand(1, requires_grad = True, dtype = torch.float) + PA.t11_guess)
        self.t12 = nn.Parameter(5*torch.rand(1, requires_grad = True, dtype = torch.float) + PA.t12_guess)
        self.t22 = nn.Parameter(5*torch.rand(1, requires_grad = True, dtype = torch.float) + PA.t22_guess)
        self.w = nn.Parameter(0.1*torch.randn(1, requires_grad = True, dtype = torch.float) + PA.w_guess)
        self.s12 = nn.Parameter(5*torch.rand(1, requires_grad = True, dtype = torch.float) + PA.s12_guess)
        self.q = nn.Parameter(0.1*torch.randn(1, requires_grad = True, dtype = torch.float) + PA.q_guess)

    def forward(self, x, batch_size):

        #Outputs H_TightBinding
        rout11 = self.e1 + self.t11*torch.cos(self.w*x)
        rout12 = self.t12*torch.cos(self.w*x)
        rout21 = self.t12*torch.cos(self.w*x)
        rout22 = self.e2 + self.t22*torch.cos(self.w*x)

        iout11 = torch.zeros((batch_size, 1)).to(device)
        iout12 = self.s12*torch.cos(self.q*x)
        iout21 = -self.s12*torch.cos(self.q*x)
        iout22 = torch.zeros((batch_size, 1)).to(device)

        #Encode H_TightBinding in a (BatchSize x 3) tensor
        real_H = torch.cat((rout11, rout12, rout21, rout22), 1)
        imaginary_H = torch.cat((iout11, iout12, iout21, iout22), 1)

        return real_H, imaginary_H
