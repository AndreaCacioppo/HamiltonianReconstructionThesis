# Hamiltonian Reconstruction

This repository contains the code necessary to reproduce the results presented in the Master's thesis "Deep learning for the parameter estimation of tight-binding Hamiltonians" by Andrea Cacioppo, available at ...

## Abstract 

The code available at this repository implements an algorithm whose purpose is the recovery of the Hamiltonian of a crystal in the tight-binding approximation given its electronic bands. The class of Hamiltonians which can currently be recovered is represented by the matrix

![](images/Hamiltonian.png)

The electronic bands are synthetically generated by taking the eigenvalues of this matrix.

## Getting Started

To get the running code clone or download the repository into your local computer.

### Prerequisites

Python and pyTorch and TensorboardX need to be installed.

## Structure of the code

The file to execute is named main.py, while the file containing all the tunable parameters is named CustomModules/Parameters.py (this and CustomModules/TightBinding.py are the only two files to be modified).

## Running the tests

In CustomModules/Parameters.py:

- Set the parameters of the artificial Hamiltonian
- Set the values of the hyperparameters

In CustomModules/TightBinding.py:

-Set the initial guesses for the Hamiltonian parameters with torch.randn and torch.rand

To read the results:

- In order to read the obtained Hamiltonian parameters and the training procedure run on a terminal: tensorboard --logdir /path/HamiltoninReconstructionThesis/Tensorboard1d and open the indicated link in a browser to see the results.

## Authors

* **Andrea Cacioppo** 

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
