import numpy as np

##################### Artificial Hamiltonian Parameters ########################

# Energy matrix
e1_true = np.random.uniform(0, 5)
e2_true = np.random.uniform(0, 5)

# Overlap matrix real
t11_true = np.random.uniform(0, 5)
t12_true = np.random.uniform(0, 5)
t22_true = np.random.uniform(0, 5)
w_true = 1.#np.random.uniform(0, 5)

# Overlap matrix imaginary (diagonal terms are zero since H is Hermitian)
s12_true = np.random.uniform(0, 5)
q_true = 2.#np.random.uniform(0, 5)

###################### Hamiltonian Parameters Guesses ##########################

e1_guess = 1. #Lower bound
e2_guess = 1. #Lower bound

t11_guess = 1. #Lower bound
t12_guess = 1. #Lower bound
t22_guess = 1. #Lower bound
w_guess = 1. #Average

s12_guess = 1. #Lower bound
q_guess = 2. #Average

############################ Model hyperparameters #############################

learning_rates = [0.005] # Of the neural Network
batch_sizes = [16]
n_hidden_neurons = [35]
regularization_parameters = [0.]
ratios = [0.5]

############################## Other parameters ################################

# Number of times the same simulation is performed
n_repetitions = 1

# Number of epochs
number_of_epochs = 10

# Noise added to eigenvalues in dataset (maximum working noise ~1)
data_noise = 0.

# Number of experimental dataset points
N = 500

# Percentage of epochs for lr scheduling and scaling scheduling (for example every 80% changes lr)
perc = 0.8

# Computes bands from fitted Hamiltonian
perf_bands = True

# Perform PCA (significantly slow for epochs higher than ~5000)
perf_pca = False

# Save gradients on Tensorboard (slower by a factor ~5)
save_grad = False

# Plot fitted theta at every step
theta_video = False
