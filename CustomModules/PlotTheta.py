import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from CustomModules.Parameters import number_of_epochs

def plot_theta_1d(epoch, x_batch, theta):

  plt.title('Fitted θ - epoch: ' + str(epoch) + '/' + str(number_of_epochs))
  x = x_batch[:,0].cpu().detach().numpy()
  y = theta.cpu().detach().numpy()
  plt.scatter(x, y, s=4)
  plt.pause(0.01)
  plt.cla()
