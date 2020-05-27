import numpy as np
import random
import cmath

from numpy.linalg import eigvalsh
from numpy import linalg as LA

def bands_1d(N, xAxis, data_noise, e1, e2, t11, t12, t22, w, s12, q):

    eVal = np.zeros((N, 2))
    eVec = np.zeros((N, 2, 2))

    H = np.zeros((N, 2, 2), dtype=complex)

    # Fill H(k)
    for i in range(N):
        H[i][0][0] = complex(e1 + t11*np.cos(w*xAxis[i]), 0.)
        H[i][0][1] = complex(t12*np.cos(w*xAxis[i]), s12*np.cos(q*xAxis[i]))
        H[i][1][0] = complex(t12*np.cos(w*xAxis[i]), -s12*np.cos(q*xAxis[i]))
        H[i][1][1] = complex(e2 + t22*np.cos(w*xAxis[i]), 0.)

    # Compute eVec from the true hamiltonian
    for i in range(N):
        eVal[i,:] = eigvalsh(H[i])

    return eVal[:,0], eVal[:,1]
