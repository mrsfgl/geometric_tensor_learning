
import numpy as np
from util.t2m import t2m
from numpy.linalg import norm

def update_Sigma(Sigma, Lx, X, Phi, Lambda, alpha, theta, track_fval = False):
    ''' The update function for auxiliary variable Sigma. 
    This function tries to approximate a covariance function commutative with Sigma.
    '''
    
    n = len(Sigma)

    for i in range(n):
        c = alpha[i]/theta[i]
        I = [np.identity(Sigma[i].shape[0]), np.identity(Sigma[i].shape[1])]
        Psq = Phi[i]@Phi[i]
        invmat = np.kron(Psq, I[0])-2*np.kron(Phi[i], Phi[i])+np.kron(I[1], Psq) + c*np.identity(Phi[i].size)
        invmat = c * np.linalg.inv(invmat)
        Lag = t2m(Lx[i],i)@t2m(X[i],i).transpose()-Lambda[i]
        vecSig = np.tensordot(invmat, np.ravel(Lag, order = 'F'), axes=([1],[0]))
        Sigma[i] = vecSig.reshape(Phi[i].shape)

    if track_fval:
        fval, fval_sig, fval_comm = fn_val_L(Sigma, Lx, X, Phi, Lambda, alpha, theta)

    return Sigma, fval, fval_sig, fval_comm


def fn_val_L(Sigma, Lx, X, Phi, Lambda, alpha, theta):
    n = len(Sigma)

    covs = [t2m(Lx[i],i)@t2m(X[i],i).transpose() for i in range(n)]
    val_sig = [alpha[i]*norm(covs[i] - Sigma[i] - Lambda[i])**2 for i in range(n)]

    val_comm = [theta[i]*norm(Sigma[i]@Phi[i]-Phi[i]@Sigma[i])**2 for i in range(n)]

    f_val = sum(val_sig) + sum(val_comm)
    return f_val, val_comm, val_sig