
from util.t2m import t2m
from util.m2t import m2t
import numpy as np
from numpy.linalg import norm

def update_X(X, Lx, Lambda, Sigma, alpha, track_fval = False):
    """ Update function for variable X. This version does not utilize a 
    gradient descent but approximates a covariance Sigma using an auxiliary variable.
    """
    n = len(X)
    for i in range(n):
        Lmat = t2m(Lx[i], i)

        A_inv = np.identity(Lmat.shape[1])/alpha[0][i]
        U = Lmat.transpose()
        V = Lmat
        B_inv = np.identity(Lmat.shape[0])/alpha[1][i]
        invmat = np.linalg.inv(B_inv+V@U/alpha[0][i])
        inverse_mat = A_inv-U@invmat@V/(alpha[0][i]**2)

        Xmat_Lag = alpha[0][i] * t2m(Lx[i]-Lambda[0][i], i)
        Sigmat = alpha[1][i] * ((Sigma[i]+Lambda[1][i]).transpose() @ Lmat)
        Lagrangian = Xmat_Lag + Sigmat
        X[i] = m2t(Lagrangian @ inverse_mat, X[i].shape, i)

    fval, fval_X, fval_sig = fn_val(X,Lx,Lambda,Sigma,alpha) if track_fval else []
            
    return X, fval, fval_X, fval_sig

def fn_val(X, Lx, Lambda, Sigma, alpha):
    n = len(X)
    val_X = [alpha[0][i]*norm(X[i]+Lambda[0][i]-Lx[i])**2 for i in range(n)]

    covs = [t2m(Lx[i],i)@t2m(X[i],i).transpose() for i in range(n)]
    val_sig = [alpha[1][i]*norm(covs[i] - Sigma[i] - Lambda[1][i])**2 for i in range(n)]

    f_val = sum(val_sig) + sum(val_X)
    return f_val, val_X, val_sig

# Function value for a single mode. Might be useful.
# def fn_val_L(Lx, L, X, Lambda, Phi, alpha, theta, i):
#     f_val = 0
#     val_L = alpha[0]*np.sum((L-Lambda[0]-Lx)**2)
#     f_val += val_L
#     val_X = alpha[1]*np.sum((X+Lambda[1]-Lx)**2)
#     f_val += val_X
#     covs = t2m(Lx,i)@t2m(X,i).transpose()
#     val_comm = theta*np.sum((covs @ Phi - Phi @ covs)**2)
#     f_val =+ val_comm
#     return f_val

