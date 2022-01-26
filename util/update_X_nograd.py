
from util.t2m import t2m
from util.m2t import m2t
import numpy as np
from numpy.linalg import norm


def update_X(X, Lx, Lambda, Sigma, alpha, track_fval=False):
    """ Update function for variable X. This version does not
    utilize a gradient descent but approximates a covariance
    Sigma using an auxiliary variable.
    """
    n = len(X)
    for i in range(n):
        Lmat = t2m(Lx[i], i)

        U = Lmat.transpose()
        V = Lmat
        B_inv = np.identity(Lmat.shape[0])/alpha[1][i]
        invmat = np.linalg.inv(B_inv+V@U/alpha[0][i])

        Xmat_Lag = alpha[0][i] * t2m(Lx[i]-Lambda[0][i], i)
        Sigmat = alpha[1][i] * ((Sigma[i]+Lambda[1][i]).transpose() @ Lmat)
        Lagrangian = Xmat_Lag + Sigmat
        X[i] = m2t(
            Lagrangian/alpha[0][i] -
            (Lagrangian @ U) @ invmat @ V/(alpha[0][i]**2),
            X[i].shape,
            i)

    (fval,
     fval_X,
     fval_sig
     ) = fn_val(X, Lx, Lambda, Sigma, alpha) if track_fval else [[],[],[]]

    return X, fval, fval_X, fval_sig


def fn_val(X, Lx, Lambda, Sigma, alpha):
    n = len(X)
    val_X = [alpha[0][i]*norm(X[i]+Lambda[0][i]-Lx[i])**2 for i in range(n)]

    covs = [t2m(Lx[i], i) @ t2m(X[i], i).transpose() for i in range(n)]
    val_sig = [alpha[1][i]*norm(covs[i] - Sigma[i] - Lambda[1][i])**2
               for i in range(n)]

    f_val = sum(val_sig) + sum(val_X)
    return f_val, val_X, val_sig
