
from util.t2m import t2m
from util.m2t import m2t
import numpy as np
from numpy.linalg import norm


def update_L(Lx, L, X, Lambda, Sigma, alpha, track_fval=False):
    """ Update function for variable Lx. This version does not
    utilize a gradient descent but approximates a covariance
    Sigma using an auxiliary variable.
    """
    n = len(Lx)
    for i in range(n):
        Xmat = t2m(X[i], i)

        c = 1/(alpha[0][i]+alpha[1][i])  # temporary constant
        invmat = np.linalg.inv(
            np.identity(Xmat.shape[0])/alpha[2][i] +
            c*Xmat@Xmat.transpose()
            )*c**2

        Lmat = alpha[0][i] * t2m(L-Lambda[0][i], i)
        Xmat_Lag = alpha[1][i] * t2m(X[i]+Lambda[1][i], i)
        Sigmat = alpha[2][i] * ((Sigma[i]+Lambda[2][i]) @ Xmat)
        Lagrangian = Lmat+Xmat_Lag+Sigmat
        Lx[i] = m2t(
            c*Lagrangian -
            (Lagrangian @ Xmat.transpose()) @ invmat @ Xmat,
            X[i].shape,
            i
            )

    (fval,
     fval_L,
     fval_X,
     fval_sig
     ) = fn_val(Lx, L, X, Lambda, Sigma, alpha) if track_fval else [[],[],[],[]]

    return Lx, fval, fval_L, fval_X, fval_sig


def fn_val(Lx, L, X, Lambda, Sigma, alpha):
    n = len(Lx)
    val_L = [alpha[0][i]*norm(L-Lambda[0][i]-Lx[i])**2 for i in range(n)]
    val_X = [alpha[1][i]*norm(X[i]+Lambda[1][i]-Lx[i])**2 for i in range(n)]

    covs = [t2m(Lx[i], i) @ t2m(X[i], i).transpose() for i in range(n)]
    val_sig = [alpha[2][i]*norm(covs[i]-Sigma[i]-Lambda[2][i])**2
               for i in range(n)]

    f_val = sum(val_sig) + sum(val_X) + sum(val_L)
    return f_val, val_L, val_X, val_sig
