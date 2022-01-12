
from util.t2m import t2m
from util.m2t import m2t
import numpy as np
from numpy.linalg import norm

def update_L(Lx, L, X, Lambda, Sigma, alpha, track_fval = False):
    """ Update function for variable Lx. This version does not utilize a 
    gradient descent but approximates a covariance Sigma using an auxiliary variable.
    """
    n = len(Lx)
    for i in range(n):
        Xmat = t2m(X[i], i)

        c = (alpha[2][i]/(alpha[0][i]+alpha[1][i])) # temporary constant
        A = c*np.identity(Xmat.shape[1])/alpha[2][i]
        invmat = np.linalg.inv(np.identity(Xmat.shape[0]) + c*Xmat@Xmat.transpose())/alpha[2][i]
        inverse_mat = A - Xmat.transpose() @ invmat @ Xmat

        Lmat = alpha[0][i] * t2m(L-Lambda[0][i], i)
        Xmat_Lag = alpha[1][i] * t2m(X[i]+Lambda[1][i], i)
        Sigmat = alpha[2][i] * ((Sigma[i]+Lambda[2][i]) @ Xmat)
        Lagrangian = Lmat+Xmat_Lag+Sigmat
        Lx[i] = m2t(Lagrangian @ inverse_mat, X[i].shape, i)

    if track_fval:
        fval, fval_L, fval_X, fval_sig = fn_val(Lx,L,X,Lambda,Sigma,alpha)
            
    return Lx, fval, fval_L, fval_X, fval_sig

def fn_val(Lx, L, X, Lambda, Sigma, alpha):
    n = len(Lx)
    val_L = [alpha[0][i]*norm(L-Lambda[0][i]-Lx[i])**2 for i in range(n)]
    val_X = [alpha[1][i]*norm(X[i]+Lambda[1][i]-Lx[i])**2 for i in range(n)]

    covs = [t2m(Lx[i],i)@t2m(X[i],i).transpose() for i in range(n)]
    val_sig = [alpha[2][i]*norm(covs[i]-Sigma[i]-Lambda[2][i])**2 for i in range(n)]

    f_val = sum(val_sig) + sum(val_X) + sum(val_L)
    return f_val, val_L, val_X, val_sig

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

