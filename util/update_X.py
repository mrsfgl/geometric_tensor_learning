from util.t2m import t2m
from util.m2t import m2t
import numpy as np

def update_X(X, Lx, Lambda, Phi, alpha, theta, lrn_rate = 0.01, step=1, num_iter = 100, track_fval = False):
    """ Update function for variable X. 
    Current version uses gradient descent for the update. 
    ! Needs to be checked.
    """
    n = len(X)
    
    fval = []
    for i in range(n):
        Phisq = np.matmul(Phi[i], Phi[i])
        Lmat = t2m(Lx[i], i)
        Lmat0 = alpha[i] * t2m(Lx[i]-Lambda[i], i)
        for x in range(num_iter):
            Xmat = t2m(X[i], i)
            grad = (theta[i]*(Xmat @ Lmat.transpose() @ Phisq @ Lmat +
            Phisq @ Xmat @ Lmat.transpose() @ Lmat -
            Phi[i] @ Xmat @ Lmat.transpose() @ Phi[i] @ Lmat) + 
            alpha[i]*Xmat - Lmat0)
            
            X[i] = m2t(Xmat - step * grad, X[i].shape, i)
            
            if x%10 == 0:
                step *= lrn_rate

            if track_fval:
                fval.append(fn_val_X(X, Lx, Lambda, Phi, alpha, theta))

    return X, fval


def fn_val_X(X, Lx, Lambda, Phi, alpha, theta):
    n = len(X)
    f_val = 0
    val_L = sum([alpha[i]*np.sum((X+Lambda[i]-Lx[i])**2) for i in range(n)])
    f_val += val_L
    covs = [t2m(Lx[i],i)@t2m(X[i],i).transpose() for i in range(n)]
    val_comm = sum([theta[i]*np.sum((covs[i] @ Phi[i] - Phi[i] @ covs[i])**2) for i in range(n)])
    f_val =+ val_comm
    return f_val