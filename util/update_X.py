from util.t2m import t2m
from util.m2t import m2t
import numpy as np

def update_X(X, Lx, Lambda, Phi, alpha, theta, lrn_rate = 1, step=0.01, num_iter = 100, track_fval = False):
    """ Update function for variable X. 
    Current version uses gradient descent for the update. 
    ! Needs to be checked.
    """
    n = len(X)
    
    fval = []
    Phisq = []
    Lmat = []
    Lmatl = []
    for i in range(n):
        Phisq.append(np.matmul(Phi[i], Phi[i]))
        Lmat.append(t2m(Lx[i], i))
        Lmatl.append(alpha[i] * t2m(Lx[i]-Lambda[i], i))

    for x in range(num_iter):
        for i in range(n):
            Xmat = t2m(X[i], i)
            grad = (theta[i]*((Xmat @ Lmat[i].transpose() @ Phisq[i]) @ Lmat[i] +
            (Phisq[i] @ Xmat @ Lmat[i].transpose()) @ Lmat[i] -
            (Phi[i] @ Xmat @ Lmat[i].transpose() @ Phi[i]) @ Lmat[i]) + 
            alpha[i]*Xmat - Lmatl[i])
            
            X[i] = m2t(Xmat - step * grad, X[i].shape, i)
        
        if x%10 == 9:
            step *= lrn_rate

        if track_fval:
            fval.append(fn_val_X(X, Lx, Lambda, Phi, alpha, theta))

    return X, fval


def fn_val_X(X, Lx, Lambda, Phi, alpha, theta):
    n = len(X)

    val_L = [alpha[i]*np.linalg.norm(X[i]+Lambda[i]-Lx[i])**2 for i in range(n)]

    covs = [t2m(Lx[i],i)@t2m(X[i],i).transpose() for i in range(n)]
    val_comm = [theta[i]*np.linalg.norm(covs[i] @ Phi[i] - Phi[i] @ covs[i])**2 for i in range(n)]

    f_val = sum(val_comm) + sum(val_L)
    return f_val, val_L, val_comm