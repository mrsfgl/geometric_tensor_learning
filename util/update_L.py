from numpy.lib.function_base import append
from util.t2m import t2m
from util.m2t import m2t
import numpy as np

def update_L(Lx, L, X, Lambda, Phi, alpha, theta, lrn_rate = 1, step=0.05, num_iter = 100, track_fval = False):
    """ Update function for variable Lx. 
    Current version uses gradient descent for the update. 
    """
    n = len(Lx)
    fval = []
    fval_L = []
    fval_X = []
    fval_comm = []
    Phisq = []
    Xmat = []
    Lagrangians = []
    for i in range(n):
        Phisq.append(np.matmul(Phi[i], Phi[i]))
        Xmat.append(t2m(X[i], i))
        Lmatl = alpha[0][i] * t2m(L-Lambda[0][i], i)
        Xmatl = alpha[1][i] * t2m(X[i]+Lambda[1][i], i)
        Lagrangians.append(Lmatl+Xmatl)

    for x in range(num_iter):
        for i in range(n):
            Lmat = t2m(Lx[i], i)
            grad = (theta[i]*((Lmat @ Xmat[i].transpose() @ Phisq[i]) @ Xmat[i] +
            (Phisq[i] @ Lmat @ Xmat[i].transpose()) @ Xmat[i] -
            2*(Phi[i] @ Lmat @ Xmat[i].transpose() @ Phi[i]) @ Xmat[i]) + 
            (alpha[0][i]+alpha[1][i])*Lmat - Lagrangians[i])
            
            Lx[i] = m2t(Lmat - step*grad, Lx[i].shape, i)

        if x%20 == 19:
            step *= lrn_rate

        if track_fval:
            fvals = fn_val_L(Lx,L,X,Lambda,Phi,alpha,theta)
            fval.append(fvals[0])
            fval_L.append(fvals[1])
            fval_X.append(fvals[2])
            fval_comm.append(fvals[3])
            
    return Lx, fval, fval_L, fval_X, fval_comm

def fn_val_L(Lx, L, X, Lambda, Phi, alpha, theta):
    n = len(Lx)
    val_L = [alpha[0][i]*np.linalg.norm(L-Lambda[0][i]-Lx[i])**2 for i in range(n)]
    val_X = [alpha[1][i]*np.linalg.norm(X[i]+Lambda[1][i]-Lx[i])**2 for i in range(n)]

    covs = [t2m(Lx[i],i)@t2m(X[i],i).transpose() for i in range(n)]
    val_comm = [theta[i]*np.linalg.norm(covs[i] @ Phi[i] - Phi[i] @ covs[i])**2 for i in range(n)]

    f_val = sum(val_comm) + sum(val_X) + sum(val_L)
    return f_val, val_L, val_X, val_comm

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

