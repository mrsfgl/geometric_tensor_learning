from util.t2m import t2m
import numpy as np
from numpy.linalg import norm


def update_Lambda(Lambda, L, Lx, X, G, Sigma, track_fval=False):
    """ Update function for dual variables Lambda.
    ! Needs to be checked.
    """
    n = len(Lx)

    temp = [L-G[i] for i in range(n)]
    Lambda[0], val_lam_change = up_Lam(Lambda[0], temp)

    temp = [L-Lx[i] for i in range(n)]
    Lambda[1], temp_change = up_Lam(Lambda[1], temp)
    val_lam_change = val_lam_change+temp_change

    temp = [Lx[i]-X[i] for i in range(n)]
    Lambda[2], temp_change = up_Lam(Lambda[2], temp)
    val_lam_change = val_lam_change+temp_change

    cov = [t2m(Lx[i], i)@t2m(X[i], i).transpose() for i in range(n)]
    temp = [cov[i]-Sigma[i] for i in range(n)]
    Lambda[3], temp_change = up_Lam(Lambda[3], temp)
    val_lam_change = val_lam_change+temp_change

    fval = fn_val_Lambda(Lambda, L, Lx, G, Sigma, cov)[0] if track_fval else []

    return Lambda, fval, sum(val_lam_change)


def up_Lam(Lambda, X):

    n = len(Lambda)
    old = [Lambda[i].copy() for i in range(n)]
    Lambda = [Lambda[i]-X[i] for i in range(n)]
    val_change = np.array([norm(Lambda[i]-old[i]) for i in range(n)])

    return Lambda, val_change


def fn_val_Lambda(Lambda, L, Lx, X, G, Sigma, covs):
    n = len(Lx)

    val_G = [norm(G[i]+Lambda[0][i]-L)**2 for i in range(n)]
    val_L = [norm(Lx[i]+Lambda[1][i]-L)**2 for i in range(n)]
    val_X = [norm(X[i]+Lambda[2][i]-Lx[i])**2 for i in range(n)]
    val_Sig = [norm(Sigma[i]+Lambda[3][i]-covs[i])**2 for i in range(n)]

    f_val = sum(val_G) + sum(val_L) + sum(val_X)
    return f_val, val_G, val_L, val_X, val_Sig
