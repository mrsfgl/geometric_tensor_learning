from util.t2m import t2m
from util.m2t import m2t
import numpy as np

def update_Lambda(Lambda, L, Lx, X, G, track_fval = False):
    """ Update function for dual variables $\Lambda$.  
    ! Needs to be checked.
    """
    n = len(Lx)
    Lambda[0] = [Lambda[0][i]+G[i]-L for i in range(n)]
    Lambda[1] = [Lambda[1][i]+Lx[i]-L for i in range(n)]
    Lambda[2] = [Lambda[2][i]+X[i]-Lx[i] for i in range(n)]

    fval = fn_val_Lambda(Lambda, L, Lx, G)[0] if track_fval else []

    return Lambda, fval


def fn_val_Lambda(Lambda, L, Lx, X, G):
    n = len(Lx)

    val_G = [np.linalg.norm(G[i]+Lambda[0][i]-L)**2 for i in range(n)]
    val_L = [np.linalg.norm(Lx[i]+Lambda[1][i]-L)**2 for i in range(n)]
    val_X = [np.linalg.norm(X[i]+Lambda[2][i]-Lx[i])**2 for i in range(n)]

    f_val = sum(val_G) + sum(val_L) + sum(val_X)
    return f_val, val_G, val_L, val_X