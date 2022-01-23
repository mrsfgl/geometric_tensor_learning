
import numpy as np
from util.t2m import t2m


def fn_val_G(G, L, Phi, Lambda, alpha, gamma):
    n = len(G)
    val_smooth = [gamma[i]*np.trace(t2m(G[i], i).transpose()
                  @ Phi[i] @ t2m(G[i], i)) for i in range(n)]
    val_Lag = [alpha[i]*np.linalg.norm(L-G[i]-Lambda[i])**2 for i in range(n)]
    fn_val = sum(val_smooth) + sum(val_Lag)
    return fn_val, val_smooth, val_Lag


def fn_val_L(L, Y, Lx, G, Lambda, alpha):
    n = len(L.shape)
    val_Y = np.linalg.norm(Y[~Y.mask]-L[~Y.mask])**2
    val_Lag1 = [alpha[0][i]*np.linalg.norm(L-G[i]-Lambda[0][i])**2
                for i in range(n)]
    val_Lag2 = [alpha[1][i]*np.linalg.norm(L-Lx[i]-Lambda[1][i])**2
                for i in range(n)]
    fn_val = val_Y + sum(val_Lag1) + sum(val_Lag2)
    return fn_val, val_Y, val_Lag1, val_Lag2
