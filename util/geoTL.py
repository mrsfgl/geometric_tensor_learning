
import numpy as np

from util.t2m import t2m
from util.m2t import m2t
from util.update_L_nograd import update_L
from util.update_L_nograd import fn_val as fnval_L
from util.update_X_nograd import update_X
from util.update_X_nograd import fn_val as fnval_X
from util.update_Sigma import update_Sigma
from util.update_Lambda_nograd import update_Lambda
from util.update_Sigma import fn_val as fnval_Sigma
from util.fn_vals import fn_val_L
from util.fn_vals import fn_val_G


def geoTL(Y,
          Phi,
          gamma=[1, 1, 1, 1],
          theta=[1, 1, 1, 1],
          alpha=[],
          max_iter=500,
          err_tol=1e-3,
          d=np.tile(2, 4),
          verbose=False
          ):
    ''' Implementation of ADMM loop for GeoTL.

    Parameters:

        Y: numpy.ma.masked_array
            Input tensor. Contaminated with noise or missing data.

        G: list( numpy.ndarray )
            Graphs over all modes.

        gamma: list of floats
            Graph smoothness term weights for all modes of Y. Should be the
            same length with the number of dimensions of Y.

        theta: list of floats
            Commutativity term weights. should be the same lenght as gamma

        alpha: list of 4 lists
            Lagrange multipliers for dual variables.

        max_iter: int
            Number of maximum ADMM iterations. Default: 500

        err_tol: float
            Error tolerance for ADMM's convergence. Default: 1e-5

        verbose: bool
            Print the iteration status.


    Outputs:
        L: numpy.ndarray
            Graph smooth and stationary tensor extracted from Y.

        fval: list
            Function values at all iterations.

        lam_val: list
            Total norm of the changes in the dual variables at every iteration.

    '''

    sizes = Y.shape
    n = len(sizes)

    L, G_var, Lx, X, Sigma, Lambda, _, _, alpha = initialize_nograd(sizes)
    var_y = np.var(Y.data)
    gamma = np.array(gamma, dtype=np.double)
    theta = np.array(theta, dtype=np.double)
    
    for i in range(n):
        gamma[i] *= var_y*(sizes[i]**2)/(25*d[i]**2)
        theta[i] *= (sizes[i]**4)/((d[i]**4)*(10**7))

    iter = 0
    fval_tot = []
    lam_val = []
    G_inv = [np.linalg.inv(gamma[i]*Phi[i] + alpha[0][i]*np.identity(sizes[i]))
             for i in range(n)]
    # ADMM Loop
    while True:
        # L Update
        if verbose:
            prev_val = fn_val_L(L, Y, Lx, G_var, Lambda[:2], alpha[:2])[0]
        temp = np.zeros(sizes)
        for i in range(n):
            temp += alpha[0][i]*(G_var[i] + Lambda[0][i])
            temp += alpha[1][i]*(Lx[i] + Lambda[1][i])
        L = temp/(1+sum(alpha[0]) + sum(alpha[1]))
        L[~Y.mask] = L[~Y.mask] + Y[~Y.mask]/(sum(alpha[0])+sum(alpha[1])+1)
        if verbose:
            fval_data = fn_val_L(L, Y, Lx, G_var, Lambda[:2], alpha[:2])
            fval_data_change = fval_data[0]-prev_val
            prev_val = fn_val_G(G_var, L, Phi, Lambda[0], alpha[0], gamma)[0]

        # G Update
        G_var = [m2t(alpha[0][i]*G_inv[i]*t2m(L-Lambda[0][i], i), sizes, i)
                 for i in range(n)]
        if verbose:
            fval_G = fn_val_G(G_var, L, Phi, Lambda[0], alpha[0], gamma)
            fval_G_change = fval_G[0] - prev_val
            prev_val = fnval_L(Lx, L, X, Lambda[1:], Sigma, alpha[1:])[0]

        # Lx Update
        Lx, fval_L, fval_low, _, _ = update_L(Lx, L, X, Lambda[1:], Sigma,
                                              alpha[1:], track_fval=verbose)
        if verbose:
            fval_L_change = fval_L - prev_val
            prev_val = sum(fnval_X(X, Lx, Lambda[2:], Sigma, alpha[2:])[1])

        # X Update
        X, _, fval_X, _ = update_X(X, Lx, Lambda[2:], Sigma, alpha[2:],
                                   track_fval=verbose)
        if verbose:
            fval_X_change = sum(fval_X) - prev_val
            prev_val = fnval_Sigma(Sigma, Lx, X, Phi, Lambda[3], alpha[3],
                                   theta)

        # Sigma Update
        Sigma, fval_Sigma, _, _ = update_Sigma(Sigma, Lx, X, Phi, Lambda[3],
                                               alpha[3], theta,
                                               track_fval=verbose
                                               )
        if verbose:
            fval_Sigma_change = fval_Sigma - prev_val[0]

            fval_tot.append(fval_data[1] + fval_G[0] + sum(fval_low) +
                            sum(fval_X) + fval_Sigma
                            )

            print(('Objective function changes for L: {:.2e}, G: {:.2e},' +
                   ' Lx: {:.2e}, X: {:.2e}, Sigma: {:.2e}').format(
                fval_data_change,
                fval_G_change,
                fval_L_change,
                fval_X_change,
                fval_Sigma_change
            ))
            print('Total Objective Function Value at iter {}: {:.3e}'.format(
                                                                iter,
                                                                fval_tot[-1]
                                                                ))
        # Dual Update
        Lambda, _, norm_lambda = update_Lambda(Lambda, L, Lx, X, G_var, Sigma)
        lam_val.append(norm_lambda)
        if iter == max_iter:
            break
        if norm_lambda <= err_tol:
            break
        iter += 1

    return L, fval_tot, lam_val


def initialize_nograd(sizes):
    n = len(sizes)
    # Parameters
    alpha = [[10**-2.8 for i in range(n)],
             [10**-1 for i in range(n)],
             [10**-1 for i in range(n)],
             [10**-3 for i in range(n)]]
    theta = [10**-4.5 for i in range(n)]
    gamma = [10**-3.5 for i in range(n)]

    # Initializations
    L = np.zeros(sizes)
    G_var = [np.zeros(sizes) for i in range(n)]
    X = [np.zeros(sizes) for i in range(n)]
    Lx = [np.zeros(sizes) for i in range(n)]
    Sigma = []
    for i in range(n):
        temp = np.random.standard_normal(sizes[i]**2).reshape([sizes[i],
                                                               sizes[i]])
        Sigma.append(temp@temp.transpose())

    Lambda = [[np.zeros(sizes) for i in range(n)],
              [np.zeros(sizes) for i in range(n)],
              [np.zeros(sizes) for i in range(n)],
              [np.zeros([sizes[i], sizes[i]]) for i in range(n)]]

    return L, G_var, Lx, X, Sigma, Lambda, gamma, theta, alpha
