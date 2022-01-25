
'''
Implementations of algorithms from the paper:
Varma and Kovačević, 'SMOOTH SIGNAL RECOVERY ON PRODUCT GRAPHS', ICASSP 2019
'''
import numpy as np
from numpy.linalg import norm
from util.merge_Tucker import merge_Tucker
from util.soft_hosvd import soft_hosvd
from util.soft_hosvd import soft_moden
from util.t2m import t2m
from util.m2t import m2t
from util.hosvd import hosvd


def srpg_nnfold_modified(
        Y, Phi,
        alpha=np.ones(4), beta=np.ones(4),
        mu=np.tile(0.01, (2, 4)),
        max_iter=300, err_tol=1e-3, verbose=False
        ):
    ''' Implementation of Reconstruction via the Nuclear Norm of Unfoldings.

    Parameters:
        Y: numpy.ndarray
            Corrupted input tensor.

        Phi: list of matrices
            Graph Laplacian list for all modes.

        alpha: list of doubles
            Parameter for mode-n smoothness

        beta: list of doubles
            Parameter for nuclear norm

        mu: list of list of doubles
            Lagrange multiplier

    Outputs:
        X: numpy.ndarray
            Output tensor.
    '''
    sizes = Y.shape
    n = len(sizes)

    X = np.zeros(sizes)
    Z = [np.zeros(sizes) for _ in range(n)]
    V = [np.zeros(sizes) for _ in range(n)]
    W = [
         [np.zeros(sizes) for _ in range(n)],
         [np.zeros(sizes) for _ in range(n)]
         ]

    # ADMM loop
    iter = 0
    change_w = np.inf
    obj_val = []
    lam_val = []
    V_inv = [np.linalg.inv(alpha[i]*Phi[i] + mu[1][i]*np.identity(sizes[i]))
             for i in range(n)]
    while iter < max_iter and change_w > err_tol:

        # X update
        X[~Y.mask] = Y[~Y.mask]/(1+sum(mu[0])+sum(mu[1]))
        for i in range(n):
            term = mu[0][i]*(W[0][i]+Z[i])
            term = term + mu[1][i]*(W[1][i] + V[i])
            X = X + term/(1+sum(mu[0])+sum(mu[1]))

        # Z update
        Z, _ = soft_hosvd(X, W[0], beta, 1/mu[0][0])

        # V update
        V = [m2t(mu[1][i]*V_inv[i]*t2m(X-W[1][i], i), sizes, i)
             for i in range(n)]

        # W update
        W_old = W.copy()
        W[0] = [W[0][i]-X+Z[i] for i in range(n)]
        W[1] = [W[1][i]-X+V[i] for i in range(n)]
        change_w = np.sqrt(
            sum(
                mu[0][i]*norm(W[0][i]-W_old[0][i])**2 +
                mu[1][i]*norm(W[1][i]-W_old[1][i])**2
                for i in range(n)
                )/np.sum(mu)
            )

        lam_val.append(change_w)
        obj_val.append(compute_obj_nnfold_modified(
            Y, Phi, X, Z, V, W, alpha, beta, mu
        )[0])
        iter += 1

    return X, obj_val, lam_val


def compute_obj_nnfold_modified(Y, Phi, X, Z, V, W, alpha, beta, mu):
    sizes = Y.shape
    n = len(sizes)
    val_Y = norm(Y-X)**2
    Vmat = [t2m(V[i], i) for i in range(n)]
    Zmat = [t2m(Z[i], i) for i in range(n)]
    val_smooth = [alpha[i]*np.trace(Vmat[i].transpose() @ Phi[i] @ Vmat[i])
                  for i in range(n)]
    val_nuc = [beta[i]*norm(Zmat[i], ord='nuc') for i in range(n)]
    val_lag = np.array([
        [mu[0][i]*norm(X-Z[i]-W[0][i])**2 for i in range(n)],
        [mu[1][i]*norm(X-V[i]-W[1][i])**2 for i in range(n)]
        ])
    fn_val = val_Y+sum(val_smooth)+sum(val_nuc)+np.sum(val_lag)
    return fn_val, val_Y, val_smooth, val_nuc, val_lag


def srpg_nnfold(
        Y, Phi,
        alpha=[1, 1, 1, 1], beta=[1, 1, 1, 1], mu=np.ones(4)*0.01,
        max_iter=300, err_tol=1e-3, verbose=False
        ):
    ''' Implementation of Reconstruction via the Nuclear Norm of Unfoldings.

    Parameters:
        Y: numpy.ndarray
            Corrupted input tensor.

        Phi: list of matrices
            Graph Laplacian list for all modes.

        alpha: double
            Parameter for mode-n smoothness

        beta: double
            Parameter for nuclear norm

        mu: double
            Lagrange multiplier

    Outputs:
        X: numpy.ndarray
            Output tensor.
    '''
    sizes = Y.shape
    n = len(sizes)

    X = np.zeros(sizes)
    Z = [np.zeros(sizes) for _ in range(n)]
    W = [np.zeros(sizes) for _ in range(n)]

    iter = 0
    obj_val = []
    lam_val = []
    while True:

        # X update
        X[~Y.mask] = Y[~Y.mask]/(1+sum(mu))
        for i in range(n):
            term = W[i]+Z[i]
            X = X + term/(1+sum(mu))

        # Z update
        Z = update_Z(X, W, Phi, alpha, beta, mu)

        # W update
        W_old = W.copy()
        W = [W[i]-X+Z[i] for i in range(n)]
        change_w = np.sqrt(sum(norm(W[i]-W_old[i])**2 for i in range(n)))

        lam_val.append(change_w)
        obj_val.append(compute_obj_nnfold(
            Y, Phi, X, Z, W, alpha, beta, mu
        )[0])
        iter += 1
        if iter == max_iter:
            break

        if change_w < err_tol:
            break

    return X, obj_val, lam_val


def update_Z(X, W, Phi, alpha, beta, mu):
    sizes = X.shape
    n = len(sizes)
    Z = [np.zeros(sizes) for _ in range(n)]

    for i in range(n):
        lag_term = Z[i]-X+W[i]
        smth_term = Phi[i]@t2m(Z[i], i)
        update_Z = t2m(mu[i]*lag_term, i)+alpha[i]*smth_term
        t = backtrack(
            t2m(Z[i], i), t2m(X-W[i], i), update_Z, Phi[i],
            alpha[i], beta[i], mu[i]
            )
        Z[i] = soft_moden(Z[i]-t*m2t(update_Z, sizes, i), t*beta[i], i)[0]

    return Z


def backtrack(Z, X, grad, Phi, alpha, beta, mu, zeta=0.5):
    def f(x):
        return (
                beta*norm(x, ord='nuc') +
                alpha*np.trace(x.transpose()@Phi@x) +
                mu*norm(X-x)**2
                )

    t = 1
    while f(Z+t*grad)-f(Z) >= t*norm(grad)**2/2:
        t *= zeta
    return t


def compute_obj_nnfold(Y, Phi, X, Z, W, alpha, beta, mu):
    sizes = Y.shape
    n = len(sizes)
    val_Y = norm(Y-X)**2
    Zmat = [t2m(Z[i], i) for i in range(n)]
    val_smooth = [alpha[i]*np.trace(Zmat[i].transpose() @ Phi[i] @ Zmat[i])
                  for i in range(n)]
    val_nuc = [beta[i]*norm(Zmat[i], ord='nuc') for i in range(n)]
    val_lag = [mu[i]*norm(X-Z[i]-W[i])**2 for i in range(n)]
    fn_val = val_Y+sum(val_smooth)+sum(val_nuc)+sum(val_lag)
    return fn_val, val_Y, val_smooth, val_nuc, val_lag


def srpg_td_a(Y, Phi, ranks, lamda=1, gamma=1, max_iter=50, verbose=False):
    ''' Implementation of Tucker Decomposition via Alternating Least
    Squares

    Parameters:
        Y: numpy.ndarray
            Corrupted input tensor.

        Phi: list of matrices
            Graph Laplacian list for all modes.

        ranks: list of integers
            Target ranks.

        lamda: double
            Parameter for mode-n smoothness

        gamma: double
            Parameters for product graph smoothness

    Outputs:
        X: numpy.ndarray
            Output tensor.
    '''

    sizes = np.array(Y.shape)
    n = len(sizes)

    F = [np.eye(sz) for sz in sizes]
    D = [np.diag(1+np.diag(phi)) for phi in Phi]
    A = [phi-np.diag(np.diag(phi)-1) for phi in Phi]

    obj_val = []
    iter = 0
    while True:
        for i in range(n):
            ind = np.setdiff1d(np.arange(n), i)
            F_curr = [F[j] for j in ind]
            M = merge_Tucker(Y.data, F_curr, ind, transpose=True)
            M = t2m(M, i)
            u = np.prod([np.trace(F[j].transpose()@D[j]@F[j]) for j in ind])
            v = np.prod([np.trace(F[j].transpose()@A[j]@F[j]) for j in ind])
            H = M@M.transpose()-(lamda+gamma*u)*D[i]+(lamda+gamma*v)*A[i]
            U, S, _ = np.linalg.svd(H)
            ind = np.argsort(S)[::-1]
            F[i] = U[:, ind[:ranks[i]]]

        iter += 1
        if verbose:
            obj_val.append(compute_obj_tda(Y, Phi, F, lamda, gamma)[0])
        if iter == max_iter:
            break

    G = merge_Tucker(Y.data, F, np.arange(n), transpose=True)
    return merge_Tucker(G, F, np.arange(n)), obj_val


def compute_obj_tda(Y, Phi, F, lamda, gamma):

    sizes = Y.shape
    n = len(sizes)
    Fsq = [f@f.transpose() for f in F]
    X = merge_Tucker(Y.data, Fsq, np.arange(n), transpose=True)
    val_y = norm(Y.data-X)**2
    val_g = [lamda*np.trace(F[i].transpose()@Phi[i]@F[i])
             for i in range(n)]
    val_h = [gamma*np.trace(F[i].transpose()@Phi[i]@F[i])
             for i in range(n)]

    fn_val = val_y+sum(val_h)+sum(val_g)
    return fn_val, val_y, val_g, val_h


def gmlsvd(Y, Phi, ranks, ranks_core=[]):
    ''' Implementation of Tucker Decomposition via Synthesis or
    Graph Multilinear SVD from:
    Shahid, Nauman, Francesco Grassi, and Pierre Vandergheynst,
    "Multilinear low-rank tensors on graphs & applications.", arXiv:1611.04835

    Parameters:
        Y: numpy.ndarray
            Corrupted input tensor.

        Phi: list of matrices
            Graph Laplacian list for all modes.

        ranks: list of integers
            Target ranks.

    Outputs:
        X: numpy.ndarray
            Output tensor.
    '''

    no_hosvd = not ranks_core
    sizes = np.array(Y.shape)
    n = len(sizes)

    P = []
    for i in range(n):
        U, S, _ = np.linalg.svd(Phi[i], hermitian=True)
        ind = np.argsort(S)[::-1]
        P.append(U[:, ind[:ranks[i]]])

    dims = np.arange(n)
    X = merge_Tucker(Y.data, P, dims, transpose=True)
    if no_hosvd:
        return merge_Tucker(X, P, dims)

    R, A, _, _ = hosvd(X, ranks_core, max_iter=50, err_tol=1e-2)
    return merge_Tucker(R, P, dims)
