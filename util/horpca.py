import numpy as np
from numpy.linalg import norm
from util.soft_hosvd import soft_hosvd


def horpca(Y, psi=1, beta=[], alpha=[], max_iter=100, err_tol=10**-5,
           verbose=False):
    ''' Higher Order Robust PCA
    Runs the ADMM algorithm for HoRPCA

    The original paper for this algorithm is by:
    Goldfarb, Donald, and Zhiwei Qin. "Robust low-rank tensor recovery: Models
        and algorithms." SIAM Journal on Matrix Analysis and Applications
        35.1 (2014): 225-253.
    '''
    sizes = Y.shape
    n = len(sizes)
    L, S, Lx, Lambda, psi, beta, alpha = init(Y, psi, beta, alpha)

    t_norm = norm(Y)
    iter = 0
    err = np.inf
    obj_val = []
    terms = []
    nuc_norm = [0 for _ in range(n)]
    val, term = compute_obj(Y, L, Lx, S, Lambda, psi, beta, alpha, nuc_norm)
    while iter != max_iter and err > err_tol:
        # L Update
        L = update_L(Y, S, Lx, Lambda, alpha)
        # val, term = compute_obj(Y,L,Lx,S,Lambda,psi,beta,alpha,nuc_norm)

        # Lx Update
        Lx, nuc_norm = soft_hosvd(L, Lambda[1], psi, 1/alpha[1])
        # val, term = compute_obj(Y,L,Lx,S,Lambda,psi,beta,alpha,nuc_norm)

        # S Update
        S_old = S.copy()
        S = soft_threshold(Y-L-Lambda[0], beta/alpha[0])
        # val, term = compute_obj(Y,L,Lx,S,Lambda,psi,beta,alpha,nuc_norm)

        # Dual variable Updates
        Lambda, dual_err = update_Lambda(Y, Lambda, n, L, S, Lx, t_norm)

        # Objective and error calculations
        # val, term = compute_obj(Y,L,Lx,S,Lambda,psi,beta,alpha,nuc_norm)
        terms.append(term)
        obj_val.append(val)
        err = max(norm(S-S_old)/(np.finfo(float).eps+norm(S_old)), dual_err)
        iter += 1
        if verbose:
            if err <= err_tol:
                print('Converged!')
            elif iter == max_iter:
                print('Max Iter')

    return L, obj_val, np.array(terms)


def compute_obj(Y, L, Lx, S, Lambda, psi, beta, alpha, nuc_norm):
    ''' Computes the objective function for HoRPCA. '''
    n = len(Lambda)

    term = [alpha[0]/2*norm(Y-L-S-Lambda[0])**2, 0, 0, 0]
    for i in range(n):
        term[1] += nuc_norm[i]
        term[3] += alpha[1]/2*norm(L-Lx[i]-Lambda[1][i])**2

    term[2] = beta*norm(S.ravel(), ord=1)
    val = sum(term)
    return val, term


def soft_threshold(T, sigma):
    ''' Soft thresholding of the tensor T with parameter sigma.
    '''
    X = np.clip(np.abs(T)-sigma, 0, None)

    return X*np.sign(T)


def init(Y, psi, beta, alpha):
    ''' Initialize variables.'''
    sizes = Y.shape
    n = len(sizes)

    # Initialize parameters using recommended choices in the paper.
    psi = [psi for i in range(n)] if len(psi) == 1 else psi
    beta = np.sqrt(max(sizes)) if not beta else beta
    std_Y = np.std(Y.ravel())
    alpha = [1/5*std_Y for i in range(n)] if len(alpha) == 0 else alpha

    # Initialize tensor variables.
    Lx = [np.zeros(sizes) for i in range(n)]
    L = np.zeros(sizes)
    S = np.zeros(sizes)
    Lambda = [
        np.zeros(sizes),
        [np.zeros(sizes) for i in range(n)]
    ]
    return L, S, Lx, Lambda, psi, beta, alpha


def update_L(Y, S, Lx, Lambda, alpha):
    '''Updates variable L.'''
    sizes = Y.shape
    n = len(sizes)
    L = np.zeros(sizes)
    temp1 = alpha[0]*(Y-S-Lambda[0])
    temp2 = alpha[1]*sum(Lx[i] + Lambda[1][i] for i in range(n))
    L[~Y.mask] = (temp1[~Y.mask] + temp2[~Y.mask])/(alpha[0]+n*alpha[1])
    L[Y.mask] = temp2[Y.mask]/(n*alpha[1])
    return L


def update_Lambda(Y, Lambda, n, L, S, Lx, t_norm):
    Lambda[0] = Lambda[0]+L+S-Y
    dual_err = 0
    for i in range(n):
        lambda_update = L-Lx[i]
        dual_err += norm(lambda_update)**2
        Lambda[1][i] = Lambda[1][i] - lambda_update

    dual_err = np.sqrt(dual_err/n)/t_norm
    return Lambda, dual_err
