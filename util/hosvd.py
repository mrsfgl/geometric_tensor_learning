
import numpy as np
from util.t2m import t2m
from util.m2t import m2t

def hosvd(X, rank):
    ''' Higher Order Singular Value Decomposition.
    Takes the HOSVD of tensor input.

    Parameters:
        X: Data tensor
        rank: Aimed ranks for the resulting tensor.

    Outputs:
        Y: Tensor with ranks `rank`at each mode unfolding.
        U_list: List of factor matrices of HOSVD.
    '''
    sizes = X.shape
    n = len(sizes)
    Y = X
    U_list = []
    for i in range(n):
        U, S, V = np.linalg.svd(t2m(Y, i), full_matrices = False)
        U_list.append(U)
        Y = m2t(np.dot(U[:,:rank[i]]*S[:rank[i]], V[:rank[i],:]), sizes, i)

    return Y, U_list