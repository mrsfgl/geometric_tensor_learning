import numpy as np
from util.t2m import t2m
from util.m2t import m2t

def merge_Tucker(C, U, dims):
    ''' Merges tensor and factor matrices in Tucker format.

    Parameters:
        C: Tensor to be merged. Should have the same dimensions with the number of columns in `U`.

        U: Factor matrices. I(n) x r(n)

        dims: Dimensions along which the merge will occur. Should have same or smaller length then `U`.

    Outputs:
        X: Merged tensor. Shape should be the same with row sizes of `U` along modes `dims`.

    '''
    n = len(dims)
    X = C.copy()
    sizes = list(X.shape)
    for i in range(n):
        sizes[dims[i]] = U[dims[i]].shape[0]
        X = m2t(U[dims[i]] @ t2m(X, dims[i]), sizes, dims[i])

    return X