
from numpy.linalg import norm


def measure_error(X, L):
    ''' Measures errors with RSE.

    Parameters:
        X: Ground truth tensor.

        L: Compared tensor.

    Outputs:
        rse: RSE value.

    '''

    return norm(X-L)/norm(X)
