
import numpy as np
from util.t2m import t2m
from numpy.linalg import norm


def update_Sigma(Sigma, Lx, X, Phi, Lambda, alpha, theta, track_fval=False):
    ''' The update function for auxiliary variable Sigma.
    This function tries to estimate a covariance commutative with Sigma.
    '''

    n = len(Sigma)

    for i in range(n):
        c = alpha[i]/theta[i]
        A = np.ravel(t2m(Lx[i], i)@t2m(X[i], i).transpose() -
                     Lambda[i], order='F')

        eye = [np.identity(Sigma[i].shape[0]), np.identity(Sigma[i].shape[1]),
               np.identity(Sigma[i].size)]
        Psq = Phi[i]@Phi[i]
        invmat = (np.kron(Psq, eye[0])-2*np.kron(Phi[i], Phi[i]) +
                  np.kron(eye[1], Psq) + c*eye[2])
        # vecSig = np.ravel(Sigma[i], order = 'F')
        # theta[i]*vecSig@np.tensordot(invmat, vecSig, axes=([1],[0])) -
        #      2*alpha[i]* sum(vecSig * A) + (alpha[i]**2)*sum(A**2)
        invmat = np.linalg.inv(invmat)
        vecSig = np.tensordot(invmat, c*A, axes=([1], [0]))
        Sigma[i] = vecSig.reshape(Phi[i].shape).transpose()

    (fval, fval_sig, fval_comm
     ) = fn_val(Sigma, Lx, X, Phi, Lambda, alpha, theta) if track_fval else [[],[],[]]

    return Sigma, fval, fval_sig, fval_comm


def fn_val(Sigma, Lx, X, Phi, Lambda, alpha, theta):
    n = len(Sigma)

    covs = [t2m(Lx[i], i)@t2m(X[i], i).transpose() for i in range(n)]
    val_sig = [alpha[i]*norm(covs[i] - Sigma[i] - Lambda[i])**2
               for i in range(n)]

    val_comm = [theta[i]*norm(Sigma[i]@Phi[i]-Phi[i]@Sigma[i])**2
                for i in range(n)]

    f_val = sum(val_sig) + sum(val_comm)
    return f_val, val_comm, val_sig
