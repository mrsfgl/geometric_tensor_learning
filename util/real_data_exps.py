
import sys
import os
import argparse

import numpy as np
from scipy.io import savemat, loadmat
from omegaconf import OmegaConf
import project_path
from sklearn.neighbors import kneighbors_graph

from util.contaminate_data import contaminate_signal
from util.t2m import t2m
from util.horpca import horpca
from util.hosvd import hosvd
from util.geoTL import geoTL
from util.measure_error import measure_error
from util.srpg import gmlsvd
from util.srpg import srpg_nnfold_modified as nnfold
from util.srpg import srpg_td_a as tda

parser = argparse.ArgumentParser()
parser.add_argument("--exp_config", dest="config_file", default='configs/real_conf.yaml',
                    help="Graph model to use to generate the data.")

args = parser.parse_args()


def grid_search(params_noise, param_list):
    '''Pipeline for experiments on synthetic data.'''

    noise_type = params_noise.noise_type
    noise_list = params_noise.SNR
    len_noise = len(noise_list)
    len_gamma = len(param_list.geoTL.gamma)
    len_theta = len(param_list.geoTL.theta)

    md = loadmat('data/coil_small.mat')
    X = md['Data']
    sizes = X.shape
    n = len(sizes)
    Phi = []
    for i in range(n):
        A = kneighbors_graph(t2m(X,i), n_neighbors=int(np.log(sizes[i]),)).todense()
        A = (A+A.T)/2
        Phi.append(np.diag(np.array(np.sum(A,1)).squeeze())-A)

    shape_data_par = (len_noise)
    shape_geoTL_par = (len_noise, len_gamma, len_theta)
    err_orig = np.zeros(shape_data_par)
    err_geoTL = np.zeros(shape_geoTL_par)
    err_horpca = np.zeros(shape_data_par)
    err_hosvd = np.zeros(shape_data_par)
    err_gmlsvd = np.zeros(shape_data_par)
    err_nnfold = np.zeros(shape_data_par)
    err_tda = np.zeros(shape_data_par)

    sizes = X.shape
    ranks = [3*np.int16(np.log(sz)) for sz in sizes]
    n = len(sizes)
    for i_n in range(len_noise):
        noise_level = noise_list[i_n]
        Y = contaminate_signal(X, noise_level,
                                noise_type=noise_type)
        err_orig[i_n] = measure_error(X, Y.data)

        for i_gam in range(len_gamma):
            curr_gamma = np.ones(n)*param_list.geoTL.gamma[i_gam]

            for i_theta in range(len_theta):
                curr_theta = np.ones(n)*param_list.geoTL.theta[i_theta]

                L_geotl, _, _ = geoTL(Y, Phi,
                                        gamma=curr_gamma.copy(),
                                        theta=curr_theta.copy(),
                                        max_iter=400,
                                        err_tol=1e-2)

                err_geoTL[i_n, i_gam, i_theta
                            ] = measure_error(X, L_geotl)

        alpha = [10**-3 for i in range(n)]
        L_horpca, _, _, _ = horpca(Y, alpha=alpha, max_iter=400)
        err_horpca[i_n] = measure_error(X, L_horpca)

        L_hosvd = hosvd(Y.data, ranks, max_iter=10, err_tol=1e-2)[0]
        err_hosvd[i_n] = measure_error(X, L_hosvd)

        L_gmlsvd = gmlsvd(Y, Phi, ranks)
        err_gmlsvd[i_n] = measure_error(X, L_gmlsvd)

        L_tda, _ = tda(Y, Phi, ranks)
        err_tda[i_n] = measure_error(X, L_tda)

        L_nnfold, _, _ = nnfold(
            Y, Phi,
            alpha=np.tile(0.01/np.sqrt(np.max(sizes)), n),
            beta=np.tile(0.5/np.sqrt(np.max(sizes)), n),
            max_iter=500,
            err_tol=1e-2)
        err_nnfold[i_n] = measure_error(X, L_nnfold)


    d = {
        'Original': err_orig,
        'geoTL': err_geoTL,
        'HoRPCA': err_horpca,
        'HoSVD': err_hosvd,
        'GMLSVD': err_gmlsvd,
        'NNFOLD': err_nnfold,
        'TDA': err_tda
        }
    return d


confs = args.config_file
if __name__ == "__main__":
    params = OmegaConf.load(confs)
    sys.stdout.write('Hit 1!\n')
    d = grid_search(params.noise, params.model)
    sys.stdout.write('Hit 2!\n')
    savemat('experiments/real_experiments/{}.mat'.format(np.random.randint(1, 3000)) , d)
