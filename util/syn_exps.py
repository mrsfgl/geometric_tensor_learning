
import sys
import os
import argparse

import numpy as np
from scipy.io import savemat
from omegaconf import OmegaConf
import project_path
from util.generate_graphs import generate_graphs
from util.generate_data import generate_smooth_stationary_data
from util.contaminate_data import contaminate_signal
from util.horpca import horpca
from util.hosvd import hosvd
from util.geoTL import geoTL
from util.measure_error import measure_error
from util.srpg import gmlsvd
from util.srpg import srpg_nnfold_modified as nnfold
from util.srpg import srpg_td_a as tda

parser = argparse.ArgumentParser()
parser.add_argument("--exp_config", dest="config_file",
                    default='configs/synthetic_conf.yaml',
                    help="Configuration file for the experiments.")

args = parser.parse_args()


def grid_search(params_noise, data_params, param_list):
    '''Pipeline for experiments on synthetic data.'''

    noise_type = params_noise.noise_type
    noise_list = params_noise.SNR
    len_sizes = len(data_params.size_list)
    len_dens = len(data_params.d_list)
    len_noise = len(noise_list)
    len_gamma = len(param_list.geoTL.gamma)
    len_theta = len(param_list.geoTL.theta)

    shape_data_par = (len_sizes, len_dens, len_noise)
    shape_geoTL_par = (len_sizes, len_dens, len_noise, len_gamma, len_theta)
    err_orig = np.zeros(shape_data_par)
    err_geoTL = np.zeros(shape_geoTL_par)
    err_horpca = np.zeros(shape_data_par)
    err_hosvd = np.zeros(shape_data_par)
    err_gmlsvd = np.zeros(shape_data_par)
    err_nnfold = np.zeros(shape_data_par)
    err_tda = np.zeros(shape_data_par)

    for i_sz in range(len_sizes):
        sizes = data_params.size_list[i_sz]
        n = len(sizes)
        ranks = [3*np.int16(np.log(sz)) for sz in sizes]

        for i_d in range(len_dens):
            d = data_params.d_list[i_d]
            Phi = generate_graphs(sizes, d)

            X_smooth, _ = generate_smooth_stationary_data(Phi)

            for i_n in range(len_noise):
                noise_level = noise_list[i_n]
                Y = contaminate_signal(X_smooth, noise_level,
                                       noise_type=noise_type)
                std_Y = np.std(Y)
                err_orig[i_sz, i_d, i_n] = measure_error(X_smooth, Y.data)

                for i_gam in range(len_gamma):
                    curr_gamma = np.ones(n)*param_list.geoTL.gamma[i_gam]

                    for i_theta in range(len_theta):
                        curr_theta = np.ones(n)*param_list.geoTL.theta[i_theta]

                        L_geotl, _, _ = geoTL(Y, Phi,
                                              gamma=curr_gamma.copy(),
                                              theta=curr_theta.copy(),
                                              max_iter=400,
                                              err_tol=1e-2,
                                              d=d)

                        err_geoTL[i_sz, i_d, i_n, i_gam, i_theta
                                  ] = measure_error(X_smooth, L_geotl)

                alpha = [10**-2 for i in range(n)]
                L_horpca, _, _, _ = horpca(Y, alpha=alpha, max_iter=500,
                                           err_tol=1e-3)
                err_horpca[i_sz, i_d, i_n] = measure_error(X_smooth, L_horpca)

                L_hosvd = hosvd(Y.data, ranks, max_iter=10, err_tol=1e-2)[0]
                err_hosvd[i_sz, i_d, i_n] = measure_error(X_smooth, L_hosvd)

                L_gmlsvd = gmlsvd(Y, Phi, ranks)
                err_gmlsvd[i_sz, i_d, i_n] = measure_error(X_smooth, L_gmlsvd)

                L_tda, _ = tda(Y, Phi, ranks)
                err_tda[i_sz, i_d, i_n] = measure_error(X_smooth, L_tda)

                L_nnfold, _, _ = nnfold(
                    Y, Phi,
                    alpha=np.tile(0.01/np.sqrt(np.max(sizes)), n),
                    beta=np.tile(0.5/np.sqrt(np.max(sizes)), n),
                    max_iter=500,
                    err_tol=1e-2)
                err_nnfold[i_sz, i_d, i_n] = measure_error(X_smooth, L_nnfold)

                # Y_rep = [Y.data for i in range(n)]
                # gamma = [gamma[i] for i in range(n)]
                # theta = [theta[i] for i in range(n)]
                # y_sigma = [t2m(Y.data, i)@t2m(Y.data, i).transpose()
                #            for i in range(n)]
                # y_smooth_val[i_sz, i_d, i_n, :] =fn_val_G(Y_rep,Y.data,Phi,
                #                                           Lambda[0],alpha[0],
                #                                           gamma)[1]
                # comm_y[i_sz, i_d, i_n, :] = fnval_Sigma(y_sigma, Lx, X, Phi,
                #                                         Lambda[3], alpha[3],
                #                                         theta)[1]

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
    d = grid_search(params.noise, params.data, params.model)
    sys.stdout.write('Hit 2!\n')
    d['params'] = params
    sys.stdout.write('Hit 3!\n')
    savemat('experiments/{}.mat'.format(np.random.randint(1, 3000)), d)
