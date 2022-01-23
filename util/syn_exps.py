
import numpy as np
from util.generate_graphs import generate_graphs
from util.generate_data import generate_smooth_stationary_data
from util.contaminate_data import contaminate_signal
from util.horpca import horpca
from util.hosvd import hosvd
from util.geoTL import geoTL
from util.measure_error import measure_error


def grid_search(noise_list, param_list):
    '''Pipeline for experiments on synthetic data.'''

    len_sizes = len(param_list.size_list)
    len_dens = len(param_list.d_list)
    len_noise = len(noise_list)

    err_orig = np.zeros((len_sizes, len_dens, len_noise, 4))
    err_geoTL = np.zeros((len_sizes, len_dens, len_noise, 4))
    err_horpca = np.zeros((len_sizes, len_dens, len_noise, 4))
    err_hosvd = np.zeros((len_sizes, len_dens, len_noise, 4))
    for i_sz in range(len_sizes):
        sizes = param_list.size_list[i_sz]
        ranks = [np.round(np.log(sz)) for sz in sizes]
        for i_d in range(len_dens):
            d = param_list.d_list[i_d]
            Phi = generate_graphs(sizes, d)

            X_smooth, V = generate_smooth_stationary_data(Phi)

            for i_n in range(len_noise):
                noise_level = noise_list[i_n]
                Y = contaminate_signal(X_smooth, noise_level)

                L_geotl, _, _ = geoTL(Y, Phi, max_iter=500, err_tol=1e-2, d=d)
                L_horpca, _, _ = horpca(Y)
                L_hosvd = hosvd(Y, ranks)[0]

                err_orig[i_sz, i_d, i_n] = measure_error(X_smooth, Y.data)
                err_geoTL[i_sz, i_d, i_n] = measure_error(X_smooth, L_geotl)
                err_horpca[i_sz, i_d, i_n] = measure_error(X_smooth, L_horpca)
                err_hosvd[i_sz, i_d, i_n] = measure_error(X_smooth, L_hosvd)

                # Y_rep = [Y.data for i in range(n)]
                # gamma = [gamma[i] for i in range(n)]
                # theta = [theta[i] for i in range(n)]
                # y_sigma = [t2m(Y.data, i)@t2m(Y.data, i).transpose()
                #            for i in range(n)]
                # y_smooth_val[i_sz, i_d, i_n, :] = fn_val_G(Y_rep, Y.data, Phi,
                #                                            Lambda[0], alpha[0],
                #                                            gamma)[1]
                # comm_y[i_sz, i_d, i_n, :] = fnval_Sigma(y_sigma, Lx, X, Phi,
                #                                         Lambda[3], alpha[3],
                #                                         theta)[1]

    return err_orig, err_geoTL, err_horpca, err_hosvd
