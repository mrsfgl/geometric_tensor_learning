
import numpy as np
from omegaconf import OmegaConf
from util.generate_graphs import generate_graphs
from util.generate_data import generate_smooth_stationary_data
from util.contaminate_data import contaminate_signal
from util.horpca import horpca
from util.hosvd import hosvd
from util.geoTL import geoTL
from util.measure_error import measure_error
from util.srpg import gmlsvd
from util.srpg import srpg_nnfold_modified as nnfold


def grid_search(noise_list, data_params, param_list):
    '''Pipeline for experiments on synthetic data.'''

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

    for i_sz in range(len_sizes):
        sizes = data_params.size_list[i_sz]
        n = len(sizes)
        ranks = [np.int16(np.log(sz)) for sz in sizes]

        for i_d in range(len_dens):
            d = data_params.d_list[i_d]
            Phi = generate_graphs(sizes, d)

            X_smooth, V = generate_smooth_stationary_data(Phi)

            for i_n in range(len_noise):
                noise_level = noise_list[i_n]
                Y = contaminate_signal(X_smooth, noise_level)

                for i_gam in range(len_gamma):
                    curr_gamma = np.ones(n)*param_list.geoTL.gamma[i_gam]

                    for i_theta in range(len_theta):
                        curr_theta = np.ones(n)*param_list.geoTL.theta[i_theta]

                        L_geotl, _, _ = geoTL(Y, Phi,
                                              gamma=curr_gamma,
                                              theta=curr_theta,
                                              max_iter=400,
                                              err_tol=1e-2,
                                              d=d)

                        err_geoTL[i_sz, i_d, i_n, i_gam, i_theta
                                  ] = measure_error(X_smooth, L_geotl)

                L_horpca, _, _, _ = horpca(Y)
                L_hosvd = hosvd(Y, ranks, max_iter=10, err_tol=1e-2)[0]
                L_gmlsvd = gmlsvd(Y, ranks)
                L_nnfold = nnfold(Y, ranks)

                err_orig[i_sz, i_d, i_n] = measure_error(X_smooth, Y.data)
                err_horpca[i_sz, i_d, i_n] = measure_error(X_smooth, L_horpca)
                err_hosvd[i_sz, i_d, i_n] = measure_error(X_smooth, L_hosvd)
                err_gmlsvd[i_sz, i_d, i_n] = measure_error(X_smooth, L_gmlsvd)
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

    return err_orig, err_geoTL, err_horpca, err_hosvd, err_gmlsvd, err_nnfold


if __name__ == "__main__":
    params = OmegaConf('configs/synthetic_conf.yaml')
    e_o, e_g, e_ho, e_hs = grid_search(params.noise.SNR, params.data, params.model)
