
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def plot_synthetic(params, filename='outputs.mat'):

    md = loadmat(filename)
    len_theta = len(params.model.geoTL.theta)
    len_gamma = len(params.model.geoTL.gamma)
    for i_gam in range(len_gamma):
        for i_theta in range(len_theta):
            for key, value in md.items():
                if key in ['__header__', '__version__', '__globals__', 'params']:
                    continue
                elif key == 'geoTL':
                    plt.plot(params.noise.SNR,
                             value[0,0,:,i_gam,i_theta].squeeze(),
                             label=key)
                else:
                    plt.plot(params.noise.SNR, value[0,0,:].squeeze(), label=key)
            plt.xlabel('SNR values of noise.')
            plt.ylabel('RSE values of the output')
            plt.legend()
            plt.show()
            fname = 'synthetic_results/gamma_ind_{}_theta_ind_{}.png'.format(i_gam, i_theta)
            plt.savefig(fname)