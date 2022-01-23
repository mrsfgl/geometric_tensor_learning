
import numpy as np


def contaminate_signal(X, target_SNR=10, missing_ratio=0):
    ''' Contaminates data with AWGN and random missing elements.

    Parameters:
        X: np.array(), double
            Original data tensor.

        target_SNR: double.
            Target SNR in dB.

        missing_ratio: double,
            Ratio of missing elements to tensor size. Should be in [0,1].

    Outputs:
        Y: Masked Array. np.ma()
            Noisy tensor with missing elements.
    '''
    # Generate noise
    sizes = X.shape
    signal_power = np.linalg.norm(X)**2/X.size
    signal_dB = 10 * np.log10(signal_power)
    noise_db = signal_dB - target_SNR
    noise_power = 10 ** (noise_db / 10)
    noise = np.sqrt(noise_power)*np.random.standard_normal(sizes)

    # Create mask
    vec_mask = np.random.uniform(size=X.size)-missing_ratio < 0
    mask = vec_mask.reshape(sizes)
    return np.ma.array(X+noise, mask=mask)
