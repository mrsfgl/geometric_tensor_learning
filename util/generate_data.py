import numpy as np
from util.merge_Tucker import merge_Tucker

def generate_syn_data(dim, ranks):
    '''Generates synthetic tensor data with dimensions `dim` and ranks `ranks`.
    
    Parameters:
    
        dim: Dimensions of the tensor

        ranks: Ranks of the tensor

    Outputs:
        
        T: Tensor of order `len(dim)`.

    '''
    n = len(dim)
    C = np.random.standard_normal(ranks)
    U = [np.linalg.svd(
        np.random.standard_normal((dim[i], ranks[i])), 
        full_matrices = False
        )[0] for i in range(n)]
    return merge_Tucker(C, U, np.arange(n))