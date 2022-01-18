from msilib.schema import Class
import numpy as np

class geoTL():
    ''' 
    Class for Geometric Tensor Learning model.
    '''
    def __init__(self, n, sizes, conf_file = ''):
        ''' Initialize the parameters and variables. '''
        if ~conf_file:
            self.n = n
            self.sizes = sizes
            self.alpha = [
                [10**-2 for i in range(self.n)],
                [10**-2 for i in range(self.n)],
                [10**-2 for i in range(self.n)],
                [10**-3 for i in range(self.n)]
            ]
            self.theta = [[10**-2 for i in range(self.n)]]
            self.gamma = [[10**-2 for i in range(self.n)]]
