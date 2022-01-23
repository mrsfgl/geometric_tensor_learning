
import networkx as nx
import numpy as np


def generate_graphs(sizes, dens_multiplier=1.1):
    ''' Generates Erdos-Renyi graphs with given size and densities.

    Parameters:

        sizes: List of integers
            Sizes of each mode of the product graph.

        dens_multiplier: constant
            Density multiplier of each mode.

    Outputs:

        Phi: List of graph Laplacians.
    '''

    n = len(sizes)
    # List of graphs for each mode
    G = [nx.erdos_renyi_graph(sizes[i],
         dens_multiplier*np.log(sizes[i])/sizes[i]) for i in range(n)]
    # Graph Laplacians of these graphs
    return [nx.laplacian_matrix(G[i]).todense() for i in range(n)]
