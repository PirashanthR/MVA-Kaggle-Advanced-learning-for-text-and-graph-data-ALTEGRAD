'''
Utils_graphkernels -- ALTEGRAD Challenge Fall 2017 -- RATNAMOGAN Pirashanth -- SAYEM Othmane

This file contains the tools to compute the graph kernel from GoW.
It is used in the GoWFeatures script.
'''
import networkx as nx
import numpy as np


#=================== Walk Kernel ============================================
def spgk(graphs,d):
    """
    Computes the shortest path kernel
    and returns the kernel matrix.

    Parameters
    ----------
    graphs : list
    A list of NetworkX graphs
    d : maximum length of shortest path between two vertices
    Returns
    -------
    K : numpy matrix
    The kernel matrix

    """
    N = len(graphs)
    all_paths = {}
    all_npaths = {}
    sp_counts = {}
    n_counts={}
    for i in range(N):
        sp_lengths = dict(nx.shortest_path_length(graphs[i])) # dictionary containing, for each node, the length of the shortest path with all other nodes in the graph
        sp_counts[i] = {}
        n_counts[i] = {}
        nodes = graphs[i].nodes()
        for v1 in nodes:
            nlabel = tuple(graphs[i].node[v1]['label'])
            if nlabel in n_counts[i]:
                n_counts[i][nlabel] += 1
            else:
                n_counts[i][nlabel] = 1
            # index of label in feature space
            if nlabel not in all_npaths:
                all_npaths[nlabel] = len(all_npaths)

            
            for v2 in nodes:
                if (v2 in (sp_lengths[v1]))and (sp_lengths[v1][v2]<=d) :
                    label = tuple(sorted([graphs[i].node[v1]['label'], graphs[i].node[v2]['label']]) + [sp_lengths[v1][v2]])
                    # update or initialize 'sp_counts[i][label]'
                    if label in sp_counts[i]:
                        sp_counts[i][label] += 1
                    else:
                        sp_counts[i][label] = 1
                    # index of label in feature space
                    if label not in all_paths:
                        all_paths[label] = len(all_paths)

    phi = np.zeros((N,len(all_paths)))
    nphi = np.zeros((N,len(all_npaths)))

    # construct feature vectors of each graph
    for i in range(N):
        for label in sp_counts[i]:
            phi[i,all_paths[label]] = sp_counts[i][label]
        for nlabel in n_counts[i]:
            nphi[i,all_npaths[nlabel]] = n_counts[i][nlabel]
    
    K = np.dot(phi,phi.T)
    Kn = np.dot(nphi,nphi.T)
    return K+Kn

###### Short Path Kernel
#K = sp_kernel(q_graph)
def normalizekm(K):
    v = np.sqrt(np.diag(K));
    nm =  np.outer(v,v)
    Knm = np.power(nm, -1)
    Knm = np.nan_to_num(Knm) 
    normalized_K = K * Knm;
    return normalized_K