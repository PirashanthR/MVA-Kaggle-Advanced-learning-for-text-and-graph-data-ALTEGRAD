'''
GoW -- ALTEGRAD Challenge Fall 2017 -- RATNAMOGAN Pirashanth -- SAYEM Othmane

This file contains the function that allows to compute what we have called
'GoWFeatures'. It essentially computes the kernel shortest path from graph of words
described here :http://aclweb.org/anthology/D17-1202 (Nikolentzos et al.)
'''

import igraph
import networkx as nx
import nltk
import itertools
from Features.Utils_graphkernels import normalizekm,spgk
import numpy as np


def create_all_questions_graphs(data_full_word,w=2):
    '''
    Create all the graph of words from a list of tokens
    '''
    q_graph=[]
    for q in data_full_word:
        stemmer = nltk.stem.PorterStemmer()
        q_stemmed = list()
        for token in q:
            q_stemmed.append(stemmer.stem(token))
        q = q_stemmed
        if len(q)<=1:
            w_to_use=0
        else:
            w_to_use=w
        q_igraph = terms_to_graph(q,w_to_use)
        edges = q_igraph.get_edgelist()
        q_graph_i = nx.Graph(edges)
        nodes = q_graph_i.nodes()
        for v in nodes:
            q_graph_i.node[v]['label'] = q_igraph.vs['name'][v]
        q_graph.append(q_graph_i)
    return q_graph

def terms_to_graph(terms, w):
    '''This function returns a directed, weighted igraph from a list of terms (the tokens from the pre-processed text) e.g., ['quick','brown','fox'].
    Edges are weighted based on term co-occurence within a sliding window of fixed size 'w'.
    '''
    
    from_to = {}
    
    # create initial complete graph (first w terms)
    terms_temp = terms[0:w]
    indexes = list(itertools.combinations(range(min(w,len(terms_temp))), r=2))
    
    new_edges = []
    
    for my_tuple in indexes:
        new_edges.append(tuple([terms_temp[i] for i in my_tuple]))
    
    for new_edge in new_edges:
        if new_edge in from_to:
            from_to[new_edge] += 1
        else:
            from_to[new_edge] = 1

    # then iterate over the remaining terms
    for i in range(w, len(terms)):
        considered_term = terms[i] # term to consider
        terms_temp = terms[(i-w+1):(i+1)] # all terms within sliding window
        
        # edges to try
        candidate_edges = []
        for p in range(w-1):
            candidate_edges.append((terms_temp[p],considered_term))
    
        for try_edge in candidate_edges:
            
            if try_edge[1] != try_edge[0]:
            
            # if edge has already been seen, update its weight
                if try_edge in from_to:
                    from_to[try_edge] += 1
                                   
                # if edge has never been seen, create it and assign it a unit weight     
                else:
                    from_to[try_edge] = 1
    
    # create empty graph
    g = igraph.Graph(directed=True)
    
    # add vertices
    g.add_vertices(sorted(set(terms)))
    
    # add edges, direction is preserved since the graph is directed
    g.add_edges(from_to.keys())
    
    # set edge and vertex weights
    g.es['weight'] = from_to.values() # based on co-occurence within sliding window
    g.vs['weight'] = g.strength() # weighted degree
    
    return(g)


def create_features_GoW(list_pairs_train,list_pairs_test,data_full_word,ids2ind,w=2,d=3):
    '''
    http://aclweb.org/anthology/D17-1202
    
    Create Shortest-Path Graph Kernels between the graph of words 
    Param: @ list_pairs_train: trainings pairs of questions 
    @list_pairs_test: test pairs of questions
    @data_full_word: each questions as tokens
    @ids2ind: idx of each questions to indice for data_full_word
    @w: window parameter
    @d: max distance between nodes of the graph (see article for details)
    '''
    
    all_graphs= create_all_questions_graphs(data_full_word,w)
    
    K_train = []
    for pair in list_pairs_train:
        graphs = []
        graphs.append(all_graphs[ids2ind[pair[0]]])
        graphs.append(all_graphs[ids2ind[pair[1]]])
        K = spgk(graphs,d)
        K = normalizekm(K)
        '''
        #Extra features that doesn't improve the outcome
        if ((nx.number_of_nodes(all_graphs[ids2ind[pair[0]]])>0)and(nx.number_of_nodes(all_graphs[ids2ind[pair[1]]])>0)):
            max_transitivy = max(nx.transitivity(all_graphs[ids2ind[pair[0]]]),nx.transitivity(all_graphs[ids2ind[pair[1]]]),1)
            min_transitivy = min(nx.transitivity(all_graphs[ids2ind[pair[0]]]),nx.transitivity(all_graphs[ids2ind[pair[1]]]))
            ratio_transitivity = min_transitivy/max_transitivy
        else:
            ratio_transitivity=0
            
        if ((nx.number_of_nodes(all_graphs[ids2ind[pair[0]]])>0)and(nx.number_of_nodes(all_graphs[ids2ind[pair[1]]])>0)):
            max_assortativity = max(nx.degree_assortativity_coefficient(all_graphs[ids2ind[pair[0]]]),nx.degree_assortativity_coefficient(all_graphs[ids2ind[pair[1]]]),1)
            min_assortativity = min(nx.degree_assortativity_coefficient(all_graphs[ids2ind[pair[0]]]),nx.degree_assortativity_coefficient(all_graphs[ids2ind[pair[1]]]))
            ratio_assortativity = min_assortativity/max_assortativity
        else:
            ratio_assortativity = 0
    
        if ((nx.number_of_nodes(all_graphs[ids2ind[pair[0]]])>0)and(nx.number_of_nodes(all_graphs[ids2ind[pair[1]]])>0)):
            max_SP = max(nx.average_shortest_path_length(all_graphs[ids2ind[pair[0]]]),nx.average_shortest_path_length(all_graphs[ids2ind[pair[1]]]),1)
            min_SP = min(nx.average_shortest_path_length(all_graphs[ids2ind[pair[0]]]),nx.average_shortest_path_length(all_graphs[ids2ind[pair[1]]]))
            ratio_SP = min_SP/max_SP
        else:
            ratio_SP=0
        
        if ((nx.number_of_nodes(all_graphs[ids2ind[pair[0]]])>0)and(nx.number_of_nodes(all_graphs[ids2ind[pair[1]]])>0)):
            max_dens = max(nx.density(all_graphs[ids2ind[pair[0]]]),nx.density(all_graphs[ids2ind[pair[1]]]),1)
            min_dens = min(nx.density(all_graphs[ids2ind[pair[0]]]),nx.density(all_graphs[ids2ind[pair[1]]]))
            ratio_dens = min_dens/max_dens
        else:
            ratio_dens=0 '''
    
        K_train.append([K[0,1]])
    
    K_train= np.array(K_train)
    
    K_test = []
    for pair in list_pairs_test:
        graphs = []
        graphs.append(all_graphs[ids2ind[pair[0]]])
        graphs.append(all_graphs[ids2ind[pair[1]]])
        K = spgk(graphs,d)
        K = normalizekm(K)
        '''
        #Extra features that doesn't improve the outcome
        if ((nx.number_of_nodes(all_graphs[ids2ind[pair[0]]])>0)and(nx.number_of_nodes(all_graphs[ids2ind[pair[1]]])>0)):
            max_transitivy = max(nx.transitivity(all_graphs[ids2ind[pair[0]]]),nx.transitivity(all_graphs[ids2ind[pair[1]]]),1)
            min_transitivy = min(nx.transitivity(all_graphs[ids2ind[pair[0]]]),nx.transitivity(all_graphs[ids2ind[pair[1]]]))
            ratio_transitivity = min_transitivy/max_transitivy
        else:
            ratio_transitivity=0
            
        if ((nx.number_of_nodes(all_graphs[ids2ind[pair[0]]])>0)and(nx.number_of_nodes(all_graphs[ids2ind[pair[1]]])>0)):
            max_assortativity = max(nx.degree_assortativity_coefficient(all_graphs[ids2ind[pair[0]]]),nx.degree_assortativity_coefficient(all_graphs[ids2ind[pair[1]]]),1)
            min_assortativity = min(nx.degree_assortativity_coefficient(all_graphs[ids2ind[pair[0]]]),nx.degree_assortativity_coefficient(all_graphs[ids2ind[pair[1]]]))
            ratio_assortativity = min_assortativity/max_assortativity
        else:
            ratio_assortativity = 0
    
        if ((nx.number_of_nodes(all_graphs[ids2ind[pair[0]]])>0)and(nx.number_of_nodes(all_graphs[ids2ind[pair[1]]])>0)):
            max_SP = max(nx.average_shortest_path_length(all_graphs[ids2ind[pair[0]]]),nx.average_shortest_path_length(all_graphs[ids2ind[pair[1]]]),1)
            min_SP = min(nx.average_shortest_path_length(all_graphs[ids2ind[pair[0]]]),nx.average_shortest_path_length(all_graphs[ids2ind[pair[1]]]))
            ratio_SP = min_SP/max_SP
        else:
            ratio_SP=0
        
        if ((nx.number_of_nodes(all_graphs[ids2ind[pair[0]]])>0)and(nx.number_of_nodes(all_graphs[ids2ind[pair[1]]])>0)):
            max_dens = max(nx.density(all_graphs[ids2ind[pair[0]]]),nx.density(all_graphs[ids2ind[pair[1]]]),1)
            min_dens = min(nx.density(all_graphs[ids2ind[pair[0]]]),nx.density(all_graphs[ids2ind[pair[1]]]))
            ratio_dens = min_dens/max_dens
        else:
            ratio_dens=0 '''
    
        K_test.append([K[0,1]])
    
    if len(list_pairs_test)>0:
        K_test= np.array(K_test)

    return K_train,K_test