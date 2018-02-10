# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 14:40:31 2017

@author: Pirashanth
"""

import numpy as np
from preprocessing.preprocess import preprocess_raw_text
from igraph import Graph
import networkx as nx
from random import randint
from scipy.io import loadmat
from gensim.models import Word2Vec
from multiprocessing import cpu_count
from sklearn.metrics.pairwise import cosine_similarity
import math

def intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))

def unmatching(text_1,text_2):    
    nb_wd__not_shared = len(([i for i in text_1 if i not in text_2]))
    nb_wd__not_shared = nb_wd__not_shared+len(([i for i in text_2 if i not in text_1]))

    return nb_wd__not_shared   

def random_walk(G, node, walk_length):
    walk = [node]
    for i in range(walk_length):
        neighbors =  list(G.neighbors(walk[i]))
        walk.append(neighbors[randint(0,len(neighbors)-1)])

    walk = [str(node) for node in walk]
    return walk

def generate_walks(graph, num_walks, walk_length):
    graph_nodes = graph.nodes()
    walks = []
    for i in range(num_walks):
        graph_nodes = np.random.permutation(graph_nodes)
        for j in range(graph.number_of_nodes()):
            walk = random_walk(graph, graph_nodes[j], walk_length)
            walks.append(walk)
    
    return walks

def neighborsofneighborslist(graph,node):
    neighbors = graph.neighbors(node)
    neighborsofneighbors = list(neighbors)
    for i in neighbors:
        new_neighb = graph.neighbors(i)
        neighborsofneighbors = list(set(new_neighb+neighborsofneighbors))
    return neighborsofneighbors

def intersectnbofnb(graph,node1,node2):
    nb_1 = neighborsofneighborslist(graph,node1)
    nb_2 = neighborsofneighborslist(graph,node2)
    return len(list(set(nb_1+nb_2)))

def dice(text_1,text_2):
    nb_wd_shared = len(([i for i in text_1 if i in text_2]))

    nb_wd_1 = len(text_1)
    nb_wd_2 = len(text_2)
    return 2*nb_wd_shared/max((nb_wd_1+nb_wd_2),1)

def jaccard(text_1,text_2):
    nb_wd_shared = len(([i for i in text_1 if i in text_2]))

    nb_union_word = len(set(text_1+text_2))
    return nb_wd_shared/max(nb_union_word,1)

def overlap(text_1,text_2):
    nb_wd_shared = len(([i for i in text_1 if i in text_2]))
    nb_wd_1 = len(text_1)
    nb_wd_2 = len(text_2)
    return nb_wd_shared/max(min(nb_wd_1,nb_wd_2),1)

def cosine_wd(text_1,text_2):
    nb_wd_shared = len(([i for i in text_1 if i in text_2]))
    nb_wd_1 = len(text_1)
    nb_wd_2 = len(text_2)
    return nb_wd_shared/max(math.sqrt(nb_wd_1*nb_wd_2),1)

def compute_all_compare_feature(text_1,text_2):
    return [dice(text_1,text_2),jaccard(text_1,text_2),overlap(text_1,text_2),cosine_wd(text_1,text_2)]



def create_graph_of_positive_pairs(list_pairs_train,y_train,list_pairs_test,prediction_test,ids2ind,thresold=0.3 ,num_walks=10,walk_length=10,d=128,window_size=5):
    pairs_to_consider_train = [list_pairs_train[i] for i in range(len(y_train)) if y_train[i]==1 ]
    pairs_to_consider_test = [list_pairs_test[i] for i in range(len(prediction_test)) if prediction_test[i]>thresold ]
    all_pairs_to_consider = pairs_to_consider_train+ pairs_to_consider_test
    ###Create the igraph Graph###     
    nb_of_vertices = max(ids2ind.values())+1
    g = Graph()
    g.add_vertices(nb_of_vertices)
    ###Create bidirectional edges###
    list_of_edges = [(ids2ind[pair[0]],ids2ind[pair[1]]) for pair in all_pairs_to_consider]
#    list_of_edges_rev = [(ids2ind[pair[1]],ids2ind[pair[0]]) for pair in all_pairs ]

    g.add_edges(list_of_edges)
    #g_nx = nx.Graph(list_of_edges)#+list_of_edges_rev)
    #walks = generate_walks(g_nx, num_walks, walk_length)
    #node_emb = Word2Vec(walks, size=d, window=window_size, min_count=0, sg=1, workers=cpu_count(), iter=5)
    

    core_number = list(g.coreness())
    page_rankprob = list(g.pagerank())
    ####Create features##########
    Feature_freq_train = []
    Feature_common_neighbors_train = []
    Feature_common_neighbors_superior_to_five_train = []
    Feature_union_neighbors_train = []
    Number_of_neighboors_1_train = []
    Number_of_neighboors_2_train = []
    
    Core_number_1_train = []
    Core_number_2_train = []
    DeepGraph_train =[]
    DeepGrah_trainv2 = []
    
    Nbneighofneigh_train = []
    #vertex_connectivity_train =[]
    page_rank_train_1 = []
    page_rank_train_2 = []
    Extra_neighbour_comparison_train = []
    Unmatching_train = []
    Shortest_path_train = []    
    
    for pair in list_pairs_train:
        max_freq = max(g.degree()[ids2ind[pair[0]]],g.degree()[ids2ind[pair[1]]])
        min_freq = min(g.degree()[ids2ind[pair[0]]],g.degree()[ids2ind[pair[1]]])
        if max_freq!=0:
            Feature_freq_train.append(min_freq/max_freq)
        else:
            Feature_freq_train.append(0)
        neighbors_1 = g.neighbors(ids2ind[pair[0]])
        neighbors_2 = g.neighbors(ids2ind[pair[1]])
        Feature_common_neighbors_train.append(len(intersect(neighbors_1,neighbors_2)))
        Feature_common_neighbors_superior_to_five_train.append(int(len(intersect(neighbors_1,neighbors_2))>5))
        Feature_union_neighbors_train.append(len(set(neighbors_1+neighbors_2)))
        Number_of_neighboors_1_train.append(len(neighbors_1))
        Number_of_neighboors_2_train.append(len(neighbors_2))
        Core_number_1_train.append(core_number[ids2ind[pair[0]]])
        Core_number_2_train.append(core_number[ids2ind[pair[1]]])
        
        #DeepGraph_train.append(cosine_similarity((node_emb.wv[str(ids2ind[pair[0]])]).reshape(1, -1),node_emb.wv[str(ids2ind[pair[1]])].reshape(1, -1)))
        #DeepGrah_trainv2.append(np.linalg.norm((node_emb.wv[str(ids2ind[pair[0]])]).reshape(1, -1) - node_emb.wv[str(ids2ind[pair[1]])].reshape(1, -1)))
    
        Nbneighofneigh_train.append(intersectnbofnb(g,ids2ind[pair[0]],ids2ind[pair[1]]))
        Extra_neighbour_comparison_train.append(compute_all_compare_feature(neighbors_1,neighbors_2))
        page_rank_train_1.append(page_rankprob[ids2ind[pair[0]]])
        page_rank_train_2.append(page_rankprob[ids2ind[pair[1]]])
        Unmatching_train.append(unmatching(neighbors_1,neighbors_2))
        #g.delete_edges([(ids2ind[pair[0]],ids2ind[pair[1]])])
        #g.delete_edges([(ids2ind[pair[1]],ids2ind[pair[0]])])
        Shortest_path_train.append(min(g.shortest_paths(source=ids2ind[pair[0]],target= ids2ind[pair[1]],mode=3)[0][0],100000))
        #g.add_edges([(ids2ind[pair[0]],ids2ind[pair[1]])])
    
    Feature_freq_train = np.array(Feature_freq_train)
    Feature_common_neighbors_train = np.array(Feature_common_neighbors_train)    
    Feature_common_neighbors_superior_to_five_train = np.array(Feature_common_neighbors_superior_to_five_train)
    Feature_union_neighbors_train = np.array(Feature_union_neighbors_train)
    Number_of_neighboors_1_train = np.array(Number_of_neighboors_1_train)
    Number_of_neighboors_2_train = np.array(Number_of_neighboors_2_train)
    Core_number_1_train = np.array(Core_number_1_train)
    Core_number_2_train = np.array(Core_number_2_train)  
    #DeepGraph_train= np.array(DeepGraph_train)
    #DeepGrah_trainv2= np.array(DeepGrah_trainv2)
    page_rank_train_1 = np.array(page_rank_train_1)
    page_rank_train_2 = np.array(page_rank_train_2)
    Nbneighofneigh_train = np.array(Nbneighofneigh_train)
    Unmatching_train = np.array(Unmatching_train)
    Shortest_path_train = np.array(Shortest_path_train)
    #vertex_connectivity_train = np.array(vertex_connectivity_train)
    Extra_neighbour_comparison_train = np.array(Extra_neighbour_comparison_train)
    
    X_GoQ_train = np.concatenate([Feature_freq_train.reshape((-1,1)),Feature_common_neighbors_train.reshape((-1,1))\
                                       ,Feature_union_neighbors_train.reshape((-1,1)),Number_of_neighboors_1_train.reshape((-1,1)),Number_of_neighboors_2_train.reshape((-1,1)),\
                                       Core_number_1_train.reshape((-1,1)),Core_number_2_train.reshape((-1,1)),\
                                       Nbneighofneigh_train.reshape((-1,1)),\
                                       page_rank_train_1.reshape((-1,1)),page_rank_train_2.reshape((-1,1)),\
                                       Extra_neighbour_comparison_train,Feature_common_neighbors_superior_to_five_train.reshape((-1,1)),\
                                       Unmatching_train.reshape((-1,1)),Shortest_path_train.reshape((-1,1))],axis=1)
    
    
    
    Feature_freq_test = []
    Feature_common_neighbors_test = []
    Feature_common_neighbors_superior_to_five_test = []
    Feature_union_neighbors_test = []
    Number_of_neighboors_1_test = []
    Number_of_neighboors_2_test = []
    Core_number_1_test = []
    Core_number_2_test = []
    DeepGraph_test =[]
    DeepGraph_testv2= []
    
    Nbneighofneigh_test = []

    #vertex_connectivity_test =[]
    Extra_neighbour_comparison_test = []
    page_rank_test_1 = []
    page_rank_test_2 = []
    
    Unmatching_test = []
    Shortest_path_test =[]

    for pair in list_pairs_test:
        max_freq = max(g.degree()[ids2ind[pair[0]]],g.degree()[ids2ind[pair[1]]])
        min_freq = min(g.degree()[ids2ind[pair[0]]],g.degree()[ids2ind[pair[1]]])
        if max_freq!=0:
            Feature_freq_test.append(min_freq/max_freq)
        else:
            Feature_freq_test.append(0)
        neighbors_1 = g.neighbors(ids2ind[pair[0]])
        neighbors_2 = g.neighbors(ids2ind[pair[1]])
        Feature_common_neighbors_test.append(len(intersect(neighbors_1,neighbors_2)))
        Feature_common_neighbors_superior_to_five_test.append(int(len(intersect(neighbors_1,neighbors_2))>5))
        Feature_union_neighbors_test.append(len(set(neighbors_1+neighbors_2)))
        Number_of_neighboors_1_test.append(len(neighbors_1))
        Number_of_neighboors_2_test.append(len(neighbors_2))
        Core_number_1_test.append(core_number[ids2ind[pair[0]]])
        Core_number_2_test.append(core_number[ids2ind[pair[1]]])
        #DeepGraph_test.append(cosine_similarity((node_emb.wv[str(ids2ind[pair[0]])]).reshape(1, -1),node_emb.wv[str(ids2ind[pair[1]])].reshape(1, -1)))
        #DeepGraph_testv2.append(np.linalg.norm((node_emb.wv[str(ids2ind[pair[0]])]).reshape(1, -1) - node_emb.wv[str(ids2ind[pair[1]])].reshape(1, -1)))
        Nbneighofneigh_test.append(intersectnbofnb(g,ids2ind[pair[0]],ids2ind[pair[1]]))
        #vertex_connectivity_test.append(g.vertex_disjoint_paths(source=ids2ind[pair[0]],target= ids2ind[pair[1]],neighbors="ignore"))
        Extra_neighbour_comparison_test.append(compute_all_compare_feature(neighbors_1,neighbors_2))
        page_rank_test_1.append(page_rankprob[ids2ind[pair[0]]])
        page_rank_test_2.append(page_rankprob[ids2ind[pair[1]]])
        Unmatching_test.append(unmatching(neighbors_1,neighbors_2))
        #g.delete_edges([(ids2ind[pair[0]],ids2ind[pair[1]])])
        #g.delete_edges([(ids2ind[pair[1]],ids2ind[pair[0]])])
        Shortest_path_test.append(min(g.shortest_paths(source=ids2ind[pair[0]],target= ids2ind[pair[1]],mode=3)[0][0],100000))
        #g.add_edges([(ids2ind[pair[0]],ids2ind[pair[1]])])


    Feature_freq_test= np.array(Feature_freq_test)
    Feature_common_neighbors_test=np.array(Feature_common_neighbors_test)    
    Feature_union_neighbors_test = np.array(Feature_union_neighbors_test)
    Number_of_neighboors_1_test = np.array(Number_of_neighboors_1_test)
    Number_of_neighboors_2_test = np.array(Number_of_neighboors_2_test)
    Core_number_1_test = np.array(Core_number_1_test)
    Core_number_2_test = np.array(Core_number_2_test)  
    #DeepGraph_test = np.array(DeepGraph_test)
    #DeepGraph_testv2 = np.array(DeepGraph_testv2)
    Nbneighofneigh_test = np.array(Nbneighofneigh_test)
    #vertex_connectivity_test = np.array(vertex_connectivity_test)
    Extra_neighbour_comparison_test = np.array(Extra_neighbour_comparison_test)
    page_rank_test_1 = np.array(page_rank_test_1)
    page_rank_test_2 = np.array(page_rank_test_2)
    Feature_common_neighbors_superior_to_five_test = np.array(Feature_common_neighbors_superior_to_five_test)
    Unmatching_test = np.array(Unmatching_test)
    Shortest_path_test = np.array(Shortest_path_test)

    X_GoQ_test = np.concatenate([Feature_freq_test.reshape((-1,1)),Feature_common_neighbors_test.reshape((-1,1)),\
                                      Feature_union_neighbors_test.reshape((-1,1)),Number_of_neighboors_1_test.reshape((-1,1)),Number_of_neighboors_2_test.reshape((-1,1)),\
                                       Core_number_1_test.reshape((-1,1)),Core_number_2_test.reshape((-1,1)),\
                                       Nbneighofneigh_test.reshape((-1,1)),\
                                       page_rank_test_1.reshape((-1,1)),page_rank_test_2.reshape((-1,1)),Extra_neighbour_comparison_test,\
                                       Feature_common_neighbors_superior_to_five_test.reshape((-1,1))\
                                       ,Unmatching_test.reshape((-1,1)),Shortest_path_test.reshape((-1,1))],axis=1)

    return X_GoQ_train,X_GoQ_test
    