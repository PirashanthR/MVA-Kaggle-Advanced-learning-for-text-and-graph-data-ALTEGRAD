'''
GoQFeatures -- ALTEGRAD Challenge Fall 2017 -- RATNAMOGAN Pirashanth -- SAYEM Othmane

This file contains the function that allows to compute what we have called
'GoQFeatures'. It computes what has been called "Graph of questions features"
Extract the features from the graph that can be created using the comparisons in train and test set
'''

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
from scipy.spatial import distance


def intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))

def unmatching(text_1,text_2): 
    '''
    Nb of unmatching neighbours between two list of neighbours
    Param: @text_1 : (list str) list of neighbours to use for the comparison
    @text_2 : (list str) list of neighbours to use for the comparison
    Return: the feature
    '''
    nb_wd__not_shared = len(([i for i in text_1 if i not in text_2]))
    nb_wd__not_shared = nb_wd__not_shared+len(([i for i in text_2 if i not in text_1]))

    return nb_wd__not_shared   

def random_walk(G, node, walk_length):
    '''
    Random walk for Deep learning for graphs
    Param: @G: graph
    @node: initial node for the random walk
    @walk_length: walk length
    Return: list of str defining the walk
    '''
    walk = [node]
    for i in range(walk_length):
        neighbors =  list(G.neighbors(walk[i]))
        walk.append(neighbors[randint(0,len(neighbors)-1)])

    walk = [str(node) for node in walk]
    return walk

def generate_walks(graph, num_walks, walk_length):
    '''
    Generate multiple walks
    Param: @G: graph
    @num_walks: number of walks to compute
    @walk_length: walk length
    Return: list of random walks
    '''
    graph_nodes = graph.nodes()
    walks = []
    for i in range(num_walks):
        graph_nodes = np.random.permutation(graph_nodes)
        for j in range(graph.number_of_nodes()):
            walk = random_walk(graph, graph_nodes[j], walk_length)
            walks.append(walk)
    
    return walks

def neighborsofneighborslist(graph,node):
    '''
    Return the union between a list of neighbours of a node and the list
    of neighbours of the neighbours of a node (neighbours of order 2)
    '''
    neighbors = graph.neighbors(node)
    neighborsofneighbors = list(neighbors)
    for i in neighbors:
        new_neighb = graph.neighbors(i)
        neighborsofneighbors = list(set(new_neighb+neighborsofneighbors))
    return neighborsofneighbors

def intersectnbofnb(graph,node1,node2):
    '''
    Nb of common neigbours 
    '''
    nb_1 = neighborsofneighborslist(graph,node1)
    nb_2 = neighborsofneighborslist(graph,node2)
    return len(list(set(nb_1+nb_2)))

def avergeNeighbDegree(graph,node1):
    '''
    Average neighbours degree
    '''
    list_nb = graph.neighbors(node1)
    degree_list = graph.degree()
    nb_apparay_degree_list = np.array(degree_list)
    freq_nb = nb_apparay_degree_list[list_nb]
    return freq_nb.mean()
    
def dice(text_1,text_2):
    '''
    Compute the dice distance between two list of neighbours
    Param: @text_1: (list) list of str of neighbours
    @text_2: (list) list of str of neighbours
    '''
    nb_wd_shared = len(([i for i in text_1 if i in text_2]))

    nb_wd_1 = len(text_1)
    nb_wd_2 = len(text_2)
    return 2*nb_wd_shared/(nb_wd_1+nb_wd_2)

def jaccard(text_1,text_2):
    '''
    Compute the jaccard distance between two list of neighbours
    Param: @text_1: (list) list of str of neighbours
    @text_2: (list) list of str of neighbours
    '''
    nb_wd_shared = len(([i for i in text_1 if i in text_2]))

    nb_union_word = len(set(text_1+text_2))
    return nb_wd_shared/nb_union_word

def overlap(text_1,text_2):
    '''
    Compute the overlap distance between two list of neighbours
    Param: @text_1: (list) list of str of neighbours
    @text_2: (list) list of str of neighbours
    '''
    nb_wd_shared = len(([i for i in text_1 if i in text_2]))
    nb_wd_1 = len(text_1)
    nb_wd_2 = len(text_2)
    return nb_wd_shared/max(min(nb_wd_1,nb_wd_2),1)

def cosine_wd(text_1,text_2):
    '''
    Compute the cosine distance between two list of neighbours
    Param: @text_1: (list) list of str of neighbours
    @text_2: (list) list of str of neighbours
    '''
    nb_wd_shared = len(([i for i in text_1 if i in text_2]))
    nb_wd_1 = len(text_1)
    nb_wd_2 = len(text_2)
    return nb_wd_shared/max(math.sqrt(nb_wd_1*nb_wd_2),1)

def compute_all_compare_feature(text_1,text_2):
    return [dice(text_1,text_2),jaccard(text_1,text_2),overlap(text_1,text_2),cosine_wd(text_1,text_2)]



def create_GoQ_Features(list_pairs_train,list_pairs_test,list_val_to_create_graph,ids2ind\
                        ,num_walks=30,walk_length=40,d=128,window_size=3):
    """
    Create the features from the graph that we can obtain using questions 
    id
    param: list_pairs_train: list of pairs of question to compare for training
            list_pars_test: list of pairs of question to compare for test phases
            ids2ind: index of each question
    return: X_GoQ_train, X_GoQ_test: GoQ features for training and testing sets
    """
    
    ###Create the igraph Graph###     
    #nb_of_vertices = max(ids2ind.values())+1
    g = Graph()
    g.add_vertices(list(ids2ind.keys()))
    list_of_edges = list_pairs_train+list_pairs_test+list_val_to_create_graph

    ###Create bidirectional edges###
    #list_of_edges_rev = [(ids2ind[pair[1]],ids2ind[pair[0]]) for pair in all_pairs ]

    g.add_edges(list_of_edges)
    #g.add_edges(list_of_edges_rev)
    
    
    list_of_edges_nx = [(ids2ind[pair[0]],ids2ind[pair[1]]) for pair in list_of_edges]
    
    g_nx = nx.Graph(list_of_edges_nx)#+list_of_edges_rev)
    walks = generate_walks(g_nx, num_walks, walk_length)
    node_emb = Word2Vec(walks, size=d, window=window_size, min_count=0, sg=1, workers=cpu_count(), iter=15)
    

    core_number = list(g.coreness())
    page_rankprob = list(g.pagerank())
    ####Create features##########
    Feature_freq_train = []
    Feature_common_neighbors_train = []
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
    
    New_features_train =[]
    for pair in list_pairs_train:
        H = g.subgraph(sum(g.neighborhood([pair[0], pair[1]], 3),[]))
        H.radius()
        
        q1_node = H.vs.find(name=pair[0])
        q2_node = H.vs.find(name=pair[1])
    
        New_features_train.append(list())
        ###########Add node index##########
        tmp = H.evcent()
        New_features_train[-1].append(tmp[q1_node.index])
        New_features_train[-1].append(tmp[q2_node.index])
        New_features_train[-1].append((tmp[q1_node.index]+tmp[q2_node.index])/2)
        New_features_train[-1].append(abs(tmp[q1_node.index]-tmp[q2_node.index]))
        New_features_train[-1].append(H.edge_disjoint_paths(q1_node.index, q2_node.index))
        New_features_train[-1] += [sum(H.pagerank([q1_node.index, q2_node.index]))/2]
        New_features_train[-1].append(H.transitivity_undirected())
    
    
        tmp = H.betweenness([q1_node.index, q2_node.index])
        New_features_train[-1].append(tmp[0])
        New_features_train[-1].append(tmp[1])
        New_features_train[-1].append((tmp[0]+tmp[1])/2)
        New_features_train[-1].append(abs(tmp[0]-tmp[1]))
        
        #assortativity
        New_features_train[-1].append(H.assortativity_degree())
        New_features_train[-1].append(H.average_path_length())
        New_features_train[-1].append(H.bibcoupling(q1_node.index)[0][q2_node.index])

        New_features_train[-1].append(H.density())
        
        tmp1 = H.eccentricity(q1_node.index)
        tmp2 = H.eccentricity(q2_node.index)
        avg  = (tmp1+tmp2)/2
        diff = abs(tmp1-tmp2)
        
        New_features_train[-1].append(tmp1)
        New_features_train[-1].append(tmp2)
        New_features_train[-1].append(avg)
        New_features_train[-1].append(diff)
        
        
        
        New_features_train[-1].append(H.girth())
        
        New_features_train[-1].append(H.maxflow_value(q1_node.index, q2_node.index))
        
        New_features_train[-1].append(H.similarity_inverse_log_weighted(q1_node.index)[0][q2_node.index])
        
        New_features_train[-1].append(H.similarity_jaccard(pairs=[(q1_node.index, q2_node.index)])[0])
    

        q1_node_glob = g.vs.find(name=pair[0])
        q2_node_glob = g.vs.find(name=pair[1])
    
        freq_1 = min(g.degree()[q1_node_glob.index],g.degree()[q2_node_glob.index])
        freq_2 = max(g.degree()[q1_node_glob.index],g.degree()[q2_node_glob.index])
        minfreq = min(freq_1,freq_2)
        maxfreq = max(freq_1,freq_2)
        a_nb1= avergeNeighbDegree(g,q1_node_glob.index)
        a_nb2= avergeNeighbDegree(g,q2_node_glob.index)

        Feature_freq_train.append([freq_1,freq_2,minfreq/maxfreq,a_nb1,a_nb2])
        
        neighbors_1 = g.neighbors(q1_node_glob.index)
        neighbors_2 = g.neighbors(q2_node_glob.index)
        Feature_common_neighbors_train.append(min(len(intersect(neighbors_1,neighbors_2)),5))
        Feature_union_neighbors_train.append(len(set(neighbors_1+neighbors_2)))
        Number_of_neighboors_1_train.append(max(len(neighbors_1),len(neighbors_2)))
        Number_of_neighboors_2_train.append(min(len(neighbors_1),len(neighbors_2)))
        Core_number_1_train.append(max(core_number[q1_node_glob.index],core_number[q2_node_glob.index]))
        Core_number_2_train.append(min(core_number[q1_node_glob.index],core_number[q2_node_glob.index]))
        
        DeepGraph_train.append(cosine_similarity((node_emb.wv[str(ids2ind[pair[0]])]).reshape(1, -1),node_emb.wv[str(ids2ind[pair[1]])].reshape(1, -1)))
        DeepGrah_trainv2.append([np.linalg.norm((node_emb.wv[str(ids2ind[pair[0]])]).reshape(1, -1) - node_emb.wv[str(ids2ind[pair[1]])].reshape(1, -1)),\
                                distance.cityblock((node_emb.wv[str(ids2ind[pair[0]])]).reshape(1, -1),node_emb.wv[str(ids2ind[pair[1]])].reshape(1, -1)),\
                                distance.jaccard((node_emb.wv[str(ids2ind[pair[0]])]).reshape(1, -1),node_emb.wv[str(ids2ind[pair[1]])].reshape(1, -1)),\
                                distance.canberra((node_emb.wv[str(ids2ind[pair[0]])]).reshape(1, -1),node_emb.wv[str(ids2ind[pair[1]])].reshape(1, -1)),\
                                distance.minkowski((node_emb.wv[str(ids2ind[pair[0]])]).reshape(1, -1),node_emb.wv[str(ids2ind[pair[1]])].reshape(1, -1),3),\
                                distance.braycurtis((node_emb.wv[str(ids2ind[pair[0]])]).reshape(1, -1),node_emb.wv[str(ids2ind[pair[1]])].reshape(1, -1))])
    
        Nbneighofneigh_train.append(intersectnbofnb(g,q1_node_glob.index,q2_node_glob.index))
        Extra_neighbour_comparison_train.append(compute_all_compare_feature(neighbors_1,neighbors_2))
        page_rank_train_1.append(max(page_rankprob[q1_node_glob.index],page_rankprob[q2_node_glob.index]))
        page_rank_train_2.append(min(page_rankprob[q1_node_glob.index],page_rankprob[q2_node_glob.index]))
        Unmatching_train.append(unmatching(neighbors_1,neighbors_2))
        g.delete_edges([(q1_node_glob.index,q2_node_glob.index)])
        #g.delete_edges([(ids2ind[pair[1]],ids2ind[pair[0]])])
        Shortest_path_train.append(min(g.shortest_paths(source=q1_node_glob.index,target=q2_node_glob.index,mode=3)[0][0],100000))
        g.add_edges([(q1_node_glob.index,q2_node_glob.index)])
        #g.add_edges([(ids2ind[pair[1]],ids2ind[pair[0]])])

    
    Feature_freq_train = np.array(Feature_freq_train)
    Feature_common_neighbors_train = np.array(Feature_common_neighbors_train)    
    Feature_union_neighbors_train = np.array(Feature_union_neighbors_train)
    Number_of_neighboors_1_train = np.array(Number_of_neighboors_1_train)
    Number_of_neighboors_2_train = np.array(Number_of_neighboors_2_train)
    Core_number_1_train = np.array(Core_number_1_train)
    Core_number_2_train = np.array(Core_number_2_train)  
    DeepGraph_train= np.array(DeepGraph_train)
    DeepGrah_trainv2= np.array(DeepGrah_trainv2)
    page_rank_train_1 = np.array(page_rank_train_1)
    page_rank_train_2 = np.array(page_rank_train_2)
    Nbneighofneigh_train = np.array(Nbneighofneigh_train)
    Unmatching_train = np.array(Unmatching_train)
    Shortest_path_train = np.array(Shortest_path_train)
    New_features_train = np.array(New_features_train)
    #vertex_connectivity_train = np.array(vertex_connectivity_train)
    Extra_neighbour_comparison_train = np.array(Extra_neighbour_comparison_train)
    
    X_GoQ_train = np.concatenate([Feature_freq_train,Feature_common_neighbors_train.reshape((-1,1))\
                                       ,Feature_union_neighbors_train.reshape((-1,1)),Number_of_neighboors_1_train.reshape((-1,1)),Number_of_neighboors_2_train.reshape((-1,1)),\
                                       Core_number_1_train.reshape((-1,1)),Core_number_2_train.reshape((-1,1)),\
                                       DeepGraph_train.reshape((-1,1)),DeepGrah_trainv2,Nbneighofneigh_train.reshape((-1,1)),\
                                       page_rank_train_1.reshape((-1,1)),page_rank_train_2.reshape((-1,1)),\
                                       Extra_neighbour_comparison_train,\
                                       Unmatching_train.reshape((-1,1)),Shortest_path_train.reshape((-1,1)),\
                                       New_features_train],axis=1)
    
    
    
    Feature_freq_test = []
    Feature_common_neighbors_test = []
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

    New_features_test = []
    for pair in list_pairs_test:
        
        H = g.subgraph(sum(g.neighborhood([pair[0], pair[1]], 3),[]))
        H.radius()
        
        q1_node = H.vs.find(name=pair[0])
        q2_node = H.vs.find(name=pair[1])
    
        New_features_test.append(list())
        ###########Add node index##########
        tmp = H.evcent()
        New_features_test[-1].append(tmp[q1_node.index])
        New_features_test[-1].append(tmp[q2_node.index])
        New_features_test[-1].append((tmp[q1_node.index]+tmp[q2_node.index])/2)
        New_features_test[-1].append(abs(tmp[q1_node.index]-tmp[q2_node.index]))
        New_features_test[-1].append(H.edge_disjoint_paths(q1_node.index, q2_node.index))
        New_features_test[-1] += [sum(H.pagerank([q1_node.index, q2_node.index]))/2]
        New_features_test[-1].append(H.transitivity_undirected())
    
    
        tmp = H.betweenness([q1_node.index, q2_node.index])
        New_features_test[-1].append(tmp[0])
        New_features_test[-1].append(tmp[1])
        New_features_test[-1].append((tmp[0]+tmp[1])/2)
        New_features_test[-1].append(abs(tmp[0]-tmp[1]))
        
        #assortativity
        New_features_test[-1].append(H.assortativity_degree())
        New_features_test[-1].append(H.average_path_length())
        New_features_test[-1].append(H.bibcoupling(q1_node.index)[0][q2_node.index])

        New_features_test[-1].append(H.density())
        
        tmp1 = H.eccentricity(q1_node.index)
        tmp2 = H.eccentricity(q2_node.index)
        avg  = (tmp1+tmp2)/2
        diff = abs(tmp1-tmp2)
        
        New_features_test[-1].append(tmp1)
        New_features_test[-1].append(tmp2)
        New_features_test[-1].append(avg)
        New_features_test[-1].append(diff)
        
        
        
        New_features_test[-1].append(H.girth())
        
        New_features_test[-1].append(H.maxflow_value(q1_node.index, q2_node.index))
        
        New_features_test[-1].append(H.similarity_inverse_log_weighted(q1_node.index)[0][q2_node.index])
        
        New_features_test[-1].append(H.similarity_jaccard(pairs=[(q1_node.index, q2_node.index)])[0])
    

        
        q1_node_glob = g.vs.find(name=pair[0])
        q2_node_glob = g.vs.find(name=pair[1])
    
        freq_1 = min(g.degree()[q1_node_glob.index],g.degree()[q2_node_glob.index])
        freq_2 = max(g.degree()[q1_node_glob.index],g.degree()[q2_node_glob.index])
        minfreq = min(freq_1,freq_2)
        maxfreq = max(freq_1,freq_2)
        a_nb1= avergeNeighbDegree(g,q1_node_glob.index)
        a_nb2= avergeNeighbDegree(g,q2_node_glob.index)

        Feature_freq_test.append([freq_1,freq_2,minfreq/maxfreq,a_nb1,a_nb2])
        neighbors_1 = g.neighbors(q1_node_glob.index)
        neighbors_2 = g.neighbors(q2_node_glob.index)
        Feature_common_neighbors_test.append(min(len(intersect(neighbors_1,neighbors_2)),5))
        Feature_union_neighbors_test.append(len(set(neighbors_1+neighbors_2)))
        Number_of_neighboors_1_test.append(max(len(neighbors_1),len(neighbors_2)))
        Number_of_neighboors_2_test.append(min(len(neighbors_1),len(neighbors_2)))
        Core_number_1_test.append(max(core_number[q1_node_glob.index],core_number[q2_node_glob.index]))
        Core_number_2_test.append(min(core_number[q1_node_glob.index],core_number[q2_node_glob.index]))
        DeepGraph_test.append(cosine_similarity((node_emb.wv[str(ids2ind[pair[0]])]).reshape(1, -1),node_emb.wv[str(ids2ind[pair[1]])].reshape(1, -1)))
        DeepGraph_testv2.append([np.linalg.norm((node_emb.wv[str(ids2ind[pair[0]])]).reshape(1, -1) - node_emb.wv[str(ids2ind[pair[1]])].reshape(1, -1)),\
                                distance.cityblock((node_emb.wv[str(ids2ind[pair[0]])]).reshape(1, -1),node_emb.wv[str(ids2ind[pair[1]])].reshape(1, -1)),\
                                distance.jaccard((node_emb.wv[str(ids2ind[pair[0]])]).reshape(1, -1),node_emb.wv[str(ids2ind[pair[1]])].reshape(1, -1)),\
                                distance.canberra((node_emb.wv[str(ids2ind[pair[0]])]).reshape(1, -1),node_emb.wv[str(ids2ind[pair[1]])].reshape(1, -1)),\
                                distance.minkowski((node_emb.wv[str(ids2ind[pair[0]])]).reshape(1, -1),node_emb.wv[str(ids2ind[pair[1]])].reshape(1, -1),3),\
                                distance.braycurtis((node_emb.wv[str(ids2ind[pair[0]])]).reshape(1, -1),node_emb.wv[str(ids2ind[pair[1]])].reshape(1, -1))])
    
        
        
        
        
        Nbneighofneigh_test.append(intersectnbofnb(g,ids2ind[pair[0]],ids2ind[pair[1]]))
        #vertex_connectivity_test.append(g.vertex_disjoint_paths(source=ids2ind[pair[0]],target= ids2ind[pair[1]],neighbors="ignore"))
        Extra_neighbour_comparison_test.append(compute_all_compare_feature(neighbors_1,neighbors_2))
        page_rank_test_1.append(max(page_rankprob[q1_node_glob.index],page_rankprob[q2_node_glob.index]))
        page_rank_test_2.append(min(page_rankprob[q1_node_glob.index],page_rankprob[q2_node_glob.index]))
        Unmatching_test.append(unmatching(neighbors_1,neighbors_2))
        g.delete_edges([(q1_node_glob.index,q2_node_glob.index)])
        #g.delete_edges([(ids2ind[pair[1]],ids2ind[pair[0]])])
        Shortest_path_test.append(min(g.shortest_paths(source=q1_node_glob.index,target=q2_node_glob.index,mode=3)[0][0],100000))
        g.add_edges([(q1_node_glob.index,q2_node_glob.index)])

    Feature_freq_test= np.array(Feature_freq_test)
    Feature_common_neighbors_test=np.array(Feature_common_neighbors_test)    
    Feature_union_neighbors_test = np.array(Feature_union_neighbors_test)
    Number_of_neighboors_1_test = np.array(Number_of_neighboors_1_test)
    Number_of_neighboors_2_test = np.array(Number_of_neighboors_2_test)
    Core_number_1_test = np.array(Core_number_1_test)
    Core_number_2_test = np.array(Core_number_2_test)  
    DeepGraph_test = np.array(DeepGraph_test)
    DeepGraph_testv2 = np.array(DeepGraph_testv2)
    Nbneighofneigh_test = np.array(Nbneighofneigh_test)
    #vertex_connectivity_test = np.array(vertex_connectivity_test)
    Extra_neighbour_comparison_test = np.array(Extra_neighbour_comparison_test)
    page_rank_test_1 = np.array(page_rank_test_1)
    page_rank_test_2 = np.array(page_rank_test_2)
    Unmatching_test = np.array(Unmatching_test)
    Shortest_path_test = np.array(Shortest_path_test)
    New_features_test = np.array(New_features_test)


    X_GoQ_test = np.concatenate([Feature_freq_test,Feature_common_neighbors_test.reshape((-1,1)),\
                                      Feature_union_neighbors_test.reshape((-1,1)),Number_of_neighboors_1_test.reshape((-1,1)),Number_of_neighboors_2_test.reshape((-1,1)),\
                                       Core_number_1_test.reshape((-1,1)),Core_number_2_test.reshape((-1,1)),\
                                       DeepGraph_test.reshape((-1,1)),DeepGraph_testv2,Nbneighofneigh_test.reshape((-1,1)),\
                                       page_rank_test_1.reshape((-1,1)),page_rank_test_2.reshape((-1,1)),Extra_neighbour_comparison_test,\
                                       Unmatching_test.reshape((-1,1)),Shortest_path_test.reshape((-1,1)),New_features_test],axis=1)

    return X_GoQ_train,X_GoQ_test