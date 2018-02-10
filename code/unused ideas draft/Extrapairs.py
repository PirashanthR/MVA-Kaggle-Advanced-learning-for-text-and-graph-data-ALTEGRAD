# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 18:52:09 2017

@author: Pirashanth
"""
import numpy as np
from igraph import Graph
from itertools import combinations
from sklearn.utils import shuffle

def create_set_same_semantic(y_train,list_pairs_train):
    y_train_array = np.array(y_train)
    list_index = np.array(range(len(y_train)))
    list_index_pos=list_index[y_train_array==1]
    positive_pairs = [list_pairs_train[i] for i in list_index_pos]
    g = Graph()
    for pair in positive_pairs:
        q1 = pair[0]
        q2 = pair[1]
        if'name' in g.vertex_attributes():
            list_of_vertices = list(g.vs['name'])
        else:
            list_of_vertices = []
        if q1 not in list_of_vertices:
            g.add_vertex(q1)
        
        if q2 not in list_of_vertices:
            g.add_vertex(q2)
        
        g.add_edge(q1,q2)
        
    list_of_disconected_subgraphs = g.decompose()
    return [i.vs['name'] for i in list_of_disconected_subgraphs]

def create_all_positive_pairs(list_semantic):
    all_pairs = []
    for semantic in list_semantic:
        all_pairs.extend(combinations(semantic,2))
    return all_pairs
    
def find_neighbours_list(list_semantic,question_id):
    for semantic in list_semantic:
        if question_id in semantic:
            return semantic
    return [question_id]

def create_all_negative_pairs(list_pairs_train,list_semantic,y_train):
    all_pairs = []
    y_train_array = np.array(y_train)
    list_index = np.array(range(len(y_train)))
    list_index_neg=list_index[y_train_array==0]
    negative_pairs = [list_pairs_train[i] for i in list_index_neg]
    for pair in negative_pairs:
        q1 = pair[0]
        q2 = pair[1]
        list_similar_quest1 = find_neighbours_list(list_semantic,q1)
        list_similar_quest2= find_neighbours_list(list_semantic,q2)
        for i in list_similar_quest1:
            for j in list_similar_quest2:
                if ((i,j) not in all_pairs) and ((j,i) not in all_pairs):
                    all_pairs.append((i,j))
    return all_pairs
    
def create_all_pairs(list_pairs_train,y_train):
    list_same_similarity = create_set_same_semantic(y_train,list_pairs_train)
    all_p = create_all_positive_pairs(list_same_similarity)
    all_n = create_all_negative_pairs(list_pairs_train,list_same_similarity,y_train)
    all_pairs = all_p+all_n
    y_train_p = np.ones((len(all_p),))
    y_train_n = np.zeros((len(all_n),))
    new_y_train = list(y_train_p) + list(y_train_n)
    all_pairs,new_y_train = shuffle(all_pairs,new_y_train,random_state=0)
    return all_pairs,new_y_train

    


    
        
    
    