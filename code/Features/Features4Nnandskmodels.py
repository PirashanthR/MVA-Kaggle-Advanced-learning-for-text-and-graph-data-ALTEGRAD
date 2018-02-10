# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 13:42:17 2017

@author: Pirashanth
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import numpy as np

def create_features_w2v_for_nn(list_pairs_train,list_pairs_test,data_matrix,ids2ind):
    '''
    Create features for the locally connected neural network for keras based on w2vec only
    Return: (list) train input for neural network
            (list) test input for neural network
    '''
    
    X_train_1 = []
    X_train_2 = []
    for i in range(len(list_pairs_train)):
        q1 = list_pairs_train[i][0]
        q2 = list_pairs_train[i][1]
        X_train_1.append(data_matrix[ids2ind[q1],:].reshape((-1,1)))
        X_train_2.append(data_matrix[ids2ind[q2],:].reshape((-1,1)))
    X_train_1 = np.concatenate(X_train_1,axis=1)
    X_train_2 = np.concatenate(X_train_2,axis=1)
    X_train = [X_train_1.T,X_train_2.T]

    X_test_1 = []
    X_test_2 = []
    for i in range(len(list_pairs_test)):
        q1 = list_pairs_test[i][0]
        q2 = list_pairs_test[i][1]
        X_test_1.append(data_matrix[ids2ind[q1],:].reshape((-1,1)))
        X_test_2.append(data_matrix[ids2ind[q2],:].reshape((-1,1)))
    if (len(X_test_1)>0):
        X_test_1 = np.concatenate(X_test_1,axis=1)
        X_test_2 = np.concatenate(X_test_2,axis=1)
        X_test = [X_test_1.T,X_test_2.T]
    else:
        X_test = []
        
    return X_train,X_test
   
def create_features_lsa_for_sk(list_pairs_train,list_pairs_test\
                               ,ids2ind,texts,n_lsa=40):
    '''
    Create features for the simple models based on lsa embedding only
    Return: (np.array(nb_sample,2*nb_lsa)) train input for neural network
            (np.array(nb_sample,2*nb_lsa)) test input for neural network
    '''
    

    vec = TfidfVectorizer(strip_accents='unicode',max_features=None,analyzer='word',use_idf=1,smooth_idf=1)
    A = vec.fit_transform(texts.values())
    LSA_features= TruncatedSVD(n_components=n_lsa).fit_transform(A)
    
    X_train_1_LSA = []
    X_train_2_LSA = []
    for i in range(len(list_pairs_train)):
        q1 = list_pairs_train[i][0]
        q2 = list_pairs_train[i][1]
        X_train_1_LSA.append(LSA_features[ids2ind[q1],:].reshape((-1,1)))
        X_train_2_LSA.append(LSA_features[ids2ind[q2],:].reshape((-1,1)))    
    X_train_1_LSA = np.concatenate(X_train_1_LSA,axis=1)
    X_train_2_LSA = np.concatenate(X_train_2_LSA,axis=1)
    X_train = np.concatenate([X_train_1_LSA.T,X_train_2_LSA.T],axis=1)



    X_test_1_LSA = []
    X_test_2_LSA = []
    for i in range(len(list_pairs_test)):
        q1 = list_pairs_test[i][0]
        q2 = list_pairs_test[i][1]
        X_test_1_LSA.append(LSA_features[ids2ind[q1],:].reshape((-1,1)))
        X_test_2_LSA.append(LSA_features[ids2ind[q2],:].reshape((-1,1)))

    if len(X_test_1_LSA)>0:
        
        X_test_1_LSA = np.concatenate(X_test_1_LSA,axis=1)
        X_test_2_LSA = np.concatenate(X_test_2_LSA,axis=1)
        X_test = np.concatenate([X_test_1_LSA.T,X_test_2_LSA.T],axis=1) 
    else:
        X_test = []

    return X_train,X_test
