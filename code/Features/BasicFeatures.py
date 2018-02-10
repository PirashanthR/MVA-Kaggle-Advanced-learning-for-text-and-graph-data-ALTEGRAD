'''
BasicFeatures -- ALTEGRAD Challenge Fall 2017 -- RATNAMOGAN Pirashanth -- SAYEM Othmane

This file contains the function that allows to compute what we have called
'BasicFeatures'. It represents features (essentially NLP features) that
can be computed in few lines of code.
'''

#Import Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from scipy.spatial import distance

from fuzzywuzzy import fuzz
import numpy as np
import math
import nltk


ngrams = lambda a, n: list(zip(*[a[i:] for i in range(n)]))


def common_n_grams(text_1,text_2,window,stemming=True):
    '''
    Compute the number of common n grams between two strings of text
    Param: @text_1: (str) string 1 to use for the comparison
    @text_2: (str) string 2 to use for the comparison
    @window: (int) n of n grams
    @stemming: (bool) use stem or not
    '''
    if stemming:
        
        stemmer = nltk.stem.SnowballStemmer('english')
        tokens_stemmed = list()
        for token in text_1:
            tokens_stemmed.append(stemmer.stem(token))
        text_11 = tokens_stemmed
        
        tokens_stemmed = list()
        for token in text_2:
            tokens_stemmed.append(stemmer.stem(token))
        text_22 = tokens_stemmed
    else:
        text_11 = text_1
        text_22 = text_2
    
    adjust_window = min(window,len(text_1),len(text_2))
    n_gram_1 = ngrams(text_11,adjust_window)
    n_gram_2 = ngrams(text_22,adjust_window)
    
    nb_ng_shared = len(([i for i in n_gram_1 if i in n_gram_2]))
    
    return nb_ng_shared

def not_common_n_grams(text_1,text_2,window,stemming=True):
    '''
    Compute the number of non common n grams between two strings of text
    Param: @text_1: (str) string 1 to use for the comparison
    @text_2: (str) string 2 to use for the comparison
    @window: (int) n of n grams
    @stemming: (bool) use stem or not
    '''
    if stemming:
        stemmer = nltk.stem.SnowballStemmer('english')
        tokens_stemmed = list()
        for token in text_1:
            tokens_stemmed.append(stemmer.stem(token))
        text_11 = tokens_stemmed
        
        tokens_stemmed = list()
        for token in text_2:
            tokens_stemmed.append(stemmer.stem(token))
        text_22 = tokens_stemmed
    else:
        text_11 = text_1
        text_22 = text_2
    
    adjust_window = min(window,len(text_1),len(text_2))
    n_gram_1 = ngrams(text_11,adjust_window)
    n_gram_2 = ngrams(text_22,adjust_window)
    
    nb_ng_shared = len(([i for i in n_gram_1 if i in n_gram_2]))
    nb_ng_not_shared = len(n_gram_1)+len(n_gram_2) - 2*nb_ng_shared
    
    return nb_ng_not_shared

def unmatching(text_1,text_2,stemming=True):
    '''
    Compute the number of non common words between two strings
    Param: @text_1: (str) string 1 to use for the comparison
    @text_2: (str) string 2 to use for the comparison
    @stemming: (bool) use stem or not
    '''
    if stemming:
        
        stemmer = nltk.stem.SnowballStemmer('english')
        tokens_stemmed = list()
        for token in text_1:
            tokens_stemmed.append(stemmer.stem(token))
        text_11 = tokens_stemmed
        
        tokens_stemmed = list()
        for token in text_2:
            tokens_stemmed.append(stemmer.stem(token))
        text_22 = tokens_stemmed
    else:
        text_11 = text_1
        text_22 = text_2
    
    nb_wd__not_shared = len(([i for i in text_11 if i not in text_22]))
    nb_wd__not_shared = nb_wd__not_shared+len(([i for i in text_22 if i not in text_11]))

    return nb_wd__not_shared        


def matching(text_1,text_2,stemming=True):
    '''
    Compute the number of common words between two strings
    Param: @text_1: (str) string 1 to use for the comparison
    @text_2: (str) string 2 to use for the comparison
    @stemming: (bool) use stem or not
    '''
    if stemming:
        
        stemmer = nltk.stem.SnowballStemmer('english')
        tokens_stemmed = list()
        for token in text_1:
            tokens_stemmed.append(stemmer.stem(token))
        text_11 = tokens_stemmed
        
        tokens_stemmed = list()
        for token in text_2:
            tokens_stemmed.append(stemmer.stem(token))
        text_22 = tokens_stemmed
    else:
        text_11 = text_1
        text_22 = text_2
    
    nb_wd_shared = len(([i for i in text_11 if i in text_22]))
    return nb_wd_shared        

def dice(text_1,text_2,stemming=True):
    '''
    Compute the dice distance between two strings
    Param: @text_1: (str) string 1 to use for the comparison
    @text_2: (str) string 2 to use for the comparison
    @stemming: (bool) use stem or not
    '''
    if stemming:
        
        stemmer = nltk.stem.SnowballStemmer('english')
        tokens_stemmed = list()
        for token in text_1:
            tokens_stemmed.append(stemmer.stem(token))
        text_11 = tokens_stemmed
        
        tokens_stemmed = list()
        for token in text_2:
            tokens_stemmed.append(stemmer.stem(token))
        text_22 = tokens_stemmed
    else:
        text_11 = text_1
        text_22 = text_2
    
    nb_wd_shared = len(([i for i in text_11 if i in text_22]))

    nb_wd_1 = len(text_1)
    nb_wd_2 = len(text_2)
    return 2*nb_wd_shared/(nb_wd_1+nb_wd_2)

def jaccard(text_1,text_2,stemming=True):
    '''
    Compute the jaccard distance between two strings
    Param: @text_1: (str) string 1 to use for the comparison
    @text_2: (str) string 2 to use for the comparison
    @stemming: (bool) use stem or not
    '''
    if stemming:
        stemmer = nltk.stem.SnowballStemmer('english')
        tokens_stemmed = list()
        for token in text_1:
            tokens_stemmed.append(stemmer.stem(token))
        text_11 = tokens_stemmed
        
        tokens_stemmed = list()
        for token in text_2:
            tokens_stemmed.append(stemmer.stem(token))
        text_22 = tokens_stemmed
    else:
        text_11 = text_1
        text_22 = text_2
    
    nb_wd_shared = len(([i for i in text_11 if i in text_22]))
    nb_union_word = len(set(text_11+text_22))
    return nb_wd_shared/nb_union_word

def overlap(text_1,text_2,stemming=True):
    '''
    Compute the overlap distance between two strings
    Param: @text_1: (str) string 1 to use for the comparison
    @text_2: (str) string 2 to use for the comparison
    @stemming: (bool) use stem or not
    '''
    if stemming:
        stemmer = nltk.stem.SnowballStemmer('english')
        tokens_stemmed = list()
        for token in text_1:
            tokens_stemmed.append(stemmer.stem(token))
        text_11 = tokens_stemmed
        
        tokens_stemmed = list()
        for token in text_2:
            tokens_stemmed.append(stemmer.stem(token))
        text_22 = tokens_stemmed
    else:
        text_11 = text_1
        text_22 = text_2
    
    nb_wd_shared = len(([i for i in text_11 if i in text_22]))

    nb_wd_1 = len(text_1)
    nb_wd_2 = len(text_2)
    return nb_wd_shared/max(min(nb_wd_1,nb_wd_2),1)

def cosine_wd(text_1,text_2,stemming=True):
    '''
    Compute the cosine distance between two strings
    Param: @text_1: (str) string 1 to use for the comparison
    @text_2: (str) string 2 to use for the comparison
    @stemming: (bool) use stem or not
    '''
    if stemming:
        stemmer = nltk.stem.SnowballStemmer('english')
        tokens_stemmed = list()
        for token in text_1:
            tokens_stemmed.append(stemmer.stem(token))
        text_11 = tokens_stemmed
        
        tokens_stemmed = list()
        for token in text_2:
            tokens_stemmed.append(stemmer.stem(token))
        text_22 = tokens_stemmed
    else:
        text_11 = text_1
        text_22 = text_2
    
    nb_wd_shared = len(([i for i in text_11 if i in text_22]))

    nb_wd_1 = len(text_1)
    nb_wd_2 = len(text_2)
    return nb_wd_shared/max(math.sqrt(nb_wd_1*nb_wd_2),1)

def presence_of_why(text):
    return int('why' in text)

def presence_of_what(text):
    return int('what' in text)

def presence_of_where(text):
    return int('where' in text)

def presence_of_how(text):
    return int('how' in text)

def presence_of_when(text):
    return int('when' in text)

def nb_of_capital_letters(text):
    return sum(1 for c in text if c.isupper())

def nb_of_question_marks(text):
    return sum(1 for c in text if c=='?')

def is_first_word_same(text_1,text_2):
    '''
    Returns if the first words of two questions are the same
    Param: @text_1: (str) string 1 to use for the comparison
    @text_2: (str) string 2 to use for the comparison
    Return :(bool) 1 if true , 0 else
    '''
    stemmer = nltk.stem.SnowballStemmer('english')
    tokens_stemmed = list()
    for token in text_1:
        tokens_stemmed.append(stemmer.stem(token))
    text_11 = tokens_stemmed
    
    tokens_stemmed = list()
    for token in text_2:
        tokens_stemmed.append(stemmer.stem(token))
    text_22 = tokens_stemmed
    if (len(text_11)==0)or(len(text_22)==0):
        return 0
    else:
        return text_22[0]==text_11[0]

def is_last_word_same(text_1,text_2):
    '''
    Returns if the last word of two questions are the same
    Param: @text_1: (str) string 1 to use for the comparison
    @text_2: (str) string 2 to use for the comparison
    Return :(bool) 1 if true , 0 else
    '''
    stemmer = nltk.stem.SnowballStemmer('english')
    tokens_stemmed = list()
    for token in text_1:
        tokens_stemmed.append(stemmer.stem(token))
    text_11 = tokens_stemmed
    
    tokens_stemmed = list()
    for token in text_2:
        tokens_stemmed.append(stemmer.stem(token))
    text_22 = tokens_stemmed
    if (len(text_11)==0)or(len(text_22)==0):
        return 0
    else:
        return text_22[-1]==text_11[-1]


def create_all_simple_features(list_pairs_train,list_pairs_test,texts,\
                               ids2ind,word_vectors,embedding_matrix,glove_embedding,sequences\
                               ,data_full_word,my_p=50,n_lsa=40):
    '''
    Create the set of basic features from the list_pairs
    param: list_pairs_train: list of pairs of question to compare for training
            list_pars_test: list of pairs of question to compare for test phases
            ids2ind: index of each question
            word_vectors : w2vec based word_vectors (gensim object)
            embedding_matrix: w2vec based word matrix
            sequences: sequences index (keras object)
            n_lsa: number of axis used for pca used in lsa embedding
    return: matrix train (nb_sample,nb_features) 
            matrix test (nb_sample,nb_features)
    '''
    vec = TfidfVectorizer()
    A = vec.fit_transform(texts.values())
    LSA_features= TruncatedSVD(n_components=n_lsa).fit_transform(A)
    vec_count = CountVectorizer()
    B = vec_count.fit_transform(texts.values())
    LSA_bis_features= TruncatedSVD(n_components=n_lsa).fit_transform(B)
    
    stemmer = nltk.stem.SnowballStemmer('english')

    ###Stem doc
    documents = [[stemmer.stem(word) for word in sentence.split(" ")] for sentence in texts]
    documents = [' '.join(doc) for doc in documents]

    vec_stem = TfidfVectorizer()
    C = vec_stem.fit_transform(documents)
    LSA_features_stem= TruncatedSVD(n_components=n_lsa).fit_transform(C)

    vec_count_stem = CountVectorizer()
    D = vec_count_stem.fit_transform(documents)
    LSA_bis_features_stem= TruncatedSVD(n_components=n_lsa).fit_transform(D)
    
    #########Init###########
    N_train = len(list_pairs_train)
    N_test= len(list_pairs_test)
    X_train = np.zeros((N_train,135))
    X_test = np.zeros((N_test,135))
    
    cleaned_docs= data_full_word
    d2v_training_data = []
    
    for idx,doc in enumerate(cleaned_docs):
        d2v_training_data.append(LabeledSentence(words=doc,tags=[idx]))
        if idx % round(len(cleaned_docs)/10) == 0:
            print(idx)
    
    d2v_dm = Doc2Vec(d2v_training_data, 
                 size=200,
                 iter=6,
                 window=5, 
                 min_count=3, 
                 workers=4)
    
    d2v_dm.delete_temporary_training_data(keep_doctags_vectors=True, 
                                      keep_inference=True)

    d2v_dbow = Doc2Vec(d2v_training_data, 
                       size=my_p, 
                       window=4,
                       iter=6,
                       min_count=3, 
                       dm=0, 
                       workers=4)
    
    d2v_dbow.delete_temporary_training_data(keep_doctags_vectors=True, 
                                            keep_inference=True)
    #####Create features for training###########
    for i in range(N_train):
        q1 = list_pairs_train[i][0]
        q2 = list_pairs_train[i][1]
        
        X_train[i,0] = 1- cosine_similarity(A[ids2ind[q1],:], A[ids2ind[q2],:])
        X_train[i,2] = abs(len(texts[q1].split()) - len(texts[q2].split()))
        X_train[i,3] = min(word_vectors.wv.wmdistance((texts[q1].lower()).split(),(texts[q2].lower()).split()),100000) #WM distance
        X_train[i,4] = min(matching((texts[q1].lower()).split(),(texts[q2].lower()).split()),7)        
        X_train[i,5]= 1- cosine_similarity(LSA_features[ids2ind[q1],:].reshape(1, -1), LSA_features[ids2ind[q2],:].reshape(1, -1))
        if (len(sequences[ids2ind[q1]])>0)and(len(sequences[ids2ind[q2]])>0) :
            mean_pos_1 = (embedding_matrix[sequences[ids2ind[q1]],:]).sum(axis=0)
            mean_pos_2 = (embedding_matrix[sequences[ids2ind[q2]],:]).sum(axis=0)
            mean_pos_1= mean_pos_1 / np.sqrt((mean_pos_1 ** 2).sum())
            mean_pos_2= mean_pos_2 / np.sqrt((mean_pos_2 ** 2).sum())
            X_train[i,1] = 1- cosine_similarity(mean_pos_1.reshape(1, -1),mean_pos_2.reshape(1, -1))
            mean_pos_1_gv = (glove_embedding[sequences[ids2ind[q1]],:]).sum(axis=0)
            mean_pos_2_gv = (glove_embedding[sequences[ids2ind[q2]],:]).sum(axis=0)
            if np.sum(mean_pos_1_gv)>0:
                mean_pos_1_gv= mean_pos_1_gv / np.sqrt((mean_pos_1_gv ** 2).sum())
            if np.sum(mean_pos_2_gv)>0:
                mean_pos_2_gv= mean_pos_2_gv / np.sqrt((mean_pos_2_gv ** 2).sum())
            X_train[i,15] = 1- cosine_similarity(mean_pos_1_gv.reshape(1, -1),mean_pos_2_gv.reshape(1, -1))
            X_train[i,17] = np.linalg.norm(mean_pos_1.reshape(1, -1)-mean_pos_2.reshape(1, -1))
            X_train[i,18] = np.linalg.norm(mean_pos_1_gv.reshape(1, -1)-mean_pos_2_gv.reshape(1, -1))
            X_train[i,35]= distance.cityblock(mean_pos_1,mean_pos_2)
            X_train[i,36]= distance.jaccard(mean_pos_1,mean_pos_2)
            X_train[i,37]= distance.canberra(mean_pos_1,mean_pos_2)
            X_train[i,38]= distance.minkowski(mean_pos_1,mean_pos_2,3)
            X_train[i,39]= distance.braycurtis(mean_pos_1,mean_pos_2)
            X_train[i,40]= distance.cityblock(mean_pos_1_gv,mean_pos_2_gv)
            X_train[i,41]= distance.jaccard(mean_pos_1_gv,mean_pos_2_gv)
            X_train[i,42]= distance.canberra(mean_pos_1_gv,mean_pos_2_gv)
            X_train[i,43]= distance.minkowski(mean_pos_1_gv,mean_pos_2_gv,3)
            X_train[i,44]= distance.braycurtis(mean_pos_1_gv,mean_pos_2_gv)
        else:
            X_train[i,1] = -1
            X_train[i,15]= -1
            X_train[i,17] = -1
            X_train[i,18] = -1
            X_train[i,35:44]=-1
            
        X_train[i,6] = fuzz.partial_ratio(texts[q1],texts[q2])/100
        X_train[i,7] = fuzz.QRatio(texts[q1],texts[q2])/100
        X_train[i,8] = 1 - cosine_similarity(LSA_bis_features[ids2ind[q1],:].reshape(1, -1), LSA_bis_features[ids2ind[q2],:].reshape(1, -1))
    
        d2v1 = d2v_dm.infer_vector(cleaned_docs[ids2ind[q1]])  
        d2v2 = d2v_dm.infer_vector(cleaned_docs[ids2ind[q2]])  
        
        d2vbow1 = d2v_dbow.infer_vector(cleaned_docs[ids2ind[q1]])  
        d2vbow2 = d2v_dbow.infer_vector(cleaned_docs[ids2ind[q2]])
        
        X_train[i,9] = 1 - cosine_similarity(d2vbow1.reshape(1, -1), d2vbow2.reshape(1, -1))
        X_train[i,10] = 1 - cosine_similarity(d2v1.reshape(1, -1), d2v2.reshape(1, -1))
        X_train[i,11] = dice((texts[q1].lower()).split(),(texts[q2].lower()).split())
        X_train[i,12] = jaccard((texts[q1].lower()).split(),(texts[q2].lower()).split())
        X_train[i,13] = overlap((texts[q1].lower()).split(),(texts[q2].lower()).split())
        X_train[i,14] = cosine_wd((texts[q1].lower()).split(),(texts[q2].lower()).split())
        X_train[i,16]= np.linalg.norm(LSA_features[ids2ind[q1],:].reshape(1, -1) - LSA_features[ids2ind[q2],:].reshape(1, -1))
        X_train[i,19] = np.linalg.norm(LSA_bis_features[ids2ind[q1],:].reshape(1, -1)- LSA_bis_features[ids2ind[q2],:].reshape(1, -1))
        X_train[i,20] = np.linalg.norm(d2vbow1.reshape(1, -1) -  d2vbow2.reshape(1, -1))
        X_train[i,21] = np.linalg.norm(d2v1.reshape(1, -1) -  d2v2.reshape(1, -1))
        X_train[i,22] = np.linalg.norm(A[ids2ind[q1],:].todense() - A[ids2ind[q2],:].todense())
        X_train[i,23]=  max(min(presence_of_why(texts[q1].lower()),1),min(presence_of_why(texts[q2].lower()),1))
        X_train[i,24]= max(min(presence_of_what(texts[q1].lower()),1),min(presence_of_what(texts[q2].lower()),1))
        X_train[i,25]= max(min(presence_of_when(texts[q1].lower()),1),min(presence_of_when(texts[q2].lower()),1))
        X_train[i,26]= max(min(presence_of_where(texts[q1].lower()),1),min(presence_of_where(texts[q2].lower()),1))
        X_train[i,27]= max(min(presence_of_how(texts[q1].lower()),1),min(presence_of_how(texts[q2].lower()),1))
        X_train[i,28]= min(min(presence_of_why(texts[q1].lower()),1),min(presence_of_why(texts[q2].lower()),1))
        X_train[i,29]= min(min(presence_of_what(texts[q1].lower()),1),min(presence_of_what(texts[q2].lower()),1))
        X_train[i,30]= min(min(presence_of_when(texts[q1].lower()),1),min(presence_of_when(texts[q2].lower()),1))
        X_train[i,31]= min(min(presence_of_where(texts[q1].lower()),1),min(presence_of_where(texts[q2].lower()),1))
        X_train[i,32]=  min(min(presence_of_how(texts[q1].lower()),1),min(presence_of_how(texts[q2].lower()),1))
        X_train[i,33]= fuzz.token_set_ratio(texts[q1],texts[q2])/100
        X_train[i,34]= fuzz.token_sort_ratio(texts[q1],texts[q2])/100
        X_train[i,45] = abs(len(texts[q1].lower())-len(texts[q2].lower()))
        X_train[i,46] = abs(len([j for j in texts[q1] if j=='?'])-len([j for j in texts[q2] if j=='?']))
        X_train[i,47] = len(texts[q1].split()) + len(texts[q2].split())
        X_train[i,48] = distance.cityblock(d2v1.reshape(1, -1),d2v2.reshape(1, -1))
        X_train[i,49] = distance.jaccard(d2v1.reshape(1, -1),d2v2.reshape(1, -1))
        X_train[i,50] = distance.canberra(d2v1.reshape(1, -1),d2v2.reshape(1, -1))
        X_train[i,51] = distance.minkowski(d2v1.reshape(1, -1),d2v2.reshape(1, -1),3)
        X_train[i,52] = distance.braycurtis(d2v1.reshape(1, -1),d2v2.reshape(1, -1))
        X_train[i,53] = distance.cityblock(LSA_features[ids2ind[q1],:].reshape(1, -1),LSA_features[ids2ind[q2],:].reshape(1, -1))
        X_train[i,54] = distance.jaccard(LSA_features[ids2ind[q1],:].reshape(1, -1),LSA_features[ids2ind[q2],:].reshape(1, -1))
        X_train[i,55] = distance.canberra(LSA_features[ids2ind[q1],:].reshape(1, -1),LSA_features[ids2ind[q2],:].reshape(1, -1))
        X_train[i,56] = distance.minkowski(LSA_features[ids2ind[q1],:].reshape(1, -1),LSA_features[ids2ind[q2],:].reshape(1, -1),3)
        X_train[i,57] = distance.braycurtis(LSA_features[ids2ind[q1],:].reshape(1, -1),LSA_features[ids2ind[q2],:].reshape(1, -1))
        X_train[i,58] = unmatching((texts[q1].lower()).split(),(texts[q2].lower()).split())
        X_train[i,59] = is_first_word_same((texts[q1].lower()).split(),(texts[q2].lower()).split())
        X_train[i,60] = is_last_word_same((texts[q1].lower()).split(),(texts[q2].lower()).split())
        
    
        #####Using QID
        X_train[i,61] = abs(int(q1) - int(q2))
        X_train[i,62] = abs((int(q1) + int(q2))/2)
        X_train[i,63] = abs(min(int(q1),int(q2)))

        ####Using N-grams
        X_train[i,64] = common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),2)
        X_train[i,65] = common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),3)
        X_train[i,66] = common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),4)
        X_train[i,67] = common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),5)
        X_train[i,68] = common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),6)
        X_train[i,69] = not_common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),2)
        X_train[i,70] = not_common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),3)
        X_train[i,71] = not_common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),4)
        X_train[i,72] = not_common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),5)
        X_train[i,73] = not_common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),6)

        X_train[i,74] = dice((texts[q1].lower()).split(),(texts[q2].lower()).split(),stemming=False)
        X_train[i,75] = jaccard((texts[q1].lower()).split(),(texts[q2].lower()).split(),stemming=False)
        X_train[i,76] = overlap((texts[q1].lower()).split(),(texts[q2].lower()).split(),stemming=False)
        X_train[i,77] = cosine_wd((texts[q1].lower()).split(),(texts[q2].lower()).split(),stemming=False)
        X_train[i,78] = min(matching((texts[q1].lower()).split(),(texts[q2].lower()).split(),stemming=False),7)
        X_train[i,79] = unmatching((texts[q1].lower()).split(),(texts[q2].lower()).split(),stemming=False)
        X_train[i,80] = common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),2,stemming=False)
        X_train[i,81] = common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),3,stemming=False)
        X_train[i,82] = common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),4,stemming=False)
        X_train[i,83] = common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),5,stemming=False)
        X_train[i,84] = common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),6,stemming=False)
        X_train[i,85] = not_common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),2,stemming=False)
        X_train[i,86] = not_common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),3,stemming=False)
        X_train[i,87] = not_common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),4,stemming=False)
        X_train[i,88] = not_common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),5,stemming=False)
        X_train[i,89] = not_common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),6,stemming=False)

    
        X_train[i,90] = distance.cityblock(LSA_bis_features[ids2ind[q1],:].reshape(1, -1),LSA_bis_features[ids2ind[q2],:].reshape(1, -1))
        X_train[i,91] = distance.jaccard(LSA_bis_features[ids2ind[q1],:].reshape(1, -1),LSA_bis_features[ids2ind[q2],:].reshape(1, -1))
        X_train[i,92] = distance.canberra(LSA_bis_features[ids2ind[q1],:].reshape(1, -1),LSA_bis_features[ids2ind[q2],:].reshape(1, -1))
        X_train[i,93] = distance.minkowski(LSA_bis_features[ids2ind[q1],:].reshape(1, -1),LSA_bis_features[ids2ind[q2],:].reshape(1, -1),3)
        X_train[i,94] = distance.braycurtis(LSA_bis_features[ids2ind[q1],:].reshape(1, -1),LSA_bis_features[ids2ind[q2],:].reshape(1, -1))
                
        
        X_train[i,95] = distance.cityblock(A[ids2ind[q1],:].todense().reshape(1, -1),A[ids2ind[q2],:].todense().reshape(1, -1))
        X_train[i,96] = distance.jaccard(A[ids2ind[q1],:].todense().reshape(1, -1),A[ids2ind[q2],:].todense().reshape(1, -1))
        X_train[i,97] = distance.canberra(A[ids2ind[q1],:].todense().reshape(1, -1),A[ids2ind[q2],:].todense().reshape(1, -1))
        X_train[i,98] = distance.minkowski(A[ids2ind[q1],:].todense().reshape(1, -1),A[ids2ind[q2],:].todense().reshape(1, -1),3)
        X_train[i,99] = distance.braycurtis(A[ids2ind[q1],:].todense().reshape(1, -1),A[ids2ind[q2],:].todense().reshape(1, -1))
        
        
        X_train[i,100] = 1- cosine_similarity(B[ids2ind[q1],:], B[ids2ind[q2],:])
        X_train[i,101] = np.linalg.norm(B[ids2ind[q1],:].todense() - B[ids2ind[q2],:].todense())
        X_train[i,102] = distance.cityblock(B[ids2ind[q1],:].todense().reshape(1, -1),B[ids2ind[q2],:].todense().reshape(1, -1))
        X_train[i,103] = distance.jaccard(B[ids2ind[q1],:].todense().reshape(1, -1),B[ids2ind[q2],:].todense().reshape(1, -1))
        X_train[i,104] = distance.canberra(B[ids2ind[q1],:].todense().reshape(1, -1),B[ids2ind[q2],:].todense().reshape(1, -1))
        X_train[i,105] = distance.minkowski(B[ids2ind[q1],:].todense().reshape(1, -1),B[ids2ind[q2],:].todense().reshape(1, -1),3)
        X_train[i,106] = distance.braycurtis(B[ids2ind[q1],:].todense().reshape(1, -1),B[ids2ind[q2],:].todense().reshape(1, -1))
        
        X_train[i,107] = 1- cosine_similarity(C[ids2ind[q1],:], C[ids2ind[q2],:])
        X_train[i,108] = np.linalg.norm(C[ids2ind[q1],:].todense() - C[ids2ind[q2],:].todense())
        X_train[i,109] = distance.cityblock(C[ids2ind[q1],:].todense().reshape(1, -1),C[ids2ind[q2],:].todense().reshape(1, -1))
        X_train[i,110] = distance.jaccard(C[ids2ind[q1],:].todense().reshape(1, -1),C[ids2ind[q2],:].todense().reshape(1, -1))
        X_train[i,111] = distance.canberra(C[ids2ind[q1],:].todense().reshape(1, -1),C[ids2ind[q2],:].todense().reshape(1, -1))
        X_train[i,112] = distance.minkowski(C[ids2ind[q1],:].todense().reshape(1, -1),C[ids2ind[q2],:].todense().reshape(1, -1),3)
        X_train[i,113] = distance.braycurtis(C[ids2ind[q1],:].todense().reshape(1, -1),C[ids2ind[q2],:].todense().reshape(1, -1))
        
        X_train[i,114] = 1- cosine_similarity(D[ids2ind[q1],:], D[ids2ind[q2],:])
        X_train[i,115] = np.linalg.norm(D[ids2ind[q1],:].todense() - D[ids2ind[q2],:].todense())
        X_train[i,116] = distance.cityblock(D[ids2ind[q1],:].todense().reshape(1, -1),D[ids2ind[q2],:].todense().reshape(1, -1))
        X_train[i,117] = distance.jaccard(D[ids2ind[q1],:].todense().reshape(1, -1),D[ids2ind[q2],:].todense().reshape(1, -1))
        X_train[i,118] = distance.canberra(D[ids2ind[q1],:].todense().reshape(1, -1),D[ids2ind[q2],:].todense().reshape(1, -1))
        X_train[i,119] = distance.minkowski(D[ids2ind[q1],:].todense().reshape(1, -1),D[ids2ind[q2],:].todense().reshape(1, -1),3)
        X_train[i,120] = distance.braycurtis(D[ids2ind[q1],:].todense().reshape(1, -1),D[ids2ind[q2],:].todense().reshape(1, -1))
        
        X_train[i,121] = 1 - cosine_similarity(LSA_features_stem[ids2ind[q1],:].reshape(1, -1), LSA_features_stem[ids2ind[q2],:].reshape(1, -1))
        X_train[i,122] = np.linalg.norm(LSA_features_stem[ids2ind[q1],:].reshape(1, -1)- LSA_features_stem[ids2ind[q2],:].reshape(1, -1))
        X_train[i,123] = distance.cityblock(LSA_features_stem[ids2ind[q1],:].reshape(1, -1),LSA_features_stem[ids2ind[q2],:].reshape(1, -1))
        X_train[i,124] = distance.jaccard(LSA_features_stem[ids2ind[q1],:].reshape(1, -1),LSA_features_stem[ids2ind[q2],:].reshape(1, -1))
        X_train[i,125] = distance.canberra(LSA_features_stem[ids2ind[q1],:].reshape(1, -1),LSA_features_stem[ids2ind[q2],:].reshape(1, -1))
        X_train[i,126] = distance.minkowski(LSA_features_stem[ids2ind[q1],:].reshape(1, -1),LSA_features_stem[ids2ind[q2],:].reshape(1, -1),3)
        X_train[i,127] = distance.braycurtis(LSA_features_stem[ids2ind[q1],:].reshape(1, -1),LSA_features_stem[ids2ind[q2],:].reshape(1, -1))

        X_train[i,128] = 1 - cosine_similarity(LSA_bis_features_stem[ids2ind[q1],:].reshape(1, -1), LSA_bis_features_stem[ids2ind[q2],:].reshape(1, -1))
        X_train[i,129] = np.linalg.norm(LSA_bis_features_stem[ids2ind[q1],:].reshape(1, -1)- LSA_bis_features_stem[ids2ind[q2],:].reshape(1, -1))
        X_train[i,130] = distance.cityblock(LSA_bis_features_stem[ids2ind[q1],:].reshape(1, -1),LSA_bis_features_stem[ids2ind[q2],:].reshape(1, -1))
        X_train[i,131] = distance.jaccard(LSA_bis_features_stem[ids2ind[q1],:].reshape(1, -1),LSA_bis_features_stem[ids2ind[q2],:].reshape(1, -1))
        X_train[i,132] = distance.canberra(LSA_bis_features_stem[ids2ind[q1],:].reshape(1, -1),LSA_bis_features_stem[ids2ind[q2],:].reshape(1, -1))
        X_train[i,133] = distance.minkowski(LSA_bis_features_stem[ids2ind[q1],:].reshape(1, -1),LSA_bis_features_stem[ids2ind[q2],:].reshape(1, -1),3)
        X_train[i,134] = distance.braycurtis(LSA_bis_features_stem[ids2ind[q1],:].reshape(1, -1),LSA_bis_features_stem[ids2ind[q2],:].reshape(1, -1))



        
    #####Create features for test###########
    for i in range(N_test):
        q1 = list_pairs_test[i][0]
        q2 = list_pairs_test[i][1]
        X_test[i,0] = 1- cosine_similarity(A[ids2ind[q1],:], A[ids2ind[q2],:])
        X_test[i,2] = abs(len(texts[q1].split()) - len(texts[q2].split()))
        X_test[i,3] = min(word_vectors.wv.wmdistance((texts[q1].lower()).split(),(texts[q2].lower()).split()),100000) #WM distance
        X_test[i,4] = min(matching((texts[q1].lower()).split(),(texts[q2].lower()).split()),7)
        X_test[i,5]= 1- cosine_similarity(LSA_features[ids2ind[q1],:].reshape(1, -1), LSA_features[ids2ind[q2],:].reshape(1, -1))
    
        if (len(sequences[ids2ind[q1]])>0)and(len(sequences[ids2ind[q2]])>0) :
            mean_pos_1 = (embedding_matrix[sequences[ids2ind[q1]],:]).sum(axis=0)
            mean_pos_2 = (embedding_matrix[sequences[ids2ind[q2]],:]).sum(axis=0)
            mean_pos_1= mean_pos_1 / np.sqrt((mean_pos_1 ** 2).sum())
            mean_pos_2= mean_pos_2 / np.sqrt((mean_pos_2 ** 2).sum())
            X_test[i,1] = 1- cosine_similarity(mean_pos_1.reshape(1, -1),mean_pos_2.reshape(1, -1))
            mean_pos_1_gv = (glove_embedding[sequences[ids2ind[q1]],:]).sum(axis=0)
            mean_pos_2_gv = (glove_embedding[sequences[ids2ind[q2]],:]).sum(axis=0)
            if np.sum(mean_pos_1_gv)!=0:
                mean_pos_1_gv= mean_pos_1_gv / np.sqrt((mean_pos_1_gv ** 2).sum())
            if np.sum(mean_pos_2_gv)!=0:
                mean_pos_2_gv= mean_pos_2_gv / np.sqrt((mean_pos_2_gv ** 2).sum())
            X_test[i,15] = 1- cosine_similarity(mean_pos_1_gv.reshape(1, -1),mean_pos_2_gv.reshape(1, -1))
            X_test[i,17] = np.linalg.norm(mean_pos_1.reshape(1, -1)-mean_pos_2.reshape(1, -1))
            X_test[i,18] = np.linalg.norm(mean_pos_1_gv.reshape(1, -1)-mean_pos_2_gv.reshape(1, -1))
            X_test[i,35]= distance.cityblock(mean_pos_1,mean_pos_2)
            X_test[i,36]= distance.jaccard(mean_pos_1,mean_pos_2)
            X_test[i,37]= distance.canberra(mean_pos_1,mean_pos_2)
            X_test[i,38]= distance.minkowski(mean_pos_1,mean_pos_2,3)
            X_test[i,39]= distance.braycurtis(mean_pos_1,mean_pos_2)
            X_test[i,40]= distance.cityblock(mean_pos_1_gv,mean_pos_2_gv)
            X_test[i,41]= distance.jaccard(mean_pos_1_gv,mean_pos_2_gv)
            X_test[i,42]= distance.canberra(mean_pos_1_gv,mean_pos_2_gv)
            X_test[i,43]= distance.minkowski(mean_pos_1_gv,mean_pos_2_gv,3)
            X_test[i,44]= distance.braycurtis(mean_pos_1_gv,mean_pos_2_gv)
        else:
            X_test[i,1] = -1
            X_test[i,15]= -1
            X_test[i,17] = -1
            X_test[i,18] = -1 
            X_test[i,35:44]=-1
        X_test[i,6] = fuzz.partial_ratio(texts[q1],texts[q2])/100
        X_test[i,7] = fuzz.QRatio(texts[q1],texts[q2])/100
        X_test[i,8] = 1 - cosine_similarity(LSA_bis_features[ids2ind[q1],:].reshape(1, -1), LSA_bis_features[ids2ind[q2],:].reshape(1, -1))
        
        d2v1 = d2v_dm.infer_vector(cleaned_docs[ids2ind[q1]])  
        d2v2 = d2v_dm.infer_vector(cleaned_docs[ids2ind[q2]])  
        
        d2vbow1 = d2v_dbow.infer_vector(cleaned_docs[ids2ind[q1]])  
        d2vbow2 = d2v_dbow.infer_vector(cleaned_docs[ids2ind[q2]])
        
        X_test[i,9] = 1 - cosine_similarity(d2vbow1.reshape(1, -1), d2vbow2.reshape(1, -1))
        X_test[i,10] = 1 - cosine_similarity(d2v1.reshape(1, -1), d2v2.reshape(1, -1))
        X_test[i,11] = dice((texts[q1].lower()).split(),(texts[q2].lower()).split())
        X_test[i,12] = jaccard((texts[q1].lower()).split(),(texts[q2].lower()).split())
        X_test[i,13] = overlap((texts[q1].lower()).split(),(texts[q2].lower()).split())
        X_test[i,14] = cosine_wd((texts[q1].lower()).split(),(texts[q2].lower()).split())
        X_test[i,16]= np.linalg.norm(LSA_features[ids2ind[q1],:].reshape(1, -1) - LSA_features[ids2ind[q2],:].reshape(1, -1))
        X_test[i,19] = np.linalg.norm(LSA_bis_features[ids2ind[q1],:].reshape(1, -1)- LSA_bis_features[ids2ind[q2],:].reshape(1, -1))
        X_test[i,20] = np.linalg.norm(d2vbow1.reshape(1, -1) -  d2vbow2.reshape(1, -1))
        X_test[i,21] = np.linalg.norm(d2v1.reshape(1, -1) -  d2v2.reshape(1, -1))
        X_test[i,22] = np.linalg.norm(A[ids2ind[q1],:].todense() - A[ids2ind[q2],:].todense())
        X_test[i,23]=  max(min(presence_of_why(texts[q1].lower()),1),min(presence_of_why(texts[q2].lower()),1))
        X_test[i,24]= max(min(presence_of_what(texts[q1].lower()),1),min(presence_of_what(texts[q2].lower()),1))
        X_test[i,25]= max(min(presence_of_when(texts[q1].lower()),1),min(presence_of_when(texts[q2].lower()),1))
        X_test[i,26]= max(min(presence_of_where(texts[q1].lower()),1),min(presence_of_where(texts[q2].lower()),1))
        X_test[i,27]= max(min(presence_of_how(texts[q1].lower()),1),min(presence_of_how(texts[q2].lower()),1))
        X_test[i,28]= min(min(presence_of_why(texts[q1].lower()),1),min(presence_of_why(texts[q2].lower()),1))
        X_test[i,29]= min(min(presence_of_what(texts[q1].lower()),1),min(presence_of_what(texts[q2].lower()),1))
        X_test[i,30]= min(min(presence_of_when(texts[q1].lower()),1),min(presence_of_when(texts[q2].lower()),1))
        X_test[i,31]= min(min(presence_of_where(texts[q1].lower()),1),min(presence_of_where(texts[q2].lower()),1))
        X_test[i,32]=  min(min(presence_of_how(texts[q1].lower()),1),min(presence_of_how(texts[q2].lower()),1))
        X_test[i,33]= fuzz.token_set_ratio(texts[q1],texts[q2])/100
        X_test[i,34]= fuzz.token_sort_ratio(texts[q1],texts[q2])/100
        X_test[i,45] = abs(len(texts[q1].lower())-len(texts[q2].lower()))
        X_test[i,46] = abs(len([j for j in texts[q1] if j=='?'])-len([j for j in texts[q2] if j=='?']))
        X_test[i,47] = len(texts[q1].split()) + len(texts[q2].split())
        X_test[i,48] = distance.cityblock(d2v1.reshape(1, -1),d2v2.reshape(1, -1))
        X_test[i,49] = distance.jaccard(d2v1.reshape(1, -1),d2v2.reshape(1, -1))
        X_test[i,50] = distance.canberra(d2v1.reshape(1, -1),d2v2.reshape(1, -1))
        X_test[i,51] = distance.minkowski(d2v1.reshape(1, -1),d2v2.reshape(1, -1),3)
        X_test[i,52] = distance.braycurtis(d2v1.reshape(1, -1),d2v2.reshape(1, -1))
        X_test[i,53] = distance.cityblock(LSA_features[ids2ind[q1],:].reshape(1, -1),LSA_features[ids2ind[q2],:].reshape(1, -1))
        X_test[i,54] = distance.jaccard(LSA_features[ids2ind[q1],:].reshape(1, -1),LSA_features[ids2ind[q2],:].reshape(1, -1))
        X_test[i,55] = distance.canberra(LSA_features[ids2ind[q1],:].reshape(1, -1),LSA_features[ids2ind[q2],:].reshape(1, -1))
        X_test[i,56] = distance.minkowski(LSA_features[ids2ind[q1],:].reshape(1, -1),LSA_features[ids2ind[q2],:].reshape(1, -1),3)
        X_test[i,57] = distance.braycurtis(LSA_features[ids2ind[q1],:].reshape(1, -1),LSA_features[ids2ind[q2],:].reshape(1, -1))
        X_test[i,58] = unmatching((texts[q1].lower()).split(),(texts[q2].lower()).split())
        X_test[i,59] = is_first_word_same((texts[q1].lower()).split(),(texts[q2].lower()).split())
        X_test[i,60] = is_last_word_same((texts[q1].lower()).split(),(texts[q2].lower()).split())
        X_test[i,61] = abs(int(q1) - int(q2))
        X_test[i,62] = abs((int(q1) + int(q2))/2)
        X_test[i,63] = abs(min(int(q1),int(q2)))
        X_test[i,64] = common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),2)
        X_test[i,65] = common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),3)
        X_test[i,66] = common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),4)
        X_test[i,67] = common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),5)
        X_test[i,68] = common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),6)
        X_test[i,69] = not_common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),2)
        X_test[i,70] = not_common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),3)
        X_test[i,71] = not_common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),4)
        X_test[i,72] = not_common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),5)
        X_test[i,73] = not_common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),6)

        X_test[i,74] = dice((texts[q1].lower()).split(),(texts[q2].lower()).split(),stemming=False)
        X_test[i,75] = jaccard((texts[q1].lower()).split(),(texts[q2].lower()).split(),stemming=False)
        X_test[i,76] = overlap((texts[q1].lower()).split(),(texts[q2].lower()).split(),stemming=False)
        X_test[i,77] = cosine_wd((texts[q1].lower()).split(),(texts[q2].lower()).split(),stemming=False)
        X_test[i,78] = min(matching((texts[q1].lower()).split(),(texts[q2].lower()).split(),stemming=False),7)
        X_test[i,79] = unmatching((texts[q1].lower()).split(),(texts[q2].lower()).split(),stemming=False)
        X_test[i,80] = common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),2,stemming=False)
        X_test[i,81] = common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),3,stemming=False)
        X_test[i,82] = common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),4,stemming=False)
        X_test[i,83] = common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),5,stemming=False)
        X_test[i,84] = common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),6,stemming=False)
        X_test[i,85] = not_common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),2,stemming=False)
        X_test[i,86] = not_common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),3,stemming=False)
        X_test[i,87] = not_common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),4,stemming=False)
        X_test[i,88] = not_common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),5,stemming=False)
        X_test[i,89] = not_common_n_grams((texts[q1].lower()).split(),(texts[q2].lower()).split(),6,stemming=False)


        X_test[i,90] = distance.cityblock(LSA_bis_features[ids2ind[q1],:].reshape(1, -1),LSA_bis_features[ids2ind[q2],:].reshape(1, -1))
        X_test[i,91] = distance.jaccard(LSA_bis_features[ids2ind[q1],:].reshape(1, -1),LSA_bis_features[ids2ind[q2],:].reshape(1, -1))
        X_test[i,92] = distance.canberra(LSA_bis_features[ids2ind[q1],:].reshape(1, -1),LSA_bis_features[ids2ind[q2],:].reshape(1, -1))
        X_test[i,93] = distance.minkowski(LSA_bis_features[ids2ind[q1],:].reshape(1, -1),LSA_bis_features[ids2ind[q2],:].reshape(1, -1),3)
        X_test[i,94] = distance.braycurtis(LSA_bis_features[ids2ind[q1],:].reshape(1, -1),LSA_bis_features[ids2ind[q2],:].reshape(1, -1))
                
        
        X_test[i,95] = distance.cityblock(A[ids2ind[q1],:].todense().reshape(1, -1),A[ids2ind[q2],:].todense().reshape(1, -1))
        X_test[i,96] = distance.jaccard(A[ids2ind[q1],:].todense().reshape(1, -1),A[ids2ind[q2],:].todense().reshape(1, -1))
        X_test[i,97] = distance.canberra(A[ids2ind[q1],:].todense().reshape(1, -1),A[ids2ind[q2],:].todense().reshape(1, -1))
        X_test[i,98] = distance.minkowski(A[ids2ind[q1],:].todense().reshape(1, -1),A[ids2ind[q2],:].todense().reshape(1, -1),3)
        X_test[i,99] = distance.braycurtis(A[ids2ind[q1],:].todense().reshape(1, -1),A[ids2ind[q2],:].todense().reshape(1, -1))
        
        
        X_test[i,100] = 1- cosine_similarity(B[ids2ind[q1],:], B[ids2ind[q2],:])
        X_test[i,101] = np.linalg.norm(B[ids2ind[q1],:].todense() - B[ids2ind[q2],:].todense())
        X_test[i,102] = distance.cityblock(B[ids2ind[q1],:].todense().reshape(1, -1),B[ids2ind[q2],:].todense().reshape(1, -1))
        X_test[i,103] = distance.jaccard(B[ids2ind[q1],:].todense().reshape(1, -1),B[ids2ind[q2],:].todense().reshape(1, -1))
        X_test[i,104] = distance.canberra(B[ids2ind[q1],:].todense().reshape(1, -1),B[ids2ind[q2],:].todense().reshape(1, -1))
        X_test[i,105] = distance.minkowski(B[ids2ind[q1],:].todense().reshape(1, -1),B[ids2ind[q2],:].todense().reshape(1, -1),3)
        X_test[i,106] = distance.braycurtis(B[ids2ind[q1],:].todense().reshape(1, -1),B[ids2ind[q2],:].todense().reshape(1, -1))
        
        X_test[i,107] = 1- cosine_similarity(C[ids2ind[q1],:], C[ids2ind[q2],:])
        X_test[i,108] = np.linalg.norm(C[ids2ind[q1],:].todense() - C[ids2ind[q2],:].todense())
        X_test[i,109] = distance.cityblock(C[ids2ind[q1],:].todense().reshape(1, -1),C[ids2ind[q2],:].todense().reshape(1, -1))
        X_test[i,110] = distance.jaccard(C[ids2ind[q1],:].todense().reshape(1, -1),C[ids2ind[q2],:].todense().reshape(1, -1))
        X_test[i,111] = distance.canberra(C[ids2ind[q1],:].todense().reshape(1, -1),C[ids2ind[q2],:].todense().reshape(1, -1))
        X_test[i,112] = distance.minkowski(C[ids2ind[q1],:].todense().reshape(1, -1),C[ids2ind[q2],:].todense().reshape(1, -1),3)
        X_test[i,113] = distance.braycurtis(C[ids2ind[q1],:].todense().reshape(1, -1),C[ids2ind[q2],:].todense().reshape(1, -1))
        
        X_test[i,114] = 1- cosine_similarity(D[ids2ind[q1],:], D[ids2ind[q2],:])
        X_test[i,115] = np.linalg.norm(D[ids2ind[q1],:].todense() - D[ids2ind[q2],:].todense())
        X_test[i,116] = distance.cityblock(D[ids2ind[q1],:].todense().reshape(1, -1),D[ids2ind[q2],:].todense().reshape(1, -1))
        X_test[i,117] = distance.jaccard(D[ids2ind[q1],:].todense().reshape(1, -1),D[ids2ind[q2],:].todense().reshape(1, -1))
        X_test[i,118] = distance.canberra(D[ids2ind[q1],:].todense().reshape(1, -1),D[ids2ind[q2],:].todense().reshape(1, -1))
        X_test[i,119] = distance.minkowski(D[ids2ind[q1],:].todense().reshape(1, -1),D[ids2ind[q2],:].todense().reshape(1, -1),3)
        X_test[i,120] = distance.braycurtis(D[ids2ind[q1],:].todense().reshape(1, -1),D[ids2ind[q2],:].todense().reshape(1, -1))
        
        X_test[i,121] = 1 - cosine_similarity(LSA_features_stem[ids2ind[q1],:].reshape(1, -1), LSA_features_stem[ids2ind[q2],:].reshape(1, -1))
        X_test[i,122] = np.linalg.norm(LSA_features_stem[ids2ind[q1],:].reshape(1, -1)- LSA_features_stem[ids2ind[q2],:].reshape(1, -1))
        X_test[i,123] = distance.cityblock(LSA_features_stem[ids2ind[q1],:].reshape(1, -1),LSA_features_stem[ids2ind[q2],:].reshape(1, -1))
        X_test[i,124] = distance.jaccard(LSA_features_stem[ids2ind[q1],:].reshape(1, -1),LSA_features_stem[ids2ind[q2],:].reshape(1, -1))
        X_test[i,125] = distance.canberra(LSA_features_stem[ids2ind[q1],:].reshape(1, -1),LSA_features_stem[ids2ind[q2],:].reshape(1, -1))
        X_test[i,126] = distance.minkowski(LSA_features_stem[ids2ind[q1],:].reshape(1, -1),LSA_features_stem[ids2ind[q2],:].reshape(1, -1),3)
        X_test[i,127] = distance.braycurtis(LSA_features_stem[ids2ind[q1],:].reshape(1, -1),LSA_features_stem[ids2ind[q2],:].reshape(1, -1))

        X_test[i,128] = 1 - cosine_similarity(LSA_bis_features_stem[ids2ind[q1],:].reshape(1, -1), LSA_bis_features_stem[ids2ind[q2],:].reshape(1, -1))
        X_test[i,129] = np.linalg.norm(LSA_bis_features_stem[ids2ind[q1],:].reshape(1, -1)- LSA_bis_features_stem[ids2ind[q2],:].reshape(1, -1))
        X_test[i,130] = distance.cityblock(LSA_bis_features_stem[ids2ind[q1],:].reshape(1, -1),LSA_bis_features_stem[ids2ind[q2],:].reshape(1, -1))
        X_test[i,131] = distance.jaccard(LSA_bis_features_stem[ids2ind[q1],:].reshape(1, -1),LSA_bis_features_stem[ids2ind[q2],:].reshape(1, -1))
        X_test[i,132] = distance.canberra(LSA_bis_features_stem[ids2ind[q1],:].reshape(1, -1),LSA_bis_features_stem[ids2ind[q2],:].reshape(1, -1))
        X_test[i,133] = distance.minkowski(LSA_bis_features_stem[ids2ind[q1],:].reshape(1, -1),LSA_bis_features_stem[ids2ind[q2],:].reshape(1, -1),3)
        X_test[i,134] = distance.braycurtis(LSA_bis_features_stem[ids2ind[q1],:].reshape(1, -1),LSA_bis_features_stem[ids2ind[q2],:].reshape(1, -1))



    return X_train,X_test