'''
Main -- ALTEGRAD Challenge Fall 2017 -- RATNAMOGAN Pirashanth -- SAYEM Othmane

This file contains the main file that will call all the needed function in order
to create our submission file.

The challenge goal is to compare pairs of questions and say if they 
are duplicates or not.

In this file, first the data stored as a csv are read.
Then the features that has been created by our team are computed.
Finally the three classifiers that are used for prediction are called.

If one wants to run this code some paths must be changed in order to fit
its system.

One can have a look at our final report in order to understand what is done.
'''

#Basic libraries 
import numpy as np
import pickle #for serealization
import os
import collections #Needed to create dictionnaries with order
import random 

#Gensim
from gensim.models.word2vec import Word2Vec

###Features####
from preprocessing.preprocess import preprocess_raw_text
from Features.BasicFeatures import create_all_simple_features
from Features.GoQFeatures import create_GoQ_Features
from Features.StackingFeatures import create_stacking_features
from Features.StackingFeatures import create_nn_model_v2
from Features.GoWFeatures import create_features_GoW
from Features.Features4Nnandskmodels import create_features_w2v_for_nn,create_features_lsa_for_sk

#Deep Learning Tool
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#Efficient gradient boosting algorithms
from xgboost import XGBClassifier 
from lightgbm import LGBMClassifier


#Sklearn tools
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

#from Rescaling import create_graph_of_positive_pairs #not efficient
print('packages loaded')


##################PATH TO DEFINE####################

#path to the folder that contains Word2vec bin
path_to_pretrained_wv = 'F:\\ALTEGRAD\\' # fill me!


#path to the Glove bin that you want to use (must contain 200 dim vect) 
GLOVE_DIR = 'F:\\ALTEGRAD\\glove.twitter.27B'

#####################################################


##################SERIALIZATION OPTION####################
#The first time that you are running the code all the options 
#must be setted as false
#If True previously dumped features and input will be loaded
#You can choose to load only some features and compute the others...

#See https://docs.python.org/2/library/pickle.html

use_pickled_data = False #If this one is false all the others must be false
use_pickled_basic_features=False 
use_pickled_GoQ = False
use_pickled_stacking = False
use_pickled_GoW = False

#####################################################


#Params deep learning and word2vec#
use_pretrained = True # when using pre-trained embeddings, convergence is faster, and absolute accuracy is slightly greater (the margin is larger when max_size is small)
max_features = int(1e4)
max_size = 20 # maximum document size allowed
word_vector_dim = int(3e2)
drop_rate = 0.2
batch_size = 64
nb_epoch =  10# increasing the number of epochs may lead to overfitting when max_size is small (especially since dataset is small)
my_optimizer = 'rmsprop' #Define optimizer to use
my_patience = 0 # for early stopping strategy

###############Read All the data###############
if (not use_pickled_data):
    
    texts = {}
    y_train = []
    pairs_test = []
    pairs_train = []
    with open('train.csv','r', encoding='utf8') as f:
        for line in f:
            l = line.split(',')
            if l[1] not in texts:
                texts[l[1]] = preprocess_raw_text(l[3][:-1],stemming=False,check_spelling=True)
            if l[2] not in texts:
                texts[l[2]] = preprocess_raw_text(l[4][:-1],stemming=False,check_spelling=True)
    
            pairs_train.append([l[1],l[2]])
    
            y_train.append(int(l[5][:-1])) # [:-1] is just to remove formatting at the end
    
    
    with open('test.csv','r', encoding='utf8') as f:
        for line in f:
            l = line.split(',')
            if l[1] not in texts:
                texts[l[1]] = preprocess_raw_text(l[3][:-1],stemming=False,check_spelling=True) #retirer le point d'interrogation
            if l[2] not in texts:
                texts[l[2]] = preprocess_raw_text(l[4][:-1],stemming=False,check_spelling=True)
    
            pairs_test.append([l[1], l[2]])
    
    ids2ind = {} # will contain the row idx of each unique text in the TFIDF matrix 
    texts= collections.OrderedDict(texts)
    for qid in texts:
        ids2ind[qid] = len(ids2ind)
    #########################################################
        
    
    #################tokenization and data processing#######
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(texts.values())
    sequences = tokenizer.texts_to_sequences(texts.values())
    word_index = collections.OrderedDict(tokenizer.word_index)
    print('Found %s unique tokens.' % len(word_index))
    index_to_word = dict((v,k) for k, v in word_index.items())
    data = pad_sequences(sequences, maxlen=max_size)
    data_full_word = [[index_to_word[idx] for idx in rev if idx!=0] for rev in data.tolist()]
    
    
    ################# End tokenization and data processing#######
    
    
    ########### Word2Vec#############
    word_vectors = Word2Vec(size=word_vector_dim, min_count=1)
    word_vectors.build_vocab(data_full_word)
    word_vectors.intersect_word2vec_format(path_to_pretrained_wv + 'GoogleNews-vectors-negative300.bin.gz', binary=True)
    embedding_matrix = np.zeros((len(word_index) + 1, word_vector_dim))
    
    for word in list(word_vectors.wv.vocab):
        idx = word_index[word]
        embedding_matrix[idx,] = word_vectors[word]
    ######## End Word2Vec##############
    

    ######################Glove###########################
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.twitter.27B.200d.txt'), encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    print('Found %s word vectors.' % len(embeddings_index))
    
    
    glove_embed_dim = 200
    glove_embedding_matrix = np.zeros((len(word_index) + 1, glove_embed_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            glove_embedding_matrix[i] = embedding_vector
    ###################### EndGlove###########################

    #Dump all the stuff computed previously
    with open('Input_data.p',"wb") as pick_file:
        pickle.dump([pairs_train,pairs_test,y_train,\
                     ids2ind,word_vectors,embedding_matrix,sequences,\
                     texts,data,data_full_word,glove_embedding_matrix],pick_file)
else:
    with open('Input_data.p',"rb") as pick_file:
        pairs_train,pairs_test,y_train,\
        ids2ind,word_vectors,embedding_matrix,sequences,\
        texts,data,data_full_word,glove_embedding_matrix = pickle.load(pick_file)
    
print('Create Basic Features')

if (not use_pickled_basic_features):
    #Compute what has been called "basic features" that are mostly nlp
    #features that can be computed in few lines of code (without graph)
    X_train_basic,X_test_basic = create_all_simple_features(pairs_train,pairs_test,texts,\
                               ids2ind,word_vectors,embedding_matrix,glove_embedding_matrix,sequences,data_full_word,my_p=50,n_lsa=40)
    with open('Basic_feature.p',"wb") as pick_file:
        pickle.dump([X_train_basic,X_test_basic],pick_file)
else:
    with open('Basic_feature.p',"rb") as pick_file:
        X_train_basic,X_test_basic = pickle.load(pick_file)

print('Create GoQ Features')
if (not use_pickled_GoQ):
    #Compute what has been called "Graph of questions features"
    #Extract the features from the graph that can be created using
    #the comparisons in train and test set
    X_train_GoQ,X_test_GoQ= create_GoQ_Features(pairs_train,pairs_test,[],ids2ind)
    with open('GoQ_feature.p',"wb") as pick_file:
        pickle.dump([X_train_GoQ,X_test_GoQ],pick_file)
else:
    with open('GoQ_feature.p',"rb") as pick_file:
        X_train_GoQ,X_test_GoQ = pickle.load(pick_file)
        
print('Creating GoW Features')
if (not use_pickled_GoW):
    #Compute what has been called "Graph of words features"
    #Compute essentially the kernel described here
    #http://aclweb.org/anthology/D17-1202 (Nikolentzos et al.)
    X_train_GoW,X_test_GoW = create_features_GoW(pairs_train,pairs_test,data_full_word,ids2ind,w=2,d=3)  
    with open('GoW_feature.p',"wb") as pick_file:
        pickle.dump([X_train_GoW,X_test_GoW],pick_file)
else:
    with open('GoW_feature.p',"rb") as pick_file:
        X_train_GoW,X_test_GoW = pickle.load(pick_file)

X_train_GoQ = np.nan_to_num(X_train_GoQ) #deal with nan et inf
X_test_GoQ = np.nan_to_num(X_test_GoQ)

print('Create Stacking Features')
if (not use_pickled_stacking):
    #Compute various features using stacking, sklearn model xgb and deep
    #learning approaches
    X_train_stack,X_test_stack = create_stacking_features(pairs_train,y_train,pairs_test,data,ids2ind,texts,\
                             max_size,embedding_matrix,glove_embedding_matrix,word_vector_dim,drop_rate,my_optimizer,sequences,word_vectors,data_full_word,\
                             X_train_basic,X_test_basic,X_train_GoQ,X_test_GoQ,X_train_GoW,X_test_GoW,my_p=50,n_lsa=40,\
                             my_patience=0,batch_size=batch_size,nb_epoch=nb_epoch)
    with open('Stack_feature.p',"wb") as pick_file:
        pickle.dump([X_train_stack,X_test_stack],pick_file)
else:
    with open('Stack_feature.p',"rb") as pick_file:
        X_train_stack,X_test_stack = pickle.load(pick_file)
 
    
#Create the final input matrices with features
X_train = np.concatenate((X_train_basic,X_train_GoQ,X_train_stack,X_train_GoW),axis=1)
X_test = np.concatenate((X_test_basic,X_test_GoQ,X_test_stack,X_test_GoW),axis=1)
y_train = np.array(y_train)

#We will use the training data in order to create the final model
nb_trainin_data= X_train.shape[0] #nb of training data
list_training_data = list(range(nb_trainin_data)) #in order to do bagging

models= []

#In the following lines 50 models with XGBoost and 50 models 
#with LGBM are computed. Each model vary by the fact that they 
#are seeing only 90% of the training data, the other randomly chose
#10% are used for early stopping strategies.

for i in range(50):
    models.append(XGBClassifier(max_depth=10,n_estimators=300, reg_lambda=1,seed=random.randint(0,10)))
    index_to_consider,list_elements_to_remove = train_test_split(list(range(X_train.shape[0])),test_size=0.1)
    X_to_train_for_this_bagg = X_train[index_to_consider,:]
    y_train_for_this_bagg = y_train[index_to_consider]
    X_val_xgb= X_train[list_elements_to_remove,:]
    y_val_xgb=y_train[list_elements_to_remove]
    models[-1].fit(X_to_train_for_this_bagg,y_train_for_this_bagg, early_stopping_rounds=30,eval_metric='logloss', verbose=True,eval_set=[(X_to_train_for_this_bagg,y_train_for_this_bagg),(X_val_xgb,y_val_xgb)])

for i in range(50):
    models.append(LGBMClassifier(max_depth=10,n_estimators=300))
    index_to_consider,list_elements_to_remove = train_test_split(list(range(X_train.shape[0])),test_size=0.1)
    X_to_train_for_this_bagg = X_train[index_to_consider,:]
    y_train_for_this_bagg = y_train[index_to_consider]
    X_val_xgb= X_train[list_elements_to_remove,:]
    y_val_xgb=y_train[list_elements_to_remove]
    models[-1].fit(X_to_train_for_this_bagg,y_train_for_this_bagg, early_stopping_rounds=30,eval_metric='logloss', verbose=True,eval_set=[(X_to_train_for_this_bagg,y_train_for_this_bagg),(X_val_xgb,y_val_xgb)])

#early stopping for deep learning
early_stopping = EarlyStopping(monitor='val_loss', # go through epochs as long as accuracy on validation set increases
                           patience=my_patience,
                           mode='min')

#conserve the model that has provided the best performances
#on the validation set (avoid overfitting)
mcp = ModelCheckpoint('weights.best.hdf5', monitor="val_acc",
                      save_best_only=True, save_weights_only=False)

#Create the input for the deep learning models
X_train_nn,X_test_nn = create_features_w2v_for_nn(pairs_train,pairs_test,data,ids2ind)


X_train_nn_GoQ = X_train_nn + [X_train]
X_test_nn_GoQ = X_test_nn + [X_test]
    
#Split the training data in order to keep some data for early stopping
index_to_consider,list_elements_to_remove = train_test_split(list(range(X_train.shape[0])),test_size=0.1)
X_to_train_for_this_bagg = [X_train_nn_GoQ[0][index_to_consider,:],X_train_nn_GoQ[1][index_to_consider,:],X_train_nn_GoQ[2][index_to_consider,:]]
y_train_for_this_bagg = y_train[index_to_consider]
X_val = [X_train_nn_GoQ[0][list_elements_to_remove,:],X_train_nn_GoQ[1][list_elements_to_remove,:],X_train_nn_GoQ[2][list_elements_to_remove,:]]
y_val=y_train[list_elements_to_remove]
    
model_v2 = create_nn_model_v2(max_size,embedding_matrix,word_vector_dim,drop_rate,my_optimizer,X_train.shape[1])

model_v2.fit(X_to_train_for_this_bagg, 
      y_train_for_this_bagg,
      batch_size = batch_size,
      epochs = 12,validation_data=(X_val,y_val),
      callbacks = [mcp]
      )

#Load the model at the best epoch
model_v2 = keras.models.load_model('weights.best.hdf5')

#predict for submission
pred_nn = model_v2.predict(X_test_nn_GoQ)[:,0]
keras.backend.clear_session()


predictions = [model_xgb1.predict_proba(X_test)[:,1] for model_xgb1 in models]
predictions = np.array(predictions)
#zWe will take as solution the less certain 
y_pred =  predictions.mean(axis=0).reshape((-1,)) #average output of LGBM and XGB

y_pred =list((y_pred + pred_nn)/2) #average previous output with the output of the neural network

with open("submission_all.csv", 'w') as f:
    f.write("Id,Score\n")
    for i in range(len(y_pred)):
        f.write(str(i)+','+str(y_pred[i])+'\n')
