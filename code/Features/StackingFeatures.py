'''
StackingFeatures -- ALTEGRAD Challenge Fall 2017 -- RATNAMOGAN Pirashanth -- SAYEM Othmane

This file contains the function that allows to compute what we have called
'StackingFeatures'. Stackings is a machine learning method that essentialy
create features from various machine learning models.
'''

from Features.Features4Nnandskmodels import create_features_w2v_for_nn,create_features_lsa_for_sk
from Features.BasicFeatures import create_all_simple_features
from Features.GoQFeatures import create_GoQ_Features
from Features.GoWFeatures import create_features_GoW
from sklearn.model_selection import train_test_split

import keras
from keras.models import Model, load_model
from keras import backend as K
from keras.layers import Input, Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense,LSTM,BatchNormalization,Subtract, Multiply, Maximum
from keras.layers.wrappers import Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.linear_model import ElasticNet



from xgboost import XGBClassifier 
from lightgbm import LGBMClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np 
import gc


def create_nn_model(max_size,embedding_matrix,word_vector_dim,drop_rate,my_optimizer):
    '''
    Create a Siamese neural network based only on pretrained word2vec embeddings and tokens
    '''
    question_1 = Input(shape=(max_size,)) # we leave the 2nd argument of shape blank because the Embedding layer cannot accept an input_shape argument
    question_2 = Input(shape=(max_size,)) # we leave the 2nd argument of shape blank because the Embedding layer cannot accept an input_shape argument
    
    embedding_1 = Embedding(input_dim=embedding_matrix.shape[0], # vocab size, including the 0-th word used for padding
                          output_dim=word_vector_dim,
                          weights=[embedding_matrix], # we pass our pre-trained embeddings
                          input_length=max_size,
                          trainable=True,
                          ) (question_1)
    
    embedding_2 = Embedding(input_dim=embedding_matrix.shape[0], # vocab size, including the 0-th word used for padding
                          output_dim=word_vector_dim,
                          weights=[embedding_matrix], # we pass our pre-trained embeddings
                          input_length=max_size,
                          trainable=True,
                          ) (question_2)
    
    
    conv_1 =Conv1D(30,4,activation = 'relu')
    
    conv_1_1 = GlobalMaxPooling1D()((Dropout(drop_rate)(BatchNormalization()(conv_1(embedding_1)))))
    conv_1_2 = GlobalMaxPooling1D()((Dropout(drop_rate)(BatchNormalization()(conv_1(embedding_2)))))
    
    lstm_1 = LSTM(30,return_sequences=False)
    
    lstm_1_1 = Dropout(drop_rate)(BatchNormalization()(lstm_1(embedding_1)))
    lstm_1_2 = Dropout(drop_rate)(BatchNormalization()(lstm_1(embedding_2)))
    
    
    
    merged_1 = keras.layers.concatenate([conv_1_1,conv_1_2,lstm_1_1,lstm_1_2],axis=-1)
    
    merged_2 = (Dense(20,activation='relu')(Dropout(drop_rate)(BatchNormalization()(merged_1))))
    prob = Dense(1,activation='sigmoid')(BatchNormalization()(merged_2))
    
    model = Model([question_1,question_2], prob)
    
    
    model.compile(loss=  'binary_crossentropy',optimizer = my_optimizer,
                  metrics = ['accuracy']) 
    
    return model


def create_nn_model_v2(max_size,embedding_matrix,word_vector_dim,drop_rate,my_optimizer,shape_dist):
    '''
    Create a Siamese neural network based on pretrained word2vec embeddings and tokens and all the other features
    '''
    # we leave the 2nd argument of shape blank because the Embedding layer cannot accept an input_shape argument
    input_1 = Input(shape=(max_size,))
    input_2 = Input(shape=(max_size,))
    
    embedding_1 = Embedding(input_dim=embedding_matrix.shape[0], # vocab size, including the 0-th word used for padding
                              output_dim=word_vector_dim,
                              weights=[embedding_matrix], # we pass our pre-trained embeddings
                              input_length=max_size,
                              trainable=True,
                              ) (input_1)
    
    embedding_2 = Embedding(input_dim=embedding_matrix.shape[0], # vocab size, including the 0-th word used for padding
                              output_dim=word_vector_dim,
                              weights=[embedding_matrix], # we pass our pre-trained embeddings
                              input_length=max_size,
                              trainable=True,
                              ) (input_2)
    
    
    embedding_dropped_1= Dropout(drop_rate)(embedding_1)
    embedding_dropped_2= Dropout(drop_rate)(embedding_2)
    
    
    ### Defining CNN
    conv1 = Conv1D(filters = 100,
                  kernel_size = 3,
                  activation = 'relu',
                  )
    
    conv2 = Conv1D(filters = 128,
                  kernel_size = 4,
                  activation = 'relu',
                  )
    
    #conv3 = Conv1D(filters = 128,
    #              kernel_size = 5,
    #              activation = 'relu',
    #              )
    
    ## First conv layer
    conv1_1 = conv1(embedding_dropped_1)
    pooled_conv1_1 = GlobalMaxPooling1D()(conv1_1)
    pooled_conv1_dropped_1 = Dropout(drop_rate)(pooled_conv1_1)
    
    conv1_2 = conv1(embedding_dropped_2)
    pooled_conv1_2 = GlobalMaxPooling1D()(conv1_2)
    pooled_conv1_dropped_2 = Dropout(drop_rate)(pooled_conv1_2)
    
    ## Second conv layer
    conv2_1 = conv2(embedding_dropped_1)
    pooled_conv2_1 = GlobalMaxPooling1D()(conv2_1)
    pooled_conv2_dropped_1 = Dropout(drop_rate)(pooled_conv2_1)
    
    conv2_2 = conv2(embedding_dropped_2)
    pooled_conv2_2 = GlobalMaxPooling1D()(conv2_2)
    pooled_conv2_dropped_2 = Dropout(drop_rate)(pooled_conv2_2)
    
    merged_1 = keras.layers.concatenate([pooled_conv1_dropped_1,pooled_conv2_dropped_1])
    merged_2 = keras.layers.concatenate([pooled_conv1_dropped_2,pooled_conv2_dropped_2])
    
    diff = Subtract()([merged_1,merged_2])
    mul = Multiply()([merged_1,merged_2])
    maxi = Maximum()([merged_1,merged_2])
    #dense_1 = Dense(20,activation='relu')(merged_1)
    #dense_2 = Dense(20,activation='relu')(merged_2)
    
    distance_input = Input(shape=(shape_dist,))
    distance_dense = BatchNormalization()(distance_input)
    distance_dense = Dense(128, activation='relu')(distance_dense)
    
    
    merge = keras.layers.concatenate([diff, mul,maxi,distance_dense], axis = -1)
    
    #prob = Dense(units = 1, # dimensionality of the output space
    #             activation = 'sigmoid'#,
    #             ) (merge)
    
    prob = Dropout(0.2)(merge)
    prob = BatchNormalization()(prob)
    prob = Dense(300, activation='relu')(prob)
        
    prob = Dropout(0.2)(prob)
    prob = BatchNormalization()(prob)
    prob = Dense(1, activation='sigmoid')(prob)
    
    model = Model([input_1,input_2,distance_input], prob)
    
    model.compile(loss=  'binary_crossentropy',optimizer = my_optimizer,
                  metrics = ['accuracy'])
    
    return model

def compute_part_of_the_stacking(X_train_nn,X_train_sk,X_train_basic,\
                                 X_train_GoQ,y_train,X_test_nn,X_test_sk,\
                                 X_test_basic,X_test_GoQ,max_size,embedding_matrix,glove_embedding\
                                 ,word_vector_dim,drop_rate,my_optimizer,batch_size,nb_epoch,my_patience=0):
    '''
    Compute the stackings features from the given input matrices that are defined
    Various easy parameters are used ...
    '''
    
    X_train_basic_GoQ = np.concatenate((X_train_basic,X_train_GoQ),axis=1)
    X_test_basic_GoQ = np.concatenate((X_test_basic,X_test_GoQ),axis=1)
    
    mcp1 = ModelCheckpoint('weights.stack.hdf5', monitor="val_acc",save_best_only=True, save_weights_only=False)
    
    model = create_nn_model(max_size,embedding_matrix,word_vector_dim,drop_rate,my_optimizer)
    model.fit(X_train_nn, 
          y_train,
          batch_size = batch_size,
          epochs = nb_epoch,
          validation_split=0.1,
          callbacks = [mcp1],
          )

    
    keras.backend.clear_session()

    
    mcp2 = ModelCheckpoint('weights.stack_gv.hdf5', monitor="val_acc",save_best_only=True, save_weights_only=False)

    print('Computing Neural Netword for Glove features')
    model_gv = create_nn_model(max_size,glove_embedding,200,drop_rate,my_optimizer)
    model_gv.fit(X_train_nn, 
          y_train,
          batch_size = batch_size,
          epochs = nb_epoch,
          validation_split=0.1,
          callbacks = [mcp2],
          )
    keras.backend.clear_session()




    
    X_train_nn_GoQ = X_train_nn + [X_train_GoQ]
    X_test_nn_GoQ = X_test_nn + [X_test_GoQ]
    
    
    mcp3 = ModelCheckpoint('weights.stack_1.hdf5', monitor="val_acc",save_best_only=True, save_weights_only=False)

    model_v2 = create_nn_model_v2(max_size,embedding_matrix,word_vector_dim,drop_rate,my_optimizer,X_train_GoQ.shape[1])
    model_v2.fit(X_train_nn_GoQ, 
          y_train,
          batch_size = batch_size,
          epochs = nb_epoch,
          validation_split=0.1,
          callbacks = [mcp3],
          )
    

    keras.backend.clear_session()



    
    mcp4 = ModelCheckpoint('weights.stack_gv_1.hdf5', monitor="val_acc",save_best_only=True, save_weights_only=False)

    model_gv_v2 = create_nn_model_v2(max_size,glove_embedding,200,drop_rate,my_optimizer,X_train_GoQ.shape[1])
    model_gv_v2.fit(X_train_nn_GoQ, 
          y_train,
          batch_size = batch_size,
          epochs = nb_epoch,
          validation_split=0.1,
          callbacks = [mcp4],
          )
    
    keras.backend.clear_session()


    
    X_train_nn_basic  = X_train_nn + [X_train_basic]
    X_test_nn_basic = X_test_nn + [X_test_basic]
    
    mcp5 = ModelCheckpoint('weights.stack_2.hdf5', monitor="val_acc",save_best_only=True, save_weights_only=False)

    
    model_v3 = create_nn_model_v2(max_size,embedding_matrix,word_vector_dim,drop_rate,my_optimizer,X_train_basic.shape[1])
    model_v3.fit(X_train_nn_basic, 
          y_train,
          batch_size = batch_size,
          epochs = nb_epoch,
          validation_split=0.1,
          callbacks = [mcp5],
          )
    
    
    keras.backend.clear_session()

    
    mcp6 = ModelCheckpoint('weights.stack_gv_2.hdf5', monitor="val_acc",save_best_only=True, save_weights_only=False)

    
    model_gv_v3 = create_nn_model_v2(max_size,glove_embedding,200,drop_rate,my_optimizer,X_train_basic.shape[1])
    model_gv_v3.fit(X_train_nn_basic, 
          y_train,
          batch_size = batch_size,
          epochs = nb_epoch,
          validation_split=0.1,
          callbacks = [mcp6],
          )
    
    keras.backend.clear_session()
    
    


    
    X_train_nn_basic_GoQ  = X_train_nn + [X_train_basic_GoQ]
    X_test_nn_basic_GoQ = X_test_nn + [X_test_basic_GoQ]
    
    mcp7 = ModelCheckpoint('weights.stack_3.hdf5', monitor="val_acc",save_best_only=True, save_weights_only=False)

    
    model_v4 = create_nn_model_v2(max_size,embedding_matrix,word_vector_dim,drop_rate,my_optimizer,X_train_basic_GoQ.shape[1])
    model_v4.fit(X_train_nn_basic_GoQ, 
          y_train,
          batch_size = batch_size,
          epochs = nb_epoch,
          validation_split=0.1,
          callbacks = [mcp7],
          )
    
    keras.backend.clear_session()
    

    mcp8 = ModelCheckpoint('weights.stack_gv_3.hdf5', monitor="val_acc",save_best_only=True, save_weights_only=False)

    
    model_gv_v4 = create_nn_model_v2(max_size,glove_embedding,200,drop_rate,my_optimizer,X_train_basic_GoQ.shape[1])
    model_gv_v4.fit(X_train_nn_basic_GoQ, 
          y_train,
          batch_size = batch_size,
          epochs = nb_epoch,
          validation_split=0.1,
          callbacks = [mcp8],
          )

    keras.backend.clear_session()
    
    
    print('Computing skmodels')
    
    rf_model = RandomForestClassifier(n_estimators=50)
    elastic_net_model = ElasticNet()
    log_reg_model = LogisticRegression()
    lin_reg_model = LinearRegression()
    km_model = KNeighborsClassifier(n_neighbors=50)
    svm_model = SVC(probability=True)
    xgb_model = XGBClassifier(max_depth=6,n_estimators=100, reg_lambda=1,seed=5)
    lightgbm_model = LGBMClassifier(max_depth=6,n_estimators=100, reg_lambda=1,seed=5)


    rf_model_GoQ = RandomForestClassifier(n_estimators=50)
    elastic_net_model_GoQ = ElasticNet()
    log_reg_model_GoQ = LogisticRegression()
    lin_reg_model_GoQ = LinearRegression()
    km_model_GoQ = KNeighborsClassifier(n_neighbors=50)
    svm_model_GoQ = SVC(probability=True)
    xgb_model_GoQ = XGBClassifier(max_depth=6,n_estimators=100, reg_lambda=1,seed=5)
    lightgbm_model_GOQ = LGBMClassifier(max_depth=6,n_estimators=100, reg_lambda=1,seed=5)

    
    rf_model_basic = RandomForestClassifier(n_estimators=50)
    elastic_net_model_basic = ElasticNet()
    log_reg_model_basic= LogisticRegression()
    lin_reg_model_basic = LinearRegression()
    km_model_basic =KNeighborsClassifier(n_neighbors=50)
    svm_model_basic = SVC(probability=True)
    xgb_model_basic = XGBClassifier(max_depth=6,n_estimators=100, reg_lambda=1,seed=5)
    lightgbm_model_basic = LGBMClassifier(max_depth=6,n_estimators=100, reg_lambda=1,seed=5)


    rf_model_basic_GoQ = RandomForestClassifier(n_estimators=200)
    elastic_net_model_basic_GoQ = ElasticNet()
    log_reg_model_basic_GoQ= LogisticRegression()
    lin_reg_model_basic_GoQ = LinearRegression()
    km_model_basic_GoQ = KNeighborsClassifier(n_neighbors=50)
    svm_model_basic_GoQ = SVC(probability=True)
    xgb_model_basic_GoQ = XGBClassifier(max_depth=6,n_estimators=150, reg_lambda=1,seed=5)
    lightgbm_model_basic_GoQ = LGBMClassifier(max_depth=6,n_estimators=150, reg_lambda=1,seed=5)


    
    print('XGB Model')
    xgb_model.fit(X_train_sk,y_train)
    xgb_model_basic.fit(X_train_basic,y_train)
    xgb_model_GoQ.fit(X_train_GoQ,y_train)
    xgb_model_basic_GoQ.fit(X_train_basic_GoQ,y_train)
    
    print('Light GBM')
    lightgbm_model.fit(X_train_sk,y_train)
    lightgbm_model_basic.fit(X_train_basic,y_train)
    lightgbm_model_GOQ.fit(X_train_GoQ,y_train)
    lightgbm_model_basic_GoQ.fit(X_train_basic_GoQ,y_train)
    
    print('Train RF Model')
    rf_model.fit(X_train_sk,y_train)
    rf_model_basic.fit(X_train_basic,y_train)
    rf_model_GoQ.fit(X_train_GoQ,y_train)
    rf_model_basic_GoQ.fit(X_train_basic_GoQ,y_train)


    print('ElasticNet Model')
    elastic_net_model.fit(X_train_sk,y_train)
    elastic_net_model_basic.fit(X_train_basic,y_train)
    elastic_net_model_GoQ.fit(X_train_GoQ,y_train)
    elastic_net_model_basic_GoQ.fit(X_train_basic_GoQ,y_train)

    
    print('Logistic Reg Model')
    log_reg_model.fit(X_train_sk,y_train)
    log_reg_model_basic.fit(X_train_basic,y_train)
    log_reg_model_GoQ.fit(X_train_GoQ,y_train)
    log_reg_model_basic_GoQ.fit(X_train_basic_GoQ,y_train)

    
    print('Linear Reg Model')
    lin_reg_model.fit(X_train_sk,y_train)
    lin_reg_model_basic.fit(X_train_basic,y_train)
    lin_reg_model_GoQ.fit(X_train_GoQ,y_train)
    lin_reg_model_basic_GoQ.fit(X_train_basic_GoQ,y_train)

    
    print('KMeans Model')
    km_model.fit(X_train_sk,y_train)
    km_model_basic.fit(X_train_basic,y_train)
    km_model_GoQ.fit(X_train_GoQ,y_train)
    km_model_basic_GoQ.fit(X_train_basic_GoQ,y_train)

    '''#Too long to compute
    print('SVM Model')
    svm_model.fit(X_train_sk,y_train)
    svm_model_basic.fit(X_train_basic,y_train)
    svm_model_GoQ.fit(X_train_GoQ,y_train)
    svm_model_basic_GoQ.fit(X_train_basic_GoQ,y_train)
    '''
    print('Predict Output Test')

    model_gv_v4 = keras.models.load_model('weights.stack_gv_3.hdf5')

    model_v4 = keras.models.load_model('weights.stack_3.hdf5')
    
    model_gv_v3 = keras.models.load_model('weights.stack_gv_2.hdf5')

    model_v3 = keras.models.load_model('weights.stack_2.hdf5')

    model_gv_v2 = keras.models.load_model('weights.stack_gv_1.hdf5')

    model_v2 = keras.models.load_model('weights.stack_1.hdf5')

    model_gv = keras.models.load_model('weights.stack_gv.hdf5')
    
    model = keras.models.load_model('weights.stack.hdf5')


    outcome_nn_test = model.predict(X_test_nn)
    outcome_nn_test_gv = model_gv.predict(X_test_nn)
    
    outcome_nn_test_v2 = model_v2.predict(X_test_nn_GoQ)
    outcome_nn_test_gv_v2 = model_gv_v2.predict(X_test_nn_GoQ)
    
    outcome_nn_test_v3  = model_v3.predict(X_test_nn_basic)
    outcome_nn_test_gv_v3  = model_gv_v3.predict(X_test_nn_basic)
    
    outcome_nn_test_v4  = model_v4.predict(X_test_nn_basic_GoQ)
    outcome_nn_test_gv_v4  = model_gv_v4.predict(X_test_nn_basic_GoQ)

    keras.backend.clear_session()
    
    outcome_rf_test = rf_model.predict_proba(X_test_sk)[:,1].reshape((-1,1))
    outcome_ada_test = elastic_net_model.predict(X_test_sk).reshape((-1,1))
    outcome_log_reg_model_test = log_reg_model.predict_proba(X_test_sk)[:,1].reshape((-1,1))
    outcme_lin_model_test = lin_reg_model.predict(X_test_sk).reshape((-1,1))
    outcome_kmeans_test = km_model.predict_proba(X_test_sk)[:,1].reshape((-1,1))
    #outcome_svm_test = svm_model.predict(X_test_sk).reshape((-1,1))
    outcome_xgb_test = xgb_model.predict_proba(X_test_sk)[:,1].reshape((-1,1))
    outcome_lgb_test = lightgbm_model.predict_proba(X_test_sk)[:,1].reshape((-1,1))

    outcome_rf_test_basic = rf_model_basic.predict_proba(X_test_basic)[:,1].reshape((-1,1))
    outcome_ada_test_basic = elastic_net_model_basic.predict(X_test_basic).reshape((-1,1))
    outcome_log_reg_model_test_basic = log_reg_model_basic.predict_proba(X_test_basic)[:,1].reshape((-1,1))
    outcme_lin_mode_testl_basic = lin_reg_model_basic.predict(X_test_basic).reshape((-1,1))
    outcome_kmeans_test_basic = km_model_basic.predict_proba(X_test_basic)[:,1].reshape((-1,1))
    #outcome_svm_test_basic = svm_model_basic.predict(X_test_basic).reshape((-1,1))
    outcome_xgb_test_basic = xgb_model_basic.predict_proba(X_test_basic)[:,1].reshape((-1,1))
    outcome_lgb_test_basic = lightgbm_model_basic.predict_proba(X_test_basic)[:,1].reshape((-1,1))


    outcome_rf_test_GoQ = rf_model_GoQ.predict_proba(X_test_GoQ)[:,1].reshape((-1,1))
    outcome_ada_test_GoQ  = elastic_net_model_GoQ.predict(X_test_GoQ).reshape((-1,1))
    outcome_log_reg_model_test_GoQ  = log_reg_model_GoQ.predict_proba(X_test_GoQ)[:,1].reshape((-1,1))
    outcme_lin_mode_testl_GoQ  = lin_reg_model_GoQ.predict(X_test_GoQ).reshape((-1,1))
    outcome_kmeans_test_GoQ  = km_model_GoQ.predict_proba(X_test_GoQ)[:,1].reshape((-1,1))
    #outcome_svm_test_GoQ  = svm_model_GoQ.predict(X_test_GoQ).reshape((-1,1))
    outcome_xgb_test_GoQ = xgb_model_GoQ.predict_proba(X_test_GoQ)[:,1].reshape((-1,1))
    outcome_lgb_test_GoQ = lightgbm_model_GOQ.predict_proba(X_test_GoQ)[:,1].reshape((-1,1))


    outcome_rf_test_basic_GoQ = rf_model_basic_GoQ.predict_proba(X_test_basic_GoQ)[:,1].reshape((-1,1))
    outcome_ada_test_basic_GoQ = elastic_net_model_basic_GoQ.predict(X_test_basic_GoQ).reshape((-1,1))
    outcome_log_reg_model_test_basic_GoQ = log_reg_model_basic_GoQ.predict_proba(X_test_basic_GoQ)[:,1].reshape((-1,1))
    outcme_lin_mode_testl_basic_GoQ = lin_reg_model_basic_GoQ.predict(X_test_basic_GoQ).reshape((-1,1))
    outcome_kmeans_test_basic_GoQ = km_model_basic_GoQ.predict_proba(X_test_basic_GoQ)[:,1].reshape((-1,1))
    #outcome_svm_test_basic_GoQ = svm_model_basic_GoQ.predict(X_test_basic_GoQ).reshape((-1,1))
    outcome_xgb_test_basic_GoQ = xgb_model_basic_GoQ.predict_proba(X_test_basic_GoQ)[:,1].reshape((-1,1))
    outcome_lgb_test_basic_GoQ = lightgbm_model_basic_GoQ.predict_proba(X_test_basic_GoQ)[:,1].reshape((-1,1))



    '''X_test = np.concatenate([outcome_nn_test_gv,outcome_nn_test\
                             ,outcome_svm_test,outcome_rf_test,outcome_ada_test,outcome_log_reg_model_test,outcme_lin_model_test,outcome_kmeans_test,\
                             outcome_rf_test_basic,outcome_ada_test_basic,outcome_log_reg_model_test_basic,outcme_lin_mode_testl_basic,outcome_kmeans_test_basic,\
                             outcome_svm_test_basic\
                             ,outcome_rf_test_GoQ,outcome_ada_test_GoQ,outcome_log_reg_model_test_GoQ,outcme_lin_mode_testl_GoQ,outcome_kmeans_test_GoQ,outcome_svm_test_GoQ,\
                             outcome_rf_test_basic_GoQ,outcome_ada_test_basic_GoQ,outcome_log_reg_model_test_basic_GoQ,outcme_lin_mode_testl_basic_GoQ,\
                             outcome_kmeans_test_basic_GoQ,outcome_svm_test_basic_GoQ],axis=1)'''

    X_test = np.concatenate([outcome_nn_test_gv,outcome_nn_test,outcome_nn_test_v2,outcome_nn_test_gv_v2,outcome_nn_test_v3,outcome_nn_test_gv_v3,outcome_nn_test_v4,outcome_nn_test_gv_v4\
                             ,outcome_rf_test,outcome_ada_test,outcome_log_reg_model_test,outcme_lin_model_test,outcome_kmeans_test,\
                             outcome_rf_test_basic,outcome_ada_test_basic,outcome_log_reg_model_test_basic,outcme_lin_mode_testl_basic,outcome_kmeans_test_basic,\
                             outcome_rf_test_GoQ,outcome_ada_test_GoQ,outcome_log_reg_model_test_GoQ,outcme_lin_mode_testl_GoQ,outcome_kmeans_test_GoQ,\
                             outcome_rf_test_basic_GoQ,outcome_ada_test_basic_GoQ,outcome_log_reg_model_test_basic_GoQ,outcme_lin_mode_testl_basic_GoQ,\
                             outcome_kmeans_test_basic_GoQ,outcome_xgb_test,outcome_lgb_test,outcome_xgb_test_basic,outcome_lgb_test_basic,outcome_xgb_test_GoQ,\
                             outcome_lgb_test_GoQ,outcome_xgb_test_basic_GoQ,outcome_lgb_test_basic_GoQ],axis=1)
    
    return X_test

def concatenate_all_train_test(p_X_train_nn,p_X_test_nn,p_X_train_sk,p_X_test_sk\
                               ,p_X_train_basic,p_X_test_basic,p_X_train_GoQ,\
                               p_X_test_GoQ):
    '''
    Concatenate matrices in order to compute easily the stacking features
    '''
    X_train_et_test_nn = [np.concatenate((p_X_train_nn[0],p_X_test_nn[0]),axis=0),\
                                         np.concatenate((p_X_train_nn[1],p_X_test_nn[1]),axis=0)]
    X_train_et_test_sk = np.concatenate((p_X_train_sk,p_X_test_sk),axis=0)
    X_train_et_test_basic = np.concatenate((p_X_train_basic,p_X_test_basic),axis=0)
    X_train_et_test_GoQ = np.concatenate((p_X_train_GoQ,p_X_test_GoQ),axis=0)
    
    return X_train_et_test_nn,X_train_et_test_sk,X_train_et_test_basic,X_train_et_test_GoQ


def create_second_level_features(list_pairs_train,y_train,list_pairs_test,data_matrix,ids2ind,texts,\
                             max_size,embedding_matrix,glove_embedding,word_vector_dim,drop_rate,my_optimizer,sequences,word_vectors,data_full_word,p_X_train_basic,p_X_test_basic,p_X_train_GoQ,p_X_test_GoQ,p_X_train_GoW,p_X_test_GoW,\
                             my_p=50,n_lsa=40,my_patience=0,batch_size=64,nb_epoch=20):    
    '''
    Create second level features -- unused
    '''
    X_train_sk,X_test_sk=create_features_lsa_for_sk(list_pairs_train,list_pairs_test\
                               ,ids2ind,texts,n_lsa)
    index_to_consider_A,index_to_consider_B, X_train_basic_A, X_train_basic_B,\
    X_train_GoQ_A,X_train_GoQ_B,X_train_GoW_A,X_train_GoW_B,X_train_sk_A,X_train_sk_B= train_test_split(list(range(len(y_train))),p_X_train_basic,\
                                                                                                        p_X_train_GoQ,p_X_train_GoW,X_train_sk,test_size=0.5)
                   
    y_train = np.array(y_train)
    y_train_A = y_train[index_to_consider_A]
    y_train_B = y_train[index_to_consider_B]

                                                                                                    
    print('Computing Neural Netword for w2v features')
    X_train_nn,X_test_nn = create_features_w2v_for_nn(list_pairs_train,list_pairs_test,data_matrix,ids2ind)

    X_train_nn_A = [X_train_nn[0][index_to_consider_A,:],X_train_nn[1][index_to_consider_A,:]]
    X_train_nn_B = [X_train_nn[0][index_to_consider_B,:],X_train_nn[1][index_to_consider_B,:]]

    nb_element_in_A = len(index_to_consider_A) 
    nb_element_in_B = len(index_to_consider_B)
    
    print('Computing Neural Netword for w2v features')
    X_train_nn,X_test_nn = create_features_w2v_for_nn(list_pairs_train,list_pairs_test,data_matrix,ids2ind)

    X_train_nn_A = [X_train_nn[0][index_to_consider_A,:],X_train_nn[1][index_to_consider_A,:]]
    X_train_nn_B = [X_train_nn[0][index_to_consider_B,:],X_train_nn[1][index_to_consider_B,:]]

    nb_element_in_A = len(index_to_consider_A) 
    nb_element_in_B = len(index_to_consider_B)
    
    X_train_et_test_nn_A,X_train_et_test_sk_A,X_train_et_test_basic_A,\
    X_train_et_test_GoQ_A= concatenate_all_train_test(X_train_nn_A,X_test_nn,X_train_sk_A,X_test_sk\
                                                                                  ,X_train_basic_A,p_X_test_basic,X_train_GoQ_A,\
                                                                                  p_X_test_GoQ)
    X_train_et_test_nn_B,X_train_et_test_sk_B,X_train_et_test_basic_B,\
    X_train_et_test_GoQ_B= concatenate_all_train_test(X_train_nn_B,X_test_nn,X_train_sk_B,X_test_sk\
                                                                                  ,X_train_basic_B,p_X_test_basic,X_train_GoQ_B,\
                                                                                  p_X_test_GoQ)

    Output_B = compute_part_of_the_stacking(X_train_nn_A,X_train_sk_A,X_train_basic_A,\
                                            X_train_GoQ_A,y_train_A,X_train_et_test_nn_B,X_train_et_test_sk_B,\
                                            X_train_et_test_basic_B,X_train_et_test_GoQ_B,max_size,embedding_matrix,glove_embedding\
                                            ,word_vector_dim,drop_rate,my_optimizer,batch_size,nb_epoch)                                                                              
    
    Output_A = compute_part_of_the_stacking(X_train_nn_B,X_train_sk_B,X_train_basic_B,\
                                            X_train_GoQ_B,y_train_B,X_train_et_test_nn_A,X_train_et_test_sk_A,\
                                            X_train_et_test_basic_A,X_train_et_test_GoQ_A,max_size,embedding_matrix,glove_embedding\
                                            ,word_vector_dim,drop_rate,my_optimizer,batch_size,nb_epoch) 
        
        
    X_test_A = Output_A[nb_element_in_A:,:]
    X_test_B = Output_B[nb_element_in_B:,:]
    X_test_stack = (X_test_A+X_test_B)/2
    
    X_train_stack_A= Output_A[:nb_element_in_A,:]
    X_train_stack_B = Output_B[:nb_element_in_B,:]
    
    X_train_stack= np.zeros((len(y_train),X_train_stack_A.shape[1]))
    X_train_stack[index_to_consider_A,:] =  X_train_stack_A 
    X_train_stack[index_to_consider_B,:] =  X_train_stack_B
    
    X_train_full = np.concatenate((p_X_train_basic,p_X_train_GoQ,p_X_train_GoW,X_train_stack),axis=1)
    X_test_full = np.concatenate((p_X_test_basic,p_X_test_GoQ,p_X_test_GoW,X_test_stack),axis=1)
    
    
    X_train_nn_full = X_train_nn + [X_train_full]
    X_test_nn_full = X_test_nn + [X_test_full]
    
    early_stopping = EarlyStopping(monitor='val_acc', # go through epochs as long as accuracy on validation set increases
                           patience=my_patience,
                           mode='max')    
    
    model_v2 = create_nn_model_v2(max_size,embedding_matrix,word_vector_dim,drop_rate,my_optimizer,X_train_full.shape[1])
    model_v2.fit(X_train_nn_full, 
          y_train,
          batch_size = batch_size,
          epochs = nb_epoch,
          validation_split=0.1,
          callbacks = [early_stopping],
          )
    
    model_gv_v2 = create_nn_model_v2(max_size,glove_embedding,200,drop_rate,my_optimizer,X_train_full.shape[1])
    model_gv_v2.fit(X_train_nn_full, 
          y_train,
          batch_size = batch_size,
          epochs = nb_epoch,
          validation_split=0.1,
          callbacks = [early_stopping],
          )
    
    X_train_nn_stack = X_train_nn + [X_train_stack]
    X_test_nn_stack = X_test_nn + [X_test_stack]

    model_v3 = create_nn_model_v2(max_size,embedding_matrix,word_vector_dim,drop_rate,my_optimizer,X_train_stack.shape[1])
    model_v3.fit(X_train_nn_stack, 
          y_train,
          batch_size = batch_size,
          epochs = nb_epoch,
          validation_split=0.1,
          callbacks = [early_stopping],
          )
    
    model_gv_v3 = create_nn_model_v2(max_size,glove_embedding,200,drop_rate,my_optimizer,X_train_stack.shape[1])
    model_gv_v3.fit(X_train_nn_stack, 
          y_train,
          batch_size = batch_size,
          epochs = nb_epoch,
          validation_split=0.1,
          callbacks = [early_stopping],
          )
    
    

    rf_model = RandomForestClassifier(n_estimators=150)
    elastic_net_model = ElasticNet()
    log_reg_model = LogisticRegression()
    lin_reg_model = LinearRegression()
    km_model = KNeighborsClassifier(n_neighbors=100)
    xgb_model = XGBClassifier(max_depth=6,n_estimators=150, reg_lambda=1,seed=5)
    lightgbm_model = LGBMClassifier(max_depth=6,n_estimators=150, reg_lambda=1,seed=5)



    
    rf_model_basic = RandomForestClassifier(n_estimators=50)
    elastic_net_model_basic = ElasticNet()
    log_reg_model_basic= LogisticRegression()
    lin_reg_model_basic = LinearRegression()
    km_model_basic =KNeighborsClassifier(n_neighbors=50)
    xgb_model_basic = XGBClassifier(max_depth=6,n_estimators=100, reg_lambda=1,seed=5)
    lightgbm_model_basic = LGBMClassifier(max_depth=6,n_estimators=100, reg_lambda=1,seed=5)

    
        
    print('XGB Model')
    xgb_model.fit(X_train_full,y_train)
    xgb_model_basic.fit(X_train_stack,y_train)

    
    print('Light GBM')
    lightgbm_model.fit(X_train_full,y_train)
    lightgbm_model_basic.fit(X_train_stack,y_train)
    
    print('Train RF Model')
    rf_model.fit(X_train_full,y_train)
    rf_model_basic.fit(X_train_stack,y_train)


    print('ElasticNet Model')
    elastic_net_model.fit(X_train_full,y_train)
    elastic_net_model_basic.fit(X_train_stack,y_train)

    
    print('Logistic Reg Model')
    log_reg_model.fit(X_train_full,y_train)
    log_reg_model_basic.fit(X_train_stack,y_train)

    
    print('Linear Reg Model')
    lin_reg_model.fit(X_train_full,y_train)
    lin_reg_model_basic.fit(X_train_stack,y_train)


    
    print('KMeans Model')
    km_model.fit(X_train_full,y_train)
    km_model_basic.fit(X_train_stack,y_train)

    #No Memory in GPU
    outcome_nn_test_v2  = model_v2.predict(X_test_nn_full)
    outcome_nn_test_gv_v2  = model_gv_v2.predict(X_test_nn_full)


    outcome_nn_test_v3  = model_v3.predict(X_test_nn_stack)
    outcome_nn_test_gv_v3  = model_gv_v3.predict(X_test_nn_stack)
    
    keras.backend.clear_session()


    outcome_rf_test = rf_model.predict_proba(X_test_full)[:,1].reshape((-1,1))
    outcome_ada_test = elastic_net_model.predict(X_test_full).reshape((-1,1))
    outcome_log_reg_model_test = log_reg_model.predict_proba(X_test_full)[:,1].reshape((-1,1))
    outcme_lin_model_test = lin_reg_model.predict(X_test_full).reshape((-1,1))
    outcome_kmeans_test = km_model.predict_proba(X_test_full)[:,1].reshape((-1,1))
    #outcome_svm_test = svm_model.predict(X_test_sk).reshape((-1,1))
    outcome_xgb_test = xgb_model.predict_proba(X_test_full)[:,1].reshape((-1,1))
    outcome_lgb_test = lightgbm_model.predict_proba(X_test_full)[:,1].reshape((-1,1))

    outcome_rf_test_basic = rf_model_basic.predict_proba(X_test_stack)[:,1].reshape((-1,1))
    outcome_ada_test_basic = elastic_net_model_basic.predict(X_test_stack).reshape((-1,1))
    outcome_log_reg_model_test_basic = log_reg_model_basic.predict_proba(X_test_stack)[:,1].reshape((-1,1))
    outcme_lin_mode_testl_basic = lin_reg_model_basic.predict(X_test_stack).reshape((-1,1))
    outcome_kmeans_test_basic = km_model_basic.predict_proba(X_test_stack)[:,1].reshape((-1,1))
    #outcome_svm_test_basic = svm_model_basic.predict(X_test_basic).reshape((-1,1))
    outcome_xgb_test_basic = xgb_model_basic.predict_proba(X_test_stack)[:,1].reshape((-1,1))
    outcome_lgb_test_basic = lightgbm_model_basic.predict_proba(X_test_stack)[:,1].reshape((-1,1))

    
    X_test_to_output = np.concatenate([outcome_nn_test_v2,outcome_nn_test_gv_v2,outcome_nn_test_v3,outcome_nn_test_gv_v3,\
                                       outcome_rf_test,outcome_ada_test,outcome_log_reg_model_test,outcme_lin_model_test,\
                                       outcome_kmeans_test,outcome_xgb_test,outcome_lgb_test,outcome_rf_test_basic,\
                                       outcome_ada_test_basic,outcome_log_reg_model_test_basic,outcme_lin_mode_testl_basic,\
                                       outcome_kmeans_test_basic,outcome_xgb_test_basic,outcome_lgb_test_basic],axis=1)

    return X_test_to_output
    
    
    

    

def create_stacking_features(list_pairs_train,y_train,list_pairs_test,data_matrix,ids2ind,texts,\
                             max_size,embedding_matrix,glove_embedding,word_vector_dim,drop_rate,my_optimizer,sequences,word_vectors,data_full_word,p_X_train_basic,p_X_test_basic,p_X_train_GoQ,p_X_test_GoQ,p_X_train_GoW,p_X_test_GoW,\
                             my_p=50,n_lsa=40,my_patience=0,batch_size=64,nb_epoch=20):
    '''
    Compute the stackings features from the given input matrices that are defined
    Param: @list_pairs_train: list of training pairs
    @list_pairs_test: list of test pairs
    @data_full_word: each questions as tokens
    @ids2ind: idx of each questions to indice for data_full_word
    @max_size: max size of word
    @embedding_matrix: word2vec pretrained embedding matrices
    @glove_embedding: glove embedding matrix
    @X_* : precomputed features
    Return: X_train_stacking, X_test_stacking : stacking features for train and test set
    '''
    
    #########Split data in two#########"
        
    X_train_sk,X_test_sk=create_features_lsa_for_sk(list_pairs_train,list_pairs_test\
                               ,ids2ind,texts,n_lsa)
    index_to_consider_A,index_to_consider_B, X_train_basic_A, X_train_basic_B,\
    X_train_GoQ_A,X_train_GoQ_B,X_train_GoW_A,X_train_GoW_B,X_train_sk_A,X_train_sk_B,pairs_A,pairs_B= train_test_split(list(range(len(y_train))),p_X_train_basic,\
                                                                                                        p_X_train_GoQ,p_X_train_GoW,X_train_sk,list_pairs_train,test_size=0.5)
                   
    y_train = np.array(y_train)
    y_train_A = y_train[index_to_consider_A]
    y_train_B = y_train[index_to_consider_B]

                                                                                                    
    print('Computing Neural Netword for w2v features')
    X_train_nn,X_test_nn = create_features_w2v_for_nn(list_pairs_train,list_pairs_test,data_matrix,ids2ind)

    X_train_nn_A = [X_train_nn[0][index_to_consider_A,:],X_train_nn[1][index_to_consider_A,:]]
    X_train_nn_B = [X_train_nn[0][index_to_consider_B,:],X_train_nn[1][index_to_consider_B,:]]

    nb_element_in_A = len(index_to_consider_A) 
    nb_element_in_B = len(index_to_consider_B)
    
    X_train_et_test_nn_A,X_train_et_test_sk_A,X_train_et_test_basic_A,\
    X_train_et_test_GoQ_A= concatenate_all_train_test(X_train_nn_A,X_test_nn,X_train_sk_A,X_test_sk\
                                                                                  ,X_train_basic_A,p_X_test_basic,X_train_GoQ_A,\
                                                                                  p_X_test_GoQ)
    X_train_et_test_nn_B,X_train_et_test_sk_B,X_train_et_test_basic_B,\
    X_train_et_test_GoQ_B= concatenate_all_train_test(X_train_nn_B,X_test_nn,X_train_sk_B,X_test_sk\
                                                                                  ,X_train_basic_B,p_X_test_basic,X_train_GoQ_B,\
                                                                                  p_X_test_GoQ)
    '''
    Outcome_second_level_B = create_second_level_features(pairs_A,y_train_A,pairs_B+list_pairs_test,data_matrix,ids2ind,texts,\
                                                          max_size,embedding_matrix,glove_embedding,word_vector_dim,drop_rate,my_optimizer,sequences,word_vectors,data_full_word,X_train_basic_A,X_train_et_test_basic_B,X_train_GoQ_A,X_train_et_test_GoQ_B,X_train_GoW_A,np.concatenate((X_train_GoW_B,p_X_test_GoW),axis=0))
    
    Outcome_second_level_A = create_second_level_features(pairs_B,y_train_B,pairs_A+list_pairs_test,data_matrix,ids2ind,texts,\
                                                          max_size,embedding_matrix,glove_embedding,word_vector_dim,drop_rate,my_optimizer,sequences,word_vectors,data_full_word,X_train_basic_B,X_train_et_test_basic_A,X_train_GoQ_B,X_train_et_test_GoQ_A,X_train_GoW_B,np.concatenate((X_train_GoW_A,p_X_test_GoW),axis=0))
    
    
    '''
    Output_B = compute_part_of_the_stacking(X_train_nn_A,X_train_sk_A,X_train_basic_A,\
                                 X_train_GoQ_A,y_train_A,X_train_et_test_nn_B,X_train_et_test_sk_B,\
                                 X_train_et_test_basic_B,X_train_et_test_GoQ_B,max_size,embedding_matrix,glove_embedding\
                                 ,word_vector_dim,drop_rate,my_optimizer,batch_size,nb_epoch)                                                                              
    
    Output_A = compute_part_of_the_stacking(X_train_nn_B,X_train_sk_B,X_train_basic_B,\
                             X_train_GoQ_B,y_train_B,X_train_et_test_nn_A,X_train_et_test_sk_A,\
                             X_train_et_test_basic_A,X_train_et_test_GoQ_A,max_size,embedding_matrix,glove_embedding\
                             ,word_vector_dim,drop_rate,my_optimizer,batch_size,nb_epoch) 
    
    
    
    '''
    Output_A = np.concatenate((Output_A,Outcome_second_level_A),axis=1)
    Output_B = np.concatenate((Output_B,Outcome_second_level_B),axis=1)
    '''
    X_test_A = Output_A[nb_element_in_A:,:]
    X_test_B = Output_B[nb_element_in_B:,:]
    X_test_stack = (X_test_A+X_test_B)/2
    
    X_train_stack_A= Output_A[:nb_element_in_A,:]
    X_train_stack_B = Output_B[:nb_element_in_B,:]
    
    X_train_stack= np.zeros((len(y_train),X_train_stack_A.shape[1]))
    X_train_stack[index_to_consider_A,:] =  X_train_stack_A 
    X_train_stack[index_to_consider_B,:] =  X_train_stack_B
    
    return X_train_stack,X_test_stack


