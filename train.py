
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 11:37:52 2023

@author: Redha Ali
"""
# import pandas as pd
import numpy as np
#import scipy
from scipy import io
# from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from keras import optimizers, regularizers
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import ExponentialDecay, CosineDecay
from keras.layers import Input, Dense, ReLU
from keras.models import Model
from keras.callbacks import EarlyStopping
import random



# Define the four dataset sites
dataset_cchmc =  io.loadmat('..\\TensorFlow_data\\cchmc_data_T1_T2.mat')
site1_data = dataset_cchmc['x_data_cchmc_T2_T1']
site1_labels = dataset_cchmc['y_data_cchmc_T2_T1']

dataset_nyu =  io.loadmat('..\\TensorFlow_data\\nyu_data_T1_T2.mat')
site2_data = dataset_nyu['x_data_nyu_T2_T1']
site2_labels = dataset_nyu['y_data_nyu_T2_T1']

dataset_mich =  io.loadmat('..\\TensorFlow_data\\mich_data_T1_T2.mat')
site3_data = dataset_mich['x_data_mich_T2_T1']
site3_labels = dataset_mich['y_data_mich_T2_T1']

dataset_wisc =  io.loadmat('..\\TensorFlow_data\\wisc_data_T1_T2.mat')
site4_data = dataset_wisc['x_data_wisc_T2_T1']
site4_labels = dataset_wisc['y_data_wisc_T2_T1']

# Stack the data and labels from all sites together
# data = np.vstack((site1_data, site2_data, site3_data, site4_data))
# labels = np.hstack((site1_labels, site2_labels, site3_labels, site4_labels))

# Define the number of folds for cross-validation
num_folds = 10
sd=42
CV_site1 = StratifiedKFold(n_splits=num_folds,shuffle=True,random_state=sd)
CV_site2 = StratifiedKFold(n_splits=num_folds,shuffle=True,random_state=sd)
CV_site3 = StratifiedKFold(n_splits=num_folds,shuffle=True,random_state=sd)
CV_site4 = StratifiedKFold(n_splits=num_folds,shuffle=True,random_state=sd)



# Calculate the size of each fold for each site
site1_fold_size = site1_data.shape[0] // num_folds
site2_fold_size = site2_data.shape[0] // num_folds
site3_fold_size = site3_data.shape[0] // num_folds
site4_fold_size = site4_data.shape[0] // num_folds

# Create a list to store the results of each fold
ss=[]
sp=[]
ba=[]
auc=[]
acc=[]


site1_indices = list(CV_site1.split(site1_data, site1_labels))
site2_indices = list(CV_site1.split(site2_data, site2_labels))
site3_indices = list(CV_site1.split(site3_data, site3_labels))
site4_indices = list(CV_site1.split(site4_data, site4_labels))

indx = 0
# Loop over the folds
for i in range(num_folds):
    print(f"Fold {i + 1}...")
    indx += 1
    # Define the start and end indices for the testing fold for each site
    site1_train_idx,site1_test_idx = site1_indices[i]
    site2_train_idx,site2_test_idx = site2_indices[i]
    site3_train_idx,site3_test_idx = site3_indices[i]
    site4_train_idx,site4_test_idx = site4_indices[i]


    # Extract the testing data and labels for each site
    site1_test_data = site1_data[site1_test_idx]
    site1_test_labels = site1_labels[site1_test_idx]

    site2_test_data = site2_data[site2_test_idx]
    site2_test_labels = site2_labels[site2_test_idx]

    site3_test_data = site3_data[site3_test_idx]
    site3_test_labels = site3_labels[site3_test_idx]

    site4_test_data = site4_data[site4_test_idx]
    site4_test_labels = site4_labels[site4_test_idx]

    # Extract the training and validation data and labels for each site
    site1_train_val_data = site1_data[site1_train_idx]
    site1_train_val_labels = site1_labels[site1_train_idx]

    site2_train_val_data = site2_data[site2_train_idx]
    site2_train_val_labels = site2_labels[site2_train_idx]
    
    site3_train_val_data = site3_data[site3_train_idx]
    site3_train_val_labels = site3_labels[site3_train_idx]
    
    site4_train_val_data = site4_data[site4_train_idx]
    site4_train_val_labels = site4_labels[site4_train_idx]
    
    # Define the ratio of samples for each set
    val_ratio = 0.1

    # Split the data and labels into training, validation, and testing sets for each site while preserving the ratio of samples for each site
    # Site #1 Splition
    site1_train_data,site1_val_data,site1_train_labels,site1_val_labels = train_test_split(site1_train_val_data,site1_train_val_labels,
                                                       test_size=val_ratio,
                                                       stratify=site1_train_val_labels,
                                                       random_state=sd)
    # Site #2 Splition
    site2_train_data,site2_val_data,site2_train_labels,site2_val_labels = train_test_split(site2_train_val_data,site2_train_val_labels,
                                                       test_size=val_ratio,
                                                       stratify=site2_train_val_labels,
                                                       random_state=sd)
    # Site #3 Splition
    site3_train_data,site3_val_data,site3_train_labels,site3_val_labels = train_test_split(site3_train_val_data,site3_train_val_labels,
                                                       test_size=val_ratio,
                                                       stratify=site3_train_val_labels,
                                                       random_state=sd)
    # Site #4 Splition
    site4_train_data,site4_val_data,site4_train_labels,site4_val_labels = train_test_split(site4_train_val_data,site4_train_val_labels,
                                                       test_size=val_ratio,
                                                       stratify=site4_train_val_labels,
                                                       random_state=sd)
    
    
    # Stack the data and labels from all sites together
    X_train = np.vstack((site1_train_data, site2_train_data, site3_train_data, site4_train_data))
    #Y_train = np.hstack((site1_train_labels, site2_train_labels, site3_train_labels, site4_train_labels))
    Y_train = np.concatenate((site1_train_labels, site2_train_labels, site3_train_labels, site4_train_labels))
    
    # Stack the data and labels from all sites together
    X_val = np.vstack((site1_val_data, site2_val_data, site3_val_data, site4_val_data))
    #Y_val = np.hstack((site1_val_labels, site2_val_labels, site3_val_labels, site4_val_labels))
    Y_val = np.concatenate((site1_val_labels, site2_val_labels, site3_val_labels, site4_val_labels))
    
    # Stack the data and labels from all sites together
    X_test = np.vstack((site1_test_data, site2_test_data, site3_test_data, site4_test_data))
    #Y_test = np.hstack((site1_test_labels, site2_test_labels, site3_test_labels, site4_test_labels))
    Y_test = np.concatenate((site1_test_labels, site2_test_labels, site3_test_labels, site4_test_labels))
    
    print('The size of training set = ',Y_train.shape)
    print('The size of val set = ',Y_val.shape)
    print('The size of test set = ',Y_test.shape)
    
    
     
    class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(Y_train), y = Y_train.reshape(-1))
    class_weights = dict(enumerate(class_weights))
    class_weights[0] = class_weights[0]+0.3
    
    #lr_schedule = ExponentialDecay(initial_learning_rate=1e-4, decay_steps=540, decay_rate=0.1)
    lr_schedule = CosineDecay(initial_learning_rate=1e-4, decay_steps=540)
    
    #optimizer = tf.optimizers.RMSprop(learning_rate=lr_schedule)
    optimizer = tf.optimizers.Adamax(learning_rate=lr_schedule)
    
    
    callback = EarlyStopping(monitor='val_accuracy', patience=20)
    
    
    def network(X_train,Y_train):
        im_shape=(X_train.shape[1])
        inputs=Input(shape=(im_shape), name='inputs_dnn')
        
        fc1 = Dense(4096, kernel_regularizer=regularizers.L2(0.0001))(inputs)
        relu1 = ReLU()(fc1)
        fc2 = Dense(1024, kernel_regularizer=regularizers.L2(0.0001))(relu1)
        relu2 = ReLU()(fc2)
        fc3 = Dense(256, kernel_regularizer=regularizers.L2(0.0001))(relu2)
        relu3 = ReLU()(fc3)
        main_output = Dense(1,kernel_regularizer=regularizers.L2(0.0001), activation="sigmoid")(relu3)
    
        model = Model(inputs= inputs, outputs=main_output)
        #optimizer =tf.optimizers.RMSprop()
        optimizer =tf.optimizers.Adamax()
        model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics = ['accuracy'])
    
        return(model)

    model = network(X_train,Y_train)   
    
    np.random.seed(sd)
    random.seed(sd)
    tf.random.set_seed(sd)
    
    history = model.fit(X_train, Y_train, epochs=100, batch_size=4, verbose=0, validation_data=(X_val,Y_val), 
                        shuffle=True, callbacks=[callback], validation_freq=1, class_weight=class_weights)

    
    y_pred = model.predict(X_test)

    cm = confusion_matrix(Y_test, np.rint(y_pred))
    TP=cm[0,0]
    FP=cm[1,0]
    FN=cm[0,1]
    TN=cm[1,1]
    auc_test = roc_auc_score(Y_test, y_pred)
    acc_test = accuracy_score(Y_test, np.rint(y_pred))
    
    indx1 = indx -1
    ss.append(TP/(TP+FN))
    sp.append(TN/(TN+FP))
    ba.append((TP/(TP+FN)+TN/(TN+FP))/2)
    auc.append(auc_test)
    acc.append(acc_test)
    
    print(f"\n")
    print(f"\n")
    print(f"------------------------------------------------------------------")
    print('Balance Accuracy = %0.3f' % ba[indx1])
    print('Accuracy = %0.3f' % acc[indx1])
    print('Sensitivity = %0.3f' % ss[indx1])
    print('Specificity = %0.3f' % sp[indx1])
    print('AUC = %0.3f' % auc[indx1])
    print(f"------------------------------------------------------------------")
    print(f"\n")
    print(f"\n")
    

print(f"\n")
print(f"------------------------------------------------------------------")
print('B-A = %0.3f' % np.mean(ba), '± %0.3f' % np.std(ba))
print('Acc = %0.3f' % np.mean(acc), '± %0.3f' % np.std(acc))
print('Sen = %0.3f' % np.mean(ss), '± %0.3f' % np.std(ss))
print('Spe = %0.3f' % np.mean(sp), '± %0.3f' % np.std(sp))
print('AUC = %0.3f' % np.mean(auc), '± %0.3f' % np.std(auc))
print(f"------------------------------------------------------------------")
print(f"\n")

    
    
    


   



    
    
