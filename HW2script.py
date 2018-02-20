#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Script to generate HW2 answers
"""

import HW2Funcs as F
import numpy as np 
import pandas as pd
from scipy.sparse import csr_matrix, hstack


def problem1(): 
    
    train = pd.read_csv('reviews_tr.csv')
    test = pd.read_csv('reviews_te.csv')
    #create subset of half of the training data
    train_sub= train.iloc[:500000].copy()
    
    train_sub['label'].replace(0,-1,inplace=True)
    test['label'].replace(0,-1,inplace=True)
    
    train_labels = train_sub['label']
    test_labels = test['label']
    
    
    ############## Unigram Representation ################
    #Create the TF training matrix
    train_unigram, train_features = F.unigram(train_sub)
    test_unigram, test_features = F.unigram(test)
    
    #Add bias to training data and test
    bias_train = np.ones((500000,1))
    bias_test = np.ones((320122,1))
    
    train_uni_mat = hstack((train_unigram, bias_train))
    test_uni_mat = hstack((test_unigram, bias_test))
    
    train_uni_mat = train_uni_mat.tocsr()
    test_uni_mat = test_uni_mat.tocsr()
    
    w_final = F.perceptron(train_uni_mat, train_labels)
    print("The training error for unigram representation is " +\
        str(F.trainingerror(train_uni_mat, train_labels, w_final)))
    
    w_final_test = F.overlap(train_features, test_features, w_final)
    
    print ("The test error for unigram representation is " +\
        str(F.testerror(test_uni_mat, test_labels, w_final_test)))
        
        

    ##### Find the words with the largest weights
    ind = np.argpartition(w_final_test, -10)[-10:]
    ind = list(ind)
    maximums = [train_features.keys()[train_features.values().index(i)] for i in ind]
    maximums.sort()
    print(maximums)

    
    negind = np.argpartition(w_final_test, 10)[-10::]
    negind = list(negind)
    minimums = [train_features.keys()[train_features.values().index(i)] for i in negind]
    minimums.sort()
    print(minimums)
        
#    ############### TD-IDF REPRESENTATION ###############
#    train_idf, train_features = F.tdidf(train_sub)
#    test_idf, test_features = F.tdidf(test)
#    
#    #Add bias to training data and test
#    bias_train = np.ones((500000,1))
#    bias_test = np.ones((320122,1))
#    
#    train_idf_mat = hstack((train_idf, bias_train))
#    test_idf_mat = hstack((test_idf, bias_test))
#    
#    train_idf_mat = train_idf_mat.tocsr()
#    test_idf_mat = test_idf_mat.tocsr()
#    
##    train_idf_mat_test = train_idf_mat[0:100,:]
##    train_labels_test = train_labels.iloc[0:100]
#    
#    w_final = F.perceptron(train_idf_mat, train_labels)
#    print("The training error for idf representation is " +\
#        str(F.trainingerror(train_idf_mat, train_labels, w_final)))
#        
#    w_final = w_final.T
#    
#    w_final_test = F.overlap(train_features, test_features, w_final)
#    
#    print("The test error for idf representation is " +\
#        str(F.testerror(test_idf_mat, test_labels, w_final_test)))
#    
#            
#    ############### Bigram  REPRESENTATION ###############
#    train_bi, train_features = F.bigram(train_sub)
#    test_bi, test_features = F.bigram(test)
#    
#    #Add bias to training data and test
#    bias_train = np.ones((500000,1))
#    bias_test = np.ones((320122,1))
#    
#    train_bi_mat = hstack((train_bi, bias_train))
#    test_bi_mat = hstack((test_bi, bias_test))
#    
#    train_bi_mat = train_bi_mat.tocsr()
#    test_bi_mat = test_bi_mat.tocsr()
#    
##    train_bi_mat_test = train_bi_mat[0:100,:]
##    train_labels_test = train_labels.iloc[0:100]
#    
#    w_final = F.perceptron(train_bi_mat, train_labels)
#    print("The training error for bigram representation is " +\
#        str( F.trainingerror(train_bi_mat,train_labels, w_final)))
#    
#    w_final_test = F.overlap(train_features, test_features, w_final)
#    
#    print("The test error for bigram representation is " +\
#        str(F.testerror(test_bi_mat, test_labels, w_final_test)))
#        
#    ############### Fourth Representation: Porter Stemmer ###############
#    train_stop, train_features = F.stopwords(train_sub)
#    test_stop, test_features = F.stopwords(test)
#    
#    #Add bias to training data and test
#    bias_train = np.ones((500000,1))
#    bias_test = np.ones((320122,1))
#    
#    train_stop_mat = hstack((train_stop, bias_train))
#    test_stop_mat = hstack((test_stop, bias_test))
#    
#    train_stop_mat = train_stop_mat.tocsr()
#    test_stop_mat = test_stop_mat.tocsr()
#    
#    train_stop_mat_test = train_stop_mat[0:100000,:]
#    train_labels_test = train_labels.iloc[0:100000]
#    
#    
#    w_final = F.perceptron(train_stop_mat_test, train_labels_test)
#    print("The training error for unigram representation is " +\
#        str(F.trainingerror(train_stop_mat_test, train_labels_test, w_final)))
#    
#    w_final_test = F.overlap(train_features, test_features, w_final)
#    
#    print ("The test error for unigram representation is " +\
#        str(F.testerror(test_uni_mat, test_labels, w_final_test)))
#        
    


problem1()