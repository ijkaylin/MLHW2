#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Machine Learning HW 2: online perceptron

@author: Kathy
"""

# -*- coding: utf-8 -*-
"""
hw1funcs.py
"""

import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from scipy.sparse import csr_matrix, hstack


def unigram(dataframe): 
    """
    Returns unigram representation of data
    @PARAMETERS: 
        - dataframe: a pandas dataframe of training data
    @RETURNS: 
        - feats: 2D sparse matrix for given dataframe
        - names: dictionary of features of the dataframe
    """
    #split the texts into tokens
    vec =  CountVectorizer(dtype=float)
    feats = vec.fit_transform(dataframe.text)
    names = vec.vocabulary_
    return feats,names

def tdidf(dataframe): 
    """
    Returns td-idf representation of data
    @PARAMETERS: 
        - dataframe: a pandas dataframe of training data
    @RETURNS: 
        - feats_idf: 2D sparse matrix for given dataframe
        - names: dictionary of features of the dataframe
    """
    vec =  CountVectorizer(dtype=float)
    feats = vec.fit_transform(dataframe.text)
    
    tfidf_transformer = TfidfTransformer()
    feats_idf = tfidf_transformer.fit_transform(feats)
    names = vec.vocabulary_
    return feats_idf,names
    

def bigram(dataframe): 
    """
    Returns bigram representation of data
    @PARAMETERS: 
        - dataframe: a pandas dataframe of training data
    @RETURNS: 
        - feats: 2D sparse matrix for given dataframe
        - names: dictionary of features of the dataframe
    """
    vec =  CountVectorizer(ngram_range=(1,2))
    feats = vec.fit_transform(dataframe.text)
    names = vec.vocabulary_
    return feats,names

def stopwords(dataframe): 
    """
    Returns stop words representation of data
    @PARAMETERS: 
        - dataframe: a pandas dataframe of training data
     @RETURNS: 
        - feats: 2D sparse matrix for given dataframe
        - names: dictionary of features of the dataframe
    """
    my_stop_words = text.ENGLISH_STOP_WORDS
    
    vec =  TfidfVectorizer(ngram_range=(1,1), stop_words = my_stop_words)
    feats = vec.fit_transform(dataframe.text)
    names = vec.vocabulary_
    return feats,names

def perceptron(training, trainlabels): 
    """
    Runs the online perceptron algorithm on data twice, shuffling the data between
    passes through the training data
    @PARAMETERS: 
        - training: scipy.sparse.csr_matrix -- representing the training 
        data with bias added
        
        - trainlabels: pd.Series containing the corresponding labels of the 
        training data 
    @RETURNS:
        - w_final : Numpy array:  final weight vector of dimension training.shape[1]
    """
    #Initialize weight vector to 0 
    nrow,ncol = training.shape
    w = np.zeros(ncol,dtype=np.float)
    total= np.zeros(ncol,dtype=np.float)
    counter = 0
    index = np.arange(np.shape(training)[0])

    
    while counter < 2: 
        #Randomly shuffle data and obtain reordered training/labels
        np.random.shuffle(index)
        training = training[index, :]
        labels = trainlabels.iloc[index]
        
        for i in range(nrow):
            y = labels[i]
            cur_row = training[i,:].A
            dotprod = cur_row.dot(w.T)
            if y*dotprod <=0.0: 
                w = w+ y*cur_row
            if counter == 1: 
                 total = np.add(total, w)
#                summed = np.add(total, w)
        counter+=1
        
#    summed = nps.array(summed)
#    wfinal= (1/(float(nrow+1)))*summed
#    wfinal = wfinal.ravel()
    wfinal= (1/(float(nrow+1)))*total
    return wfinal

def trainingerror(training, trainlabels, wfinal): 
    """
    Returns the training error rate for a given training data matrix and weight
    vector
    
    @PARAMETERS: 
        - training: scipy.sparse.csr_matrix -- representing the training 
        data with bias added
        -trainlabels: Pandas series: containing the actual predictions
        - wfinal: (1,num_features) Numpy array 
        
    @RETURNS:
        - error: Float -- rate of misclassification
    
    """
    #Create np float array product
    product = training.dot(wfinal.T)
    preds= np.where(product>=0.0, 1, -1)
    hits = (preds == trainlabels).sum()
    error = 1- (hits / training.shape[0])
    return error

def overlap(traindict, testdict, wfinal):
    """
    Identifies which features overlap between the training and test set and sets
    values of nonoverlap features in w_final to 0
    
    @PARAMETERS: 
        - traindict: dictionary of features of training data
        
        - testdict: dictionary of features of test data
        
        - wfinal: (1,num_features of training) Numpy array 
        
    @RETURNS:
        - final: (1,num_features of test) Numpy array 
        
    """
    train_keys = set(traindict.values())
    test_keys = set(testdict.values())
    deletethese = train_keys - test_keys
    deletethese = list(deletethese)

    final = np.delete(wfinal.T, deletethese, axis = 0)
    return final
    
    
def testerror(test, testlabels, wfinaltest): 
    """
    Returns the training error rate for a given training data matrix and weight
    vector
    
    @PARAMETERS: 
        - test: scipy.sparse.csr_matrix -- representing the test data
        - testlabels: pd series of labels from test data
        - wfinal: (1,num_features) Numpy array 
        
    @RETURNS:
        - error: Float -- rate of misclassification
    
    """
    
    product = test.dot(wfinaltest)
    preds= np.where(product>=0.0, 1, -1)
    preds = preds.ravel()
    hits = (preds == testlabels).sum()
    error = 1- (hits / float(test.shape[0]))
    return error

#a = trainingerror(train_uni_mat_test, train_labels_test, w_final)
#b = testerror(test_uni_mat, test_labels, w_final_test)
