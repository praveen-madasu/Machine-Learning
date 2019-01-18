# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 14:20:39 2019

@author: prave
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def import_dataset():
    dataset = pd.read_csv('50_Startups.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 4].values

    return X, y

# Encoding the categorical data
def encode_data(X):
    labelencoder = LabelEncoder()
    X[:, 3] = labelencoder.fit_transform(X[:, 3])
    onehotencoder = OneHotEncoder(categorical_features = [3])
    X = onehotencoder.fit_transform(X).toarray()
    
    # Avoiding dummy vatiable trap
    X = X[:, 1:]
    
    return X

def feature_scaling(X):
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X[:, 2:] = sc_X.fit_transform(X[:, 2:])
    
    return X

def create_matrices(X):
    ones = np.ones([X.shape[0], 1])
    X = np.concatenate((ones, X), axis = 1)
    beta = np.array([0, 0, 0, 0, 0, 0])
    alpha = 0.01
    
    return X, alpha, beta

def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    return X_train, X_test, y_train, y_test

def cost_function(X, y, beta):
    n = len(y)
    J = np.sum(np.power((X.dot(beta) - y), 2)) / (2 * n)
    
    return J

def gradient_decent(X, y, alpha, beta, iterations):
    cost_history = [0] * iterations
    n = len(y)
    
    for iteration in range(iterations):  
        # Calculate Hypothesis
        hypothesis = X.dot(beta)
        #print(np.shape(X), " ", np.shape(beta), " ", np.shape(hypothesis), " ", np.shape(y))
        # Loss = difference between hypothesis & actual Y
        loss = hypothesis - y

        # Gradient calculation
        gradient = X.T.dot(loss) / n

        # changing values of beta using gradient decent
        beta = beta - alpha * gradient
        
        # New cost value
        cost = cost_function(X, y , beta)
        cost_history[iteration] = cost
    
    return beta, cost_history

def main():
    X, y = import_dataset()
    X = encode_data(X)
    
    X = feature_scaling(X)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    X_test, alpha, beta = create_matrices(X_test)
    
    cost_function(X_test, y_test, beta)
    new_beta , cost_history = gradient_decent(X_test, y_test, alpha, beta, 1)
    
    y_pred = X_test.dot(new_beta)
    print(y_pred)    

if __name__ == "__main__":
    main()