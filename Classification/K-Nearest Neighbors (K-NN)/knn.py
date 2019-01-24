# Implementation of KNN classifier

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def import_dataset():
    dataset = pd.read_csv('Social_Network_Ads.csv')
    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, 4].values
    
    return X, y

def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    
    return X_train, X_test, y_train, y_test

def feature_scaling(X):
    SC = StandardScaler()
    X_train = SC.fit_transform(X)
    
    return X_train

def get_euclidean_distance(X, y, k):
    ecu_dist = np.sqrt(np.sum((X - y)**2 , axis=1))
    
    return np.argsort(ecu_dist)[0:k]

def get_accuracy(y_pred, y_test):
    error = np.sum((y_pred - y_test)**2)
    accuracy = 100 - (error/len(y_test))*100
    
    return accuracy

def predict(X_test, X_train, y_train, k):
    points_labels = []
    
    for point in X_test:
        distances = get_euclidean_distance(X_train, point, k)
        
        results = []
        for index in distances:
            results.append(y_train[index])
            
        label = Counter(results).most_common(1)
        points_labels.append([point, label[0][0]])
        
    return points_labels
    
def get_k_value(X_test, X_train, y_train, y_test):
    acc = []
    for k in range(1, 10):
        y_pred = predict(X_test, X_train, y_train, k)
        predictions = []
        for result in y_pred:
            predictions.append(result[1])
        
        acc.append([get_accuracy(predictions, y_test), k])
    
    return acc

def plot_kvalue(acc):
    plotx = []
    ploty = []
    
    for a in acc:
        plotx.append(a[1])
        ploty.append(a[0])
        
    plt.plot(plotx, ploty)
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.show()
        
def main():
    X, y = import_dataset()
    X = feature_scaling(X)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    acc = get_k_value(X_test, X_train, y_train, y_test)
    plot_kvalue(acc)

if __name__ == '__main__':
    main()