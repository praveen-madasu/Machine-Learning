# Polynomial Regression Implementation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from itertools import combinations_with_replacement

def import_dataset():
    dataset = pd.read_csv('Position_Salaries.csv')
    X = dataset.iloc[:, 1:2].values
    y = dataset.iloc[:, 2].values
    
    return X, y

def encode_dataset(X):
    labelencoder = LabelEncoder()
    X[:, 1] = labelencoder.fit_transform(X[:, 1])
    
    return X

def split_dataset():
    pass

def create_matrices(X):
    ones = np.ones([X.shape[0], 1])
    X = np.concatenate((ones, X), axis = 1)
    
    return X

def feature_scaling(X):
    X = (X - np.mean(X)) / (np.max(X) - np.min(X))

    return X

def polynomial_features(X, degree):
    n_samples, n_features = np.shape(X)

    def index_combinations():
        combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
        flat_combs = [item for sublist in combs for item in sublist]
        return flat_combs
    
    combinations = index_combinations()
    n_output_features = len(combinations)
    X_new = np.empty((n_samples, n_output_features))
    for i, index_combs in enumerate(combinations):  
        X_new[:, i] = np.prod(X[:, index_combs], axis=1)

    return X_new

def hypothesis(X, beta):  
    hyp = X.dot(beta)
    
    return hyp

def compute_cost(X, y, beta):
    n = len(y)
    h = hypothesis(X, beta)
    loss = h - y
    J = np.sum(np.power(loss, 2)) / (2 * n)
    
    return J

def fit(X, y, degree = 1, numOfIter = 10, alpha = 1e-7):
    n = len(y)
    beta = np.zeros(degree + 1)
    cost_history = [0] * numOfIter
    
    for i in range(numOfIter):
        hyp = hypothesis(X, beta)
        loss = hyp - y
        gradient = X.T.dot(loss) / n
        beta = beta - (alpha * gradient)
        cost = compute_cost(X, y, beta)
        cost_history[i] = cost
        
    return beta, cost_history

def plot_regression_line(X, y, y_pred):
    plt.scatter(X, y, color = 'red')
    plt.plot(X, y_pred, color = 'blue')
    plt.title('Polynomial Regression')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()

def main():
    X, y = import_dataset()
    new_X = polynomial_features(X, 4)
    new_beta, cost_history = fit(new_X, y, 4, 10)
    y_pred = new_X.dot(new_beta)
    plot_regression_line(X, y, y_pred)

if __name__ == '__main__':
    main()