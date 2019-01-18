# Implementation of Logistic Regression Classifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

def import_dataset():
    dataset = pd.read_csv("Social_Network_Ads.csv")
    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, 4].values
    
    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    
    return X_train,X_test, y_train, y_test

def create_matrix(X):
    ones = np.ones([X.shape[0], 1])
    X = np.concatenate((ones, X), axis = 1)
    
    return X

def feature_scaling(X):
    SC = StandardScaler()
    X_train = SC.fit_transform(X)
    
    return X_train

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def hypothesis(X, beta):
    z = X.dot(beta)
    hyp = 1 / (1 + np.exp(-z))
    
    return hyp

def compute_cost(X, y, beta):
    n_samples = len(y)
    hyp = hypothesis(X, beta)
    J = (-y * np.log(hyp) - (1 - y) * np.log(1 - hyp)) / n_samples
    
    return J

def fit(X, y, alpha = 0.001, iterations = 10):
    n_samples = len(y)
    beta = np.array([0, 0, 0])
    cost_history = [0] * iterations
    
    for i in range(iterations):
        hyp = hypothesis(X, beta)
        loss = hyp - y
        gradient = X.T.dot(loss) / n_samples
        beta = beta - alpha * gradient
        cost = compute_cost(X, y, beta)
        cost_history[i] = cost

    return beta, cost_history

def predict(X, beta):
    # we don't need the 1st column which is for b0
    z = X.dot(beta[1:])
    
    return sigmoid(z)

def get_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
        
    print(cm)

def plot_results(X, y, beta, b_training = 'false'):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == 0][:, 1], X[y == 0][:, 2], color='r', label='0')
    plt.scatter(X[y == 1][:, 1], X[y == 1][:, 2], color='g', label='1')
    plt.legend()
    
    if(b_training == 'true'):
        plt.title('Logistic Regression (Training set)')
    else:
        plt.title('Logistic Regression (Test set)')
    
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    x1_min, x1_max = X[:,1].min(), X[:,1].max(),
    x2_min, x2_max = X[:,2].min(), X[:,2].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    probs = predict(grid, beta).reshape(xx1.shape)
    plt.contour(xx1, xx2, probs, [0.5], linewidths=2, colors='black')

def main():
    X, y = import_dataset()
    X = feature_scaling(X)
    
    # Adding one more column for beta0(b0)
    X = create_matrix(X)
    X_train, X_test, y_train, y_test = split_data(X, y)
    beta, cost_history = fit(X_train, y_train, 0.001, 1000)
    
    # Plot training set results
    plot_results(X_train, y_train, beta, 'true')
    
    # Plot test set results
    plot_results(X_test, y_test, beta, 'false')
    get_confusion_matrix(y_test, np.round(predict(X_test[:, 1:], beta)).astype(int))

if __name__ == "__main__":
    main()