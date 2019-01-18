# Implementing Simple Linear Regression

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
def import_dataset():
    dataset = pd.read_csv('Salary_Data.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values
    return X, y

# Estimate Coefficients
def estimate_coef(X, y):
    # number of observations
    n = np.size(X)
    
    # Mean for calculating y pred
    m_x, m_y = np.mean(X), np.mean(y)

    covar = 0.0
    var = 0.0
    for i in range(n):
        covar += (X[i] - m_x) * (y[i] - m_y)
    
    var = np.sum(np.power((values - m_x), 2) for values in X)
    
    # calculating coefficients
    b1 = covar / var
    b0 = m_y - b1 * m_x
    
    return (b0, b1)
    
def plot_regression_line(x, y, b):
    plt.scatter(x, y, color = 'red')
    
    # Expected hypothesis
    y_pred = b[0] + b[1] * x
    
    plt.plot(x, y_pred, color = 'green')
    plt.title('Salary vs Experience')
    plt.xlabel('Experience')
    plt.ylabel('Salary')
    plt.legend()
    plt.show()

def main():
    X, y = import_dataset()
    b = estimate_coef(X, y)
    print("Estimated Coefficients: \n b0 = {} \n b1 = {}".format(b[0], b[1]))
    
    plot_regression_line(X, y, b)
    
if __name__ == "__main__":
    main()
