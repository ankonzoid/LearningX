"""

 logistic_regression_GD.py  (author: Anson Wong / git: ankonzoid)

 * NOT FINISHED * 
 Should follow: https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html

"""
import os
import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

class LogisticRegressorGD():

    def __init__(self):
        pass

    def fit(self, X, y):
        #
        # Use binary cross-entropy loss instead of MSE
        #
        # L = -(1/N) * sum[i=1, N] (y*log(y_hat) + (1-y)*log(1-y_hat))
        #   = -(1/N) * sum[i=1, N] (y*log(sigma(x)) + (1-y)*log(1-sigma(x)))
        #
        # y_hat = sigma(x)
        #
        # dL/daj = -(1/N) * sum[i=1, N] * y * xj for j=1,2,...,d
        # dL/daj = -(1/N) * sum[i=1, N] * y for j=0
        #
        # GD: a{i+1} <- a{i} - \eta * dL/dxj
        #
        print("Fitting...")

        # Initialize fit parameters [a0, a1, a2, ..., ad] where a0 is y-intercept
        N, d = X.shape
        a_fit = np.random.normal(size=d+1)

        # Make gradient descent updates
        loss_tolerance = 1E-4
        fit_tolerance = 1E-4
        eta = 1E-6  # learning rate
        converge = False
        loss = 9E99
        while not converge:

            # Compute gradient
            gradient = np.zeros(d+1)
            for i in range(N):
                t = a_fit[0] + np.dot(a_fit[1:], X[i])
                y_i = y[i]
                gradient[0] += -(y_i-sigmoid(t))
                gradient[1:] += -(y_i-sigmoid(t))*X[i, 0:]
            gradient /= N

            # Perform gradient descent step
            a_fit_new = a_fit - eta * gradient

            # Compute loss (to keep track)
            loss_new = 0.0
            y_pred = []
            for x_i, y_i in zip(X, y):
                t = a_fit_new[0] + np.dot(a_fit_new[1:], x_i)
                y_pred_i = sigmoid(t)
                y_pred.append(y_pred_i)
                loss_new += -(y_i*np.log(y_pred_i) + (1-y_i)*np.log(1-y_pred_i))
            loss_new /= N
            print("loss = {}".format(loss))

            # See if loss and fit parameters have converged
            if np.abs(loss_new - loss) < loss_tolerance and np.linalg.norm(a_fit_new - a_fit) < fit_tolerance:
                converge = True

            # Update fit parameters
            a_fit = a_fit_new
            loss = loss_new

        # Save fit parameters
        self.a_fit = a_fit

    def predict(self, X):
        y_pred = []
        for x in X:
            t = self.a_fit[0] + np.dot(self.a_fit[1:], x)
            y_pred_i = sigmoid(t)
            y_pred.append(y_pred_i)
        y_pred = np.array(y_pred)
        return y_pred

# Main Driver
if __name__ == "__main__":
    X = np.arange(0, 100).reshape(-1, 1)
    y = np.concatenate([np.zeros(30), np.ones(70)])

    exit("Current implementation unfinished! Next version...")

    model = LogisticRegressorGD()
    model.fit(X, y)
    y_pred = model.predict(X)

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.plot(X.flatten(), y, 'o')
    plt.plot(X.flatten(), y_pred, 'r')
    plt.show()
