"""

 linear_regression_GD.py  (author: Anson Wong / git: ankonzoid)

"""
import os
import numpy as np
import pandas as pd

class LinearRegressorGD():

    def __init__(self):
        pass

    def fit(self, X, y):
        #
        # L = sum[i=1, N] (y_hat - y)^2
        #
        # y_hat = a0 + a1*x1 + a2*x2 + ... + ad*xd
        #
        # dL/dxj = sum[i=1, N] 2*(y_hat_i - y_i)*d(y_hat_i - y_i)/daj
        #        = 2*sum[i=1, N] xj*(y_hat_i - y_i)    for   j=1,2,...,d
        #
        # dL/dxj = 2*sum[i=1, N] (y_hat_i - y_i)    for    j=0
        #
        # GD: a{i+1} <- a{i} - \eta * dL/dxj
        #
        print("Fitting...")

        # Initialize fit parameters [a0, a1, a2, ..., ad] where a0 is y-intercept
        N, d = X.shape
        a_fit = np.random.normal(size=d+1)

        # Make gradient descent updates
        loss_tolerance = 1E-2
        fit_tolerance = 1E-4
        eta = 1E-6  # learning rate
        converge = False
        loss = 9E99
        while not converge:

            # Compute gradient
            gradient = np.zeros(d+1)
            for i in range(N):
                y_hat_i = a_fit[0] + np.dot(a_fit[1:], X[i])
                y_i = y[i]
                gradient[0] += 2*(y_hat_i - y_i)
                gradient[1:] += 2*(y_hat_i - y_i)*X[i, 0:]
            gradient /= N

            # Perform gradient descent step
            a_fit_new = a_fit - eta * gradient

            # Compute loss (to keep track)
            y_pred = []
            for x in X:
                y_pred_i = a_fit_new[0] + np.dot(a_fit_new[1:], x)
                y_pred.append(y_pred_i)
            y_pred = np.array(y_pred)
            loss_new = np.linalg.norm(y_pred - y) ** 2 / N
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
            y_pred_i = self.a_fit[0] + np.dot(self.a_fit[1:], x)
            y_pred.append(y_pred_i)
        y_pred = np.array(y_pred)
        return y_pred

# Main Driver
if __name__ == "__main__":
    X = np.arange(100).reshape(-1, 1)
    y = .4 * X.flatten() + 3 + np.random.uniform(-10, 10, size=(100,))

    model = LinearRegressorGD()
    model.fit(X, y)
    y_pred = model.predict(X)

    mse = ((y_pred - y) ** 2).mean(axis=0)
    print("mse =", mse)

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.plot(X.flatten(), y, 'o')
    plt.plot(X.flatten(), y_pred, 'r')
    plt.show()
