"""

 logistic_regression_GD.py  (author: Anson Wong / git: ankonzoid)

"""
import os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def split(fracs, N, seed):
    def is_intersect(arr1, arr2):
        n_intersect = len(np.intersect1d(arr1, arr2))
        if n_intersect == 0: return False
        else: return True
    fracs = [round(frac, 2) for frac in fracs]
    if sum(fracs) != 1.00:
        raise Exception("fracs do not sum to one!")
    indices = list(range(N))
    random.Random(seed).shuffle(indices)
    indices = np.array(indices, dtype=int)
    n_fracs = []
    for i in range(len(fracs) - 1):
        n_fracs.append(int(max(fracs[i] * N, 0)))
    n_fracs.append(int(max(N - sum(n_fracs), 0)))
    if sum(n_fracs) != N:
        raise Exception("n_fracs do not sum to N!")
    n_selected = 0
    indices_fracs = []
    for n_frac in n_fracs:
        indices_frac = indices[n_selected:n_selected + n_frac]
        indices_fracs.append(indices_frac)
        n_selected += n_frac
    for a, indices_frac_A in enumerate(indices_fracs):
        for b, indices_frac_B in enumerate(indices_fracs):
            if a == b:
                continue
            if is_intersect(indices_frac_A, indices_frac_B):
                raise Exception("there are intersections!")
    return indices_fracs

class LogisticRegressorGD():

    def __init__(self):
        pass

    def fit(self, X, y, X_tune, y_tune):
        #
        # Use binary cross-entropy loss instead of MSE
        #
        # L = -(1/N) * sum[i=1, N] (y*log(y_hat) + (1-y)*log(1-y_hat))
        #   = -(1/N) * sum[i=1, N] (y*log(sigma(a*x + a0)) + (1-y)*log(1-sigma(a*x + a0)))
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
        eta = 10  # learning rate
        converge = False
        loss = 9E99
        while not converge:

            # Compute gradient
            gradient = np.zeros(d+1)
            for i in range(N):
                y_pred_i = sigmoid(a_fit[0] + np.dot(a_fit[1:], X[i]))
                gradient[0] += (y_pred_i - y[i])
                gradient[1:] += (y_pred_i - y[i]) * X[i, 0:]
            gradient /= N

            # Perform gradient descent step
            a_fit_new = a_fit - eta * gradient

            # Compute loss (to keep track)
            y_pred = np.array([sigmoid(a_fit_new[0] + np.dot(a_fit_new[1:], x)) for x in X_tune])
            loss_new = log_loss(y_tune, y_pred)
            print("loss = {}".format(loss))

            # See if loss and fit parameters have converged
            if np.abs(loss_new - loss) < loss_tolerance and loss_new > loss_tolerance:
                converge = True

            # Update fit parameters
            a_fit = a_fit_new
            loss = loss_new

        # Save fit parameters
        self.a_fit = a_fit

    def predict(self, X):
        y_pred = np.array([sigmoid(self.a_fit[0] + np.dot(self.a_fit[1:], x)) for x in X])
        return y_pred

def plot_sorted(X_train, y_pred_train, X_valid, y_pred_valid):
    idx_train_sorted = np.argsort(X_train.flatten())
    idx_valid_sorted = np.argsort(X_valid.flatten())
    plt.plot(X_train[idx_train_sorted].flatten(), y_pred_train[idx_train_sorted], 'r--')
    plt.plot(X_valid[idx_valid_sorted].flatten(), y_pred_valid[idx_valid_sorted], 'k-')

# Main Driver
if __name__ == "__main__":

    random.seed(10)
    N0, N1 = 400, 400
    N = N0 + N1
    X = np.array([random.uniform(0, 0.6) for _ in range(N0)] + [random.uniform(0.4, 1) for _ in range(N1)]).reshape(-1, 1)
    y = np.array([0 for _ in range(N0)] + [1 for _ in range(N1)], dtype=int)
    assert len(X) == len(y)

    indices_valid, indices_train = split(fracs=[0.2, 0.8], N=N, seed=1)
    X_train, y_train = X[indices_train], y[indices_train]
    X_valid, y_valid = X[indices_valid], y[indices_valid]

    model = LogisticRegressorGD()
    model.fit(X, y, X_tune=X_valid, y_tune=y_valid)
    y_pred_train = model.predict(X_train)
    y_pred_valid = model.predict(X_valid)
    bce_train = log_loss(y_train, y_pred_train)
    bce_valid = log_loss(y_valid, y_pred_valid)
    print("bce_train =", bce_train)
    print("bce_valid =", bce_valid)

    plt.figure(1)
    plt.plot(X.flatten(), y, 'o')
    plot_sorted(X_train, y_pred_train, X_valid, y_pred_valid)
    plt.show()
