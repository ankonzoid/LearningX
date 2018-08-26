"""

 modal_clf.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np
from scipy import stats

class modal_clf:

    def __init__(self):
        self.y_mode = None

    def fit(self, X, y):
        self.y_mode = stats.mode(y, axis=None)[0][0]

    def predict(self, X):
        assert self.y_mode is not None
        return self.y_mode * np.ones(len(X), dtype=int)

    def loss(self, X, y, y_pred):
        return gini_impurity(y)

def gini_impurity(y):
    p2 = 0.0
    y_classes = list(set(y))
    for c in y_classes:
        p2 += (np.sum(y == c) / len(y))**2
    loss = 1.0 - p2
    return loss