"""

 DT_sklearn_clf.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np

class DT_sklearn_clf:

    def __init__(self, max_depth=20, min_samples_leaf=10):
        from sklearn.tree import DecisionTreeClassifier
        self.model = DecisionTreeClassifier(max_depth=max_depth,
                                            min_samples_leaf=min_samples_leaf,
                                            criterion="gini", splitter="best")

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def loss(self, X, y, y_pred):
        return gini_impurity(y)

def gini_impurity(y):
    p2 = 0.0
    y_classes = list(set(y))
    for c in y_classes:
        p2 += (np.sum(y == c) / len(y))**2
    loss = 1.0 - p2
    return loss