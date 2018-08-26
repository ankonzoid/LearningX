"""

 DT_sklearn_regr.py  (author: Anson Wong / git: ankonzoid)

"""
from sklearn.metrics import mean_squared_error

class DT_sklearn_regr:

    def __init__(self, max_depth=20, min_samples_leaf=10):
        from sklearn.tree import DecisionTreeRegressor
        self.model = DecisionTreeRegressor(max_depth=max_depth,
                                           min_samples_leaf=min_samples_leaf,
                                           criterion="mse")

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def loss(self, X, y, y_pred):
        return mean_squared_error(y, y_pred)