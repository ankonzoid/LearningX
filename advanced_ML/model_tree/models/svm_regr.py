"""

 svm_regr.py  (author: Anson Wong / git: ankonzoid)

"""
from sklearn.metrics import mean_squared_error

class svm_regr:

    def __init__(self):
        from sklearn.svm import SVR
        self.model = SVR()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def loss(self, X, y, y_pred):
        return mean_squared_error(y, y_pred)
     
    def get_params(self):
        return None
