"""

 GradientBoostedTree.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostedTreeRegressor:

    def __init__(self, n_estimators=20, max_depth=5,
                 min_samples_split=2, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        """
        Method:
         1) Train tree with greedy splitting on dataset (X, y)
         2) Recursively (n_estimator times) compute the residual between the
            truth and prediction values (res = y-y_pred), and use the residual as
            the next estimator's training y (keep X the same)
         3) The prediction at each estimator is the trained prediction plus
            the previous trained prediction, making the full prediction of the final
            model the sum of the predictions of each model.
        """
        self.models = []
        y_i = y
        y_pred_i = np.zeros(y.shape)
        for i in range(self.n_estimators):
            # Create tree with greedy splits
            model = DecisionTreeRegressor(max_depth=self.max_depth,
                                          min_samples_split=self.min_samples_split,
                                          min_samples_leaf=self.min_samples_leaf)
            model.fit(X, y_i)
            # Boosting procedure
            y_pred = model.predict(X) + y_pred_i  # add previous prediction
            res = y - y_pred  # compute residual
            y_i = res  # set training label as residual
            y_pred_i = y_pred  # update prediction value
            self.models.append(model)

    def predict(self, X):
        y_pred = np.zeros(len(X))
        for model in self.models:
            y_pred += model.predict(X)
        return y_pred