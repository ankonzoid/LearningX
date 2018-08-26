"""

 RandomForestRegressor.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class RandomForestTreeRegressor:

    def __init__(self, n_estimators=20, max_depth=5,
                 min_samples_split=2, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        """
        Method:
         1) Create n_estimator tree estimators with random splitting
            on d/3 max features
         2) For each estimator i, sample (X, y) randomly N times with replacement
            and train the estimator on this sampled dataset (X_i, y_i)
         3) Predict by taking the mean predictions of each estimator
        """
        self.models = []
        N, d = X.shape
        for i in range(self.n_estimators):
            # Create tree with random splitting on d/3 max features
            model = DecisionTreeRegressor(max_depth=self.max_depth,
                                          min_samples_split=self.min_samples_split,
                                          min_samples_leaf=self.min_samples_leaf,
                                          splitter="random",
                                          max_features=d/3)
            # Bagging procedure
            idx_sample = np.random.choice(N, N)  # random sampling of length N
            model.fit(X[idx_sample], y[idx_sample])  # fit on random sampling
            self.models.append(model)

    def predict(self, X):
        y_pred = np.zeros(len(X))
        for model in self.models:
            y_pred += model.predict(X)
        y_pred /= self.n_estimators
        return y_pred