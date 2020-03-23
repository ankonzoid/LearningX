"""

 NN_regr.py  (author: Anson Wong / git: ankonzoid)

"""
import os
from sklearn.metrics import mean_squared_error

class NN_regressor:

    def __init__(self):
        pass

    def fit(self, X, y):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        assert len(X.shape) == 2
        N, d = X.shape

        from keras.models import Sequential
        from keras.layers import Dense
        from keras.optimizers import Adam
        model = Sequential()
        model.add(Dense(10, input_dim=d, activation="relu"))
        model.add(Dense(10, activation="relu"))
        model.add(Dense(1, activation="relu"))
        model.compile(loss="mse", optimizer=Adam(lr=0.005))
        self.model = model

        n_epochs = 100
        self.model.fit(X, y, epochs=n_epochs, verbose=False)

    def predict(self, X):
        return self.model.predict(X)

    def loss(self, X, y, y_pred):
        return mean_squared_error(y, y_pred)
     
    def get_params(self):
        return None
