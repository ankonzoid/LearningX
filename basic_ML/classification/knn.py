"""

 knn.py  (author: Anson Wong / git: ankonzoid)

"""
import os
import numpy as np
import pandas as pd

class KNN():

    def __init__(self, k=10):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # Assign class labels to data based on closest distance to cluster centers
        N, d = X.shape
        y_pred = -1 * np.zeros(len(X), dtype=int)
        for i in range(N):
            neg_distances = -np.array([np.linalg.norm(X[i] - x_train_j) for x_train_j in self.X_train])
            idx_topk = neg_distances.argsort()[-self.k:][::-1]  # top-k samples in negative distance
            nearby_classes = list(self.y_train[idx_topk])
            y_pred[i] = int(max(set(nearby_classes), key=nearby_classes.count))  # predict modal class
        return y_pred

# Main driver
if __name__ == "__main__":
    
    def load_csv_data(data_csv_filename, mode="clf", verbose=False):
        if verbose:
            print("Loading data from '{}' (mode={})...".format(data_csv_filename, mode))
        df = pd.read_csv(data_csv_filename)  # dataframe
        df_header = df.columns.values  # header
        header = list(df_header)
        N, d = len(df), len(df_header) - 1
        X = np.array(df.drop(['y'], axis=1))  # extract X by dropping y column
        y_raw = np.array(df['y'])  # extract y

        y_classes_raw = sorted(list(set(y_raw)))
        y_classes_dict = {}
        for i, y_class_i in enumerate(y_classes_raw):
            y_classes_dict[y_class_i] = i

        y = np.array([y_classes_dict[y_i] for y_i in y_raw])
        y_classes = list(set(y))
        assert X.shape == (N, d)  # check X.shape
        assert y.shape == (N,)  # check y.shape
        if mode == "clf":
            assert y.dtype in ['int64']  # check y are integers
        elif mode == "regr":
            assert y.dtype in ['int64', 'float64']  # check y are integers/floats
        else:
            exit("err: invalid mode given!")
        if verbose:
            print(" header={}\n X.shape={}\n y.shape={}\n len(y_classes)={}\n".format(header, X.shape, y.shape, len(y_classes)))
        return X, y, header

    data_csv_data_filename = os.path.join("data", "iris.data.txt")
    X, y, header = load_csv_data(data_csv_data_filename, mode="clf", verbose=True)
    model = KNN(k=5)
    model.fit(X, y)
    y_pred = model.predict(X)
    print("acc_train =", np.sum(y_pred==y)/len(y))
