"""

 kmeans.py  (author: Anson Wong / git: ankonzoid)

"""
import os
import numpy as np
import pandas as pd

class KMeans():

    def __init__(self, k=4):
        self.k = k

    def fit(self, X):
        N, d = X.shape
        self.cluster_centers = X[np.random.choice(N, self.k, replace=False)]
        converge = False
        y = -1 * np.zeros(len(X))
        while not converge:
            # Create class dictionary
            class_dict = {}
            for k in range(self.k):
                class_dict[k] = []

            # Assign class labels to data based on closest distance to cluster centers
            for i in range(N):
                distances = [np.linalg.norm(X[i] - self.cluster_centers[k]) for k in range(self.k)]
                c = np.argmin(distances)
                y[i] = c
                class_dict[c].append(i)

            # Find center of mass
            cluster_centers_new = []
            for k in range(self.k):
                cluster_center_new = np.mean(X[class_dict[k]], axis=0)
                cluster_centers_new.append(cluster_center_new)
            cluster_centers_new = np.array(cluster_centers_new)

            # Compute average deviation of cluster centers
            d_deviation_avg = 0
            for k in range(self.k):
                d_deviation_avg += np.linalg.norm(cluster_centers_new[k] - self.cluster_centers[k])
            d_deviation_avg /= self.k

            # Set new cluster center
            self.cluster_centers = cluster_centers_new

            # Announce convergence if d_deviation_avg is small enough
            threshold = 1E-10
            if d_deviation_avg < threshold:
                converge = True

    def predict(self, X):
        # Assign class labels to data based on closest distance to cluster centers
        N, d = X.shape
        y_pred = -1 * np.zeros(len(X))
        for i in range(N):
            distances = [np.linalg.norm(X[i] - self.cluster_centers[k]) for k in range(self.k)]
            c = np.argmin(distances)
            y_pred[i] = c
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
    model = KMeans(k=4)
    model.fit(X)
    y_pred = model.predict(X)
    print("y_pred", y_pred)
