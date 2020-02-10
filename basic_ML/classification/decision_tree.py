"""

 decision_tree.py  (author: Anson Wong / git: ankonzoid)

"""
import os
import numpy as np
import pandas as pd

class DecisionTree():

    def __init__(self):
        pass

    def fit(self, X, y):

        def loss_func(X, y):
            # Gini impurity
            p2 = 0.0
            y_classes = list(set(y))
            for c in y_classes:
                p2 += (np.sum(y == c) / len(y)) ** 2
            loss = 1.0 - p2
            return loss

        def perform_split(X, y):
            N, d = X.shape
            loss_best = 9E99
            X_left, y_left = [], []
            X_right, y_right = [], []
            threshold_best = None
            j_threshold_best = None
            for j in range(d):
                for i in range(N):

                    # Set threshold
                    threshold = X[i, j]

                    # Split data by threshold
                    X_left_split, y_left_split = [], []
                    X_right_split, y_right_split = [], []
                    for X_i, y_i in zip(X, y):
                        if X_i[j] <= threshold:
                            X_left_split.append(X_i)
                            y_left_split.append(y_i)
                        else:
                            X_right_split.append(X_i)
                            y_right_split.append(y_i)
                    X_left_split = np.array(X_left_split)
                    y_left_split = np.array(y_left_split)
                    X_right_split = np.array(X_right_split)
                    y_right_split = np.array(y_right_split)

                    # Compute weighted loss
                    N_left = len(X_left_split)
                    N_right = len(X_right_split)
                    loss_left = loss_func(X_left_split, y_left_split)
                    loss_right = loss_func(X_right_split, y_right_split)
                    loss = (N_left/N)*loss_left + (N_right/N)*loss_right

                    # Is this the best split?
                    if loss < loss_best:
                        X_left, y_left = X_left_split, y_left_split
                        X_right, y_right = X_right_split, y_right_split
                        loss_best = loss
                        threshold_best = threshold
                        j_threshold_best = j

            # Split?
            split = len(X_left)>0 and len(X_right)>0

            return X_left, y_left, X_right, y_right, split, threshold_best, j_threshold_best

        def build_tree(X, y):
            X_left, y_left, X_right, y_right, split, threshold, j_threshold = perform_split(X, y)
            if split:
                child_left = build_tree(X_left, y_left)
                child_right = build_tree(X_right, y_right)
                node = {"split": split,
                        "threshold": threshold,
                        "j_threshold": j_threshold,
                        "child_left": child_left,
                        "child_right": child_right}
            else:
                node = {"value": max(set(list(y)), key=list(y).count)}
            return node

        self.tree = build_tree(X, y)

    def predict(self, X):

        def traverse(node, x):
            if "split" in node.keys():
                if x[node["j_threshold"]] <= node["threshold"]:
                    return traverse(node["child_left"], x)
                else:
                    return traverse(node["child_right"], x)
            else:
                return node["value"]

        N, d = X.shape
        y_pred = []
        for i in range(N):
            y_pred_i = traverse(self.tree, X[i])
            y_pred.append(y_pred_i)
        y_pred = np.array(y_pred)
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
    model = DecisionTree()
    model.fit(X, y)
    y_pred = model.predict(X)
    print("acc_train =", np.sum(y_pred==y)/len(y))
