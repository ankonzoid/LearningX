"""

 utils.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np
import pandas as pd
from copy import copy

def load_csv_data(input_csv_filename, mode="clf", verbose=False):
    if verbose:
        print("Loading data from '{}' (mode={})...".format(input_csv_filename, mode))
    df = pd.read_csv(input_csv_filename)  # dataframe
    df_header = df.columns.values  # header
    header = list(df_header)
    N, d = len(df), len(df_header) - 1
    X = np.array(df.drop(['y'], axis=1))  # extract X by dropping y column
    y = np.array(df['y'])  # extract y
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

def cross_validate(model, X, y, kfold=5, seed=1):

    def make_crossval_folds(N, kfold, seed=1):
        np.random.seed(seed)
        idx_all_permute = np.random.permutation(N)
        N_fold = int(N / kfold)
        idx_folds = []
        for i in range(kfold):
            start = i * N_fold
            end = min([(i + 1) * N_fold, N])
            idx_folds.append(idx_all_permute[start:end])
        return idx_folds

    N = len(X)
    idx_all = np.arange(0, N)
    idx_folds = make_crossval_folds(N, kfold, seed=seed)
    assert len(idx_folds) == kfold

    print("Cross-validating (kfold={}, seed={})...".format(kfold, seed))

    loss_train_avg, loss_val_avg = 0.0, 0.0
    for i in range(kfold):

        # Split data
        idx_fold = idx_folds[i]
        idx_rest = np.delete(idx_all, idx_fold)
        X_rest, y_rest = X[idx_rest], y[idx_rest]
        X_fold, y_fold = X[idx_fold], y[idx_fold]

        # Train
        model_rest = copy(model)
        model_rest.fit(X_rest, y_rest)

        # Evaluate
        y_pred_rest = model_rest.predict(X_rest)
        y_pred_fold = model_rest.predict(X_fold)

        # Compute losses
        loss_train = model.loss(X_rest, y_rest, y_pred_rest)
        loss_val = model.loss(X_fold, y_fold, y_pred_fold)

        loss_train_avg += loss_train
        loss_val_avg += loss_val

        print(" [fold {}/{}] loss_train={:.6}, loss_validation={:.6}".format(i+1, kfold, loss_train, loss_val))

    loss_train_avg /= kfold
    loss_val_avg /= kfold

    print("  -> loss_train_avg={:.6f}, loss_validation_avg={:.6f}\n".format(loss_train_avg, loss_val_avg))

    return loss_train_avg, loss_val_avg