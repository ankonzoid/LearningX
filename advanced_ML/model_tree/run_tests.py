"""

 run_tests.py  (author: Anson Wong / git: ankonzoid)

 Runs 3 tests to make sure our model tree works as expected.

"""
import os, csv
import numpy as np
import matplotlib.pyplot as plt
from src.utils import load_csv_data
from src.ModelTree import ModelTree

def main():
    # ====================
    # Run sanity checks on model tree before training (using our own data)
    #  1) Reproduce model result on depth-0 model tree
    #  2) Reproduce sklearn DecisionTreeRegressor result using mean regression + mse
    #  3) Reproduce sklearn DecisionTreeClassifier result using modal class + gini loss
    # ====================
    run_tests(ModelTree, os.path.join("data", "data_clf.csv"))

    # ====================
    # For 1D polynomial data using a model tree with linear regression model
    # ====================

    # Generate 1D polynomial data and save as a csv
    func = lambda x: (x-1)*(x-4)*(x-8)*(x-8)
    data_csv_data_filename = os.path.join("data", "data_poly4_regr.csv")
    generate_csv_data(func, data_csv_data_filename,  x_range=(0, 10), N=500)

    # Read generated data
    X, y, header = load_csv_data(data_csv_data_filename, mode="regr", verbose=True)
    assert X.shape[1] == 1

    # Train different depth model tree fits and plot results
    from models.mean_regr import mean_regr
    plot_model_tree_fit(mean_regr(), X, y)
    from models.linear_regr import linear_regr
    plot_model_tree_fit(linear_regr(), X, y)


# ********************************
#
# Side functions
#
# ********************************


def plot_model_tree_fit(model, X, y):
        output_filename = os.path.join("output", "test_{}_fit.png".format(model.__class__.__name__))
        print("Saving model tree predictions plot y vs x to '{}'...".format(output_filename))

        plt.figure(figsize=(20, 10))
        figure_str = "23"
        for depth in range(6):
            # Form model tree
            print(" -> training model tree depth={}...".format(depth))
            model_tree = ModelTree(model, max_depth=depth, min_samples_leaf=10,
                                   search_type="greedy", n_search_grid=100)

            # Train model tree
            model_tree.fit(X, y, verbose=False)
            y_pred = model_tree.predict(X)

            # Plot predictions
            plt.subplot(int(figure_str + str(depth + 1)))
            plt.plot(X[:, 0], y, '.', markersize=5, color='k')
            plt.plot(X[:, 0], y_pred, '.', markersize=5, color='r')
            plt.legend(['data', 'fit'])
            plt.title("depth = {}".format(depth))
            plt.xlabel("x", fontsize=15)
            plt.ylabel("y", fontsize=15)
            plt.grid()

        plt.suptitle('Model tree (model = {}) fits for different depths'.format(model.__class__.__name__), fontsize=25)
        plt.savefig(output_filename, bbox_inches='tight')
        plt.close()

def generate_csv_data(func, output_csv_filename, x_range=(0, 1), N=500):
    x_vec = np.linspace(x_range[0], x_range[1], N)
    y_vec = np.vectorize(func)(x_vec)
    with open(output_csv_filename, "w") as f:
        writer = csv.writer(f)
        field_names = ["x1", "y"]
        writer.writerow(field_names)
        for (x, y) in zip(x_vec, y_vec):
            field_values = [x, y]
            writer.writerow(field_values)

def run_tests(ModelTree, data_csv_filename):

    print("Running model tree tests...")
    eps = 1E-6 # tolerance for test acceptance
    X, y, header = load_csv_data(data_csv_filename, mode="regr")

    # Test 1
    print(" [1/3] Checking depth-0 model tree...")
    from models.linear_regr import linear_regr
    model = linear_regr()
    MTR_0 = ModelTree(model, max_depth=0, min_samples_leaf=20,
                      search_type="greedy", n_search_grid=100)
    loss_model = experiment(model, X, y)
    loss_MTR_0 = experiment(MTR_0, X, y)
    print("  -> loss(linregr)={:.6f}, loss(MTR_0_linregr)={:.6f}...".format(loss_model, loss_MTR_0))
    if np.abs(loss_model - loss_MTR_0) > eps:
        exit("err: passed test 1!")
    else:
        print("  -> passed test 1!")

    # Test 2
    print(" [2/3] Reproducing DecisionTreeRegressor sklearn (depth=20) result...")
    from models.mean_regr import mean_regr
    MTR = ModelTree(mean_regr(), max_depth=20, min_samples_leaf=10,
                    search_type="greedy", n_search_grid=100)
    from models.DT_sklearn_regr import DT_sklearn_regr
    DTR_sklearn = DT_sklearn_regr(max_depth=20, min_samples_leaf=10)
    loss_MTR = experiment(MTR, X, y)
    loss_DTR_sklearn = experiment(DTR_sklearn, X, y)
    print("  -> loss(MTR)={:.6f}, loss(DTR_sklearn)={:.6f}...".format(loss_MTR, loss_DTR_sklearn))
    if np.abs(loss_MTR - loss_DTR_sklearn) > eps:
        exit("err: passed test 2!")
    else:
        print("  -> passed test 2!")

    # Test 3
    print(" [3/3] Reproducing DecisionTreeClassifier sklearn (depth=20) result...")
    from models.modal_clf import modal_clf
    MTC = ModelTree(modal_clf(), max_depth=20, min_samples_leaf=10,
                    search_type="greedy", n_search_grid=100)
    from models.DT_sklearn_clf import DT_sklearn_clf
    DTC_sklearn = DT_sklearn_clf(max_depth=20, min_samples_leaf=10)
    loss_MTC = experiment(MTC, X, y)
    loss_DTC_sklearn = experiment(DTC_sklearn, X, y)
    print("  -> loss(MTC)={:.6f}, loss(DTC_sklearn)={:.6f}...".format(loss_MTC, loss_DTC_sklearn))
    if np.abs(loss_MTC - loss_DTC_sklearn) > eps:
        exit("err: passed test 3!")
    else:
        print("  -> passed test 3!")
    print()

def experiment(model, X, y):
    model.fit(X, y)  # train model
    y_pred = model.predict(X)
    loss = model.loss(X, y, y_pred)  # compute loss
    return loss

# Driver
if __name__ == "__main__":
    main()