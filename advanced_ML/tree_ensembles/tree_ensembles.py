"""

 tree_ensembles.py  (author: Anson Wong / git: ankonzoid)

 We provide from-scratch vanilla tree ensemble regressor implementations:

  1) Bagged Trees  (src/BaggedTrees.py)
  2) Random Forest  (src/RandomForest.py)
  3) Gradient Boosted Trees  (src/GradientBoostedTree.py)

"""
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Generate 1D polynomial data for trees to fit on
    def generate_data(func, x_range=(0, 10), N=100):
        X = np.linspace(x_range[0], x_range[1], N)
        y = np.vectorize(func)(X)
        X = X[:, None]
        return X, y
    func = lambda x: (x-1)*(x-4)*(x-8)*(x-8)
    X, y = generate_data(func, x_range=(0, 10), N=100)

    # Tree ensemble models
    from src.BaggedTrees import BaggedTreeRegressor
    from src.RandomForest import RandomForestTreeRegressor
    from src.GradientBoostedTree import GradientBoostedTreeRegressor

    # Train and plot models (pdf + png)
    def plot_model(tree_ensemble_model, X, y, plot_filename,
                   n_estimators_vec=[0, 10, 20], max_depth_vec=[1, 4, 8], name=""):
        from sklearn.metrics import mean_absolute_error
        print("Plotting '{}' model to '{}'...".format(name, plot_filename))
        Ni, Nj = len(max_depth_vec), len(n_estimators_vec)
        plt.figure(figsize=(16, 10))
        for i in range(len(max_depth_vec)):
            for j in range(len(n_estimators_vec)):
                # Train model
                model = tree_ensemble_model(n_estimators=n_estimators_vec[j], max_depth=max_depth_vec[i])
                model.fit(X, y)
                y_pred = model.predict(X)
                err = mean_absolute_error(y, y_pred)
                print(" -> n_estimators = {}, max_depth = {} -> err = {:.6f}".format(
                    n_estimators_vec[j], max_depth_vec[i], err))
                # Plot
                plt.subplot(int("{}{}{}".format(Ni, Nj, i*Ni + j + 1)))
                X_flatten = X.reshape((-1,))
                plt.plot(X_flatten, y, 'ok', markersize=5, label='data')
                plt.plot(X_flatten, y_pred, 'or', markersize=3, linewidth=3, label='pred')
                plt.title('(n_estimators, depth) = ({}, {})'.format(
                    n_estimators_vec[j], max_depth_vec[i]))
                plt.tick_params(
                    axis='both', which='both',
                    bottom=False, left=False, top=False,
                    labelbottom=False, labelleft=False)
        plt.suptitle("{} predictions".format(name), fontsize=30)
        plt.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0.)
        plt.savefig(plot_filename + ".png", bbox_inches=None)  # supports ".pdf" too
        plt.close()

    plot_model(BaggedTreeRegressor, X, y, "output/bagged_trees",
               n_estimators_vec=[1, 25, 50], max_depth_vec=[2, 4, 6], name="Bagged Trees")
    plot_model(RandomForestTreeRegressor, X, y, "output/random_forest",
               n_estimators_vec=[1, 25, 50], max_depth_vec=[2, 4, 6], name="Random Forest")
    plot_model(GradientBoostedTreeRegressor, X, y, "output/gradient_boosted_trees",
               n_estimators_vec=[1, 10, 20], max_depth_vec=[1, 2, 3], name="Gradient Boosted Trees")

# Driver
if __name__ == "__main__":
    main()