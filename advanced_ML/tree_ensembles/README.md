# Tree Ensembles: Bagging and Boosting

When it comes to improving the capabilities of a single decision tree model by training more trees, there are typically tree ensemble approaches: Bagging (Bootstrap Aggregating) and Boosting (Gradient Boosting). Under the bias-variance tradeoff, bagging and boosting work in opposite directions with bagging helping with reducing overfitting, and boosting helping with adding complexity to underfit models. We provide here from-scratch vanilla tree bagging and boosting ensemble implementations as a way to learn and familiarize with the techniques.

Bagging constructs multiple slightly overfit tree estimators in parallel by training each tree on a `(X_i, y_i)` dataset built by sampling the original `(X, y)` dataset `N` times *with replacement* where `N` are the number of examples in `(X, y)`. The bagging prediction is then the prediction average of the estimators in the tree ensemble. Refer to the bagging schematic below.

<p align="center"> 
<img src="https://github.com/ankonzoid/ML_algorithms/blob/master/tree_ensembles/coverart/bagging.png" width="90%">
</p>

We also provide here another form of bagging which is Random Forest where the trees have random splits (as opposed to greedy splits) and on limited features (as opposed to all features). The heuristic is still towards the idea of exposing the individual trees to less data to give them new ways to overfit for the betterment of the ensemble as a whole.

Gradient boosting takes on a sequential approach to constructing its tree estimators. First, gradient boosting trains the the first tree estimator on `(X, y)` (same as training a single decision tree). Afterwards the training residual (`res_i = y_truth - y_pred_i`) becomes the training label for the next tree estimator (`y <- res_i`, `X` untouched), and this is done recursively until the residuals are small enough. The boosting prediction is then the prediction sum of the estimator predictions in the tree ensemble. Refer to the boosting schematic below.

<p align="center"> 
<img src="https://github.com/ankonzoid/ML_algorithms/blob/master/tree_ensembles/coverart/boosting.png" width="90%">
</p>

### Usage

Run the command

> python3 tree_ensembles.py

### Example Output

```
Plotting 'Bagged Trees' model to 'output/bagged_trees'...
 -> n_estimators = 1, max_depth = 2 -> err = 28.267222
 -> n_estimators = 25, max_depth = 2 -> err = 26.193547
 -> n_estimators = 50, max_depth = 2 -> err = 24.991074
 -> n_estimators = 1, max_depth = 4 -> err = 15.404056
 -> n_estimators = 25, max_depth = 4 -> err = 10.269663
 -> n_estimators = 50, max_depth = 4 -> err = 10.182270
 -> n_estimators = 1, max_depth = 6 -> err = 8.597073
 -> n_estimators = 25, max_depth = 6 -> err = 3.889720
 -> n_estimators = 50, max_depth = 6 -> err = 3.948179
Plotting 'Random Forest' model to 'output/random_forest'...
 -> n_estimators = 1, max_depth = 2 -> err = 41.366102
 -> n_estimators = 25, max_depth = 2 -> err = 36.330356
 -> n_estimators = 50, max_depth = 2 -> err = 36.951298
 -> n_estimators = 1, max_depth = 4 -> err = 20.296716
 -> n_estimators = 25, max_depth = 4 -> err = 15.419515
 -> n_estimators = 50, max_depth = 4 -> err = 14.500037
 -> n_estimators = 1, max_depth = 6 -> err = 5.900154
 -> n_estimators = 25, max_depth = 6 -> err = 3.624212
 -> n_estimators = 50, max_depth = 6 -> err = 5.031134
Plotting 'Gradient Boosted Trees' model to 'output/gradient_boosted_trees'...
 -> n_estimators = 1, max_depth = 1 -> err = 39.794614
 -> n_estimators = 10, max_depth = 1 -> err = 15.043073
 -> n_estimators = 20, max_depth = 1 -> err = 10.857658
 -> n_estimators = 1, max_depth = 2 -> err = 32.042706
 -> n_estimators = 10, max_depth = 2 -> err = 7.148548
 -> n_estimators = 20, max_depth = 2 -> err = 3.747128
 -> n_estimators = 1, max_depth = 3 -> err = 19.074080
 -> n_estimators = 10, max_depth = 3 -> err = 2.383883
 -> n_estimators = 20, max_depth = 3 -> err = 0.603004
```

In the `output` directory, you will find the following tree ensemble prediction plots (in `.png` format):

<p align="center"> 
<img src="https://github.com/ankonzoid/ML_algorithms/blob/master/tree_ensembles/output/bagged_trees.png" width="75%">
</p>

<p align="center"> 
<img src="https://github.com/ankonzoid/ML_algorithms/blob/master/tree_ensembles/output/random_forest.png" width="75%">
</p>

<p align="center"> 
<img src="https://github.com/ankonzoid/ML_algorithms/blob/master/tree_ensembles/output/gradient_boosted_trees.png" width="75%">
</p>


### Libraries

* numpy, sklearn, matplotlib

### Authors

Anson Wong