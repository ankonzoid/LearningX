# Building Model Trees

In CART (Classification and Regression Tree) algorithms, we build a tree by recursively splitting the training data via feature threshold cuts, such that the split data achieves the lowest overall weighted loss possible. By assigning a regression/classification model with a set loss function, we are appending a model to each node of the tree which motivates the concept of model trees which we provide the implementation for here.

<p align="center"> 
<img src="https://github.com/ankonzoid/ML_algorithms/blob/master/model_tree/coverart/model_tree.png" width="65%">
</p>

To quickly visualize how a model tree could prove more useful than regular CARTs, consider the generated 1D training data below where we naively try to fit a linear regression model to it (this is exactly the `depth=0` model tree). The fit is poor as expected as the training data is generated from a 4th-order polynomial, but if you consider splitting the data into disjoint `x` segments as done by increasing the depth of the linear regression model tree, we can build a collection of linear regressors which accurately fit the individual segments well (`depth=1, 2, 3, 4, 5`), and thus giving us a well-trained model without needing too much explicit knowledge of the underlying complexity of the training data distribution!

<p align="center"> 
<img src="https://github.com/ankonzoid/ML_algorithms/blob/master/model_tree/output/test_linear_regr_fit.png" width="95%">
</p>

To hit the nail on the head with this example, we can also directly compare results of fitting the 1D training data using a linear regression model tree fitting with *scikit learn*'s default decision tree regressor (which uses mean-value regression). In the plot below, we can immediately see how *scikit learn*'s decision tree regressor at `depth=5` is still is not a great model because it struggles to capture the `x` variability in the data, whereas the model tree is already able to capture much of the training distribution at depths lower than 5.

<p align="center"> 
<img src="https://github.com/ankonzoid/ML_algorithms/blob/master/model_tree/output/test_mean_regr_fit.png" width="95%">
</p>

### Usage

Below are the model tree parameters to set before running the code:

* `model`: `mean_regr`, `linear_regr`, `logistic_regr`, `svm_regr`, `modal_clf` (you can create your own as well)
* `max_depth`: 1, 2, 3, 4, ...
* `min_samples_leaf`: 1, 2, 3, 4, ...
* `search_type`: "greedy", "grid", "adaptive"
* `n_search_grid`: number uniform grid search points (activated only if `search_type = grid` or `search_type = adaptive`) 
* `verbose`: True, False

When your parameters are set, run the command

> python3 run_model_tree.py

which:

1) Loads the data

2) Trains the model tree

3) Outputs: a tree-diagram schematic, trained model tree, model tree training predictions

4) Cross validates the model

The `stdout` text of the run should look like:

```
Loading data from 'input/data_clf.csv' (mode=regr)...
 header=['x1', 'x2', 'x3', 'x4', 'y']
 X.shape=(1372, 4)
 y.shape=(1372,)
 len(y_classes)=2

Training model tree with 'linear_regr'...
 max_depth=4, min_samples_leaf=10, search_type=greedy...
 node 0 @ depth 0: loss=0.033372, j_feature=1, threshold=-1.862400, N=(339,1033)
  node 1 @ depth 1: loss=0.016889, j_feature=0, threshold=0.234600, N=(209,130)
    node 3 @ depth 2: loss=0.006617, j_feature=0, threshold=-0.657670, N=(195,14)
      *leaf 5 @ depth 3: loss=0.000000, N=195
      *leaf 6 @ depth 3: loss=0.004635, N=14
    node 4 @ depth 2: loss=0.010361, j_feature=0, threshold=0.744280, N=(13,117)
      *leaf 7 @ depth 3: loss=0.000000, N=13
      *leaf 8 @ depth 3: loss=0.000000, N=117
  node 2 @ depth 1: loss=0.023927, j_feature=2, threshold=-1.544300, N=(346,687)
    node 9 @ depth 2: loss=0.014254, j_feature=1, threshold=5.202200, N=(149,197)
      node 11 @ depth 3: loss=0.007080, j_feature=0, threshold=2.017700, N=(139,10)
        *leaf 13 @ depth 4: loss=0.000000, N=139
        *leaf 14 @ depth 4: loss=0.003821, N=10
      *leaf 12 @ depth 3: loss=0.000000, N=197
    node 10 @ depth 2: loss=0.018931, j_feature=0, threshold=0.559390, N=(377,310)
      node 15 @ depth 3: loss=0.020929, j_feature=3, threshold=-1.566800, N=(154,223)
        *leaf 17 @ depth 4: loss=0.010759, N=154
        *leaf 18 @ depth 4: loss=0.020452, N=223
      node 16 @ depth 3: loss=0.002916, j_feature=1, threshold=-0.045533, N=(23,287)
        *leaf 19 @ depth 4: loss=0.016037, N=23
        *leaf 20 @ depth 4: loss=0.000000, N=287
 -> loss_train=0.004876

Saving model tree diagram to 'output/model_tree.png'...
Saving model tree to 'output/model_tree.p'...
Saving mode tree predictions to 'output/model_tree_pred.csv'
Cross-validating (kfold=5, seed=1)...
 [fold 1/5] loss_train=0.00424305, loss_validation=0.011334
 [fold 2/5] loss_train=0.00373604, loss_validation=0.0138225
 [fold 3/5] loss_train=0.00249428, loss_validation=0.00959152
 [fold 4/5] loss_train=0.00207239, loss_validation=0.0103934
 [fold 5/5] loss_train=0.00469358, loss_validation=0.010235
  -> loss_train_avg=0.003448, loss_validation_avg=0.011075
```

In the `output` directory, you will find the pickled model tree (`output/model_tree.p`), a visualization of the model tree created (`output/model_tree.png`)

<p align="center"> 
<img src="https://github.com/ankonzoid/ML_algorithms/blob/master/model_tree/output/model_tree.png" width="95%">
</p>

as well as a `model_tree_pred.csv` file containing the model tree's predictions and tree-traversal explanations

```
x,y,y_pred,explanation
[ 3.6216   8.6661  -2.8073  -0.44699],0,0.0,"['x2 = 8.666100 > -1.862400', 'x3 = -2.807300 <= -1.544300', 'x2 = 8.666100 > 5.202200']"
[ 4.5459  8.1674 -2.4586 -1.4621],0,0.0,"['x2 = 8.167400 > -1.862400', 'x3 = -2.458600 <= -1.544300', 'x2 = 8.167400 > 5.202200']"
[ 3.866   -2.6383   1.9242   0.10645],0,0.0,"['x2 = -2.638300 <= -1.862400', 'x1 = 3.866000 > 0.234600', 'x1 = 3.866000 > 0.744280']"
...
```

### Test usage

To ensure our model tree implementation is working, you can also run our test function

> python3 run_tests.py

which will:

1) Reproduce the original model result at model tree depth-zero

2) Reproduce sklearn's default Decision Tree Classifier implementation (`DecisionTreeClassifier`) using modal classification with Gini class impurity loss
 
3) Reproduce sklearn's default Decision Tree Regressor implementation(`DecisionTreeRegressor`) using mean regression with mean squared error loss

4) Generate plots of model tree predictions of a 4th-order polynomial at different tree depths using linear regression model trees (`output/test_linear_regr_fit.png`) and mean regression model trees (`output/test_mean_regr_fit.png`)

### Libraries

* numpy, pandas, pickle, sklearn, scipy, graphviz

### Authors

Anson Wong