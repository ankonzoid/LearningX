"""

 ModelTree.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np
from copy import deepcopy
from graphviz import Digraph

class ModelTree(object):

    def __init__(self, model, max_depth=5, min_samples_leaf=10,
                 search_type="greedy", n_search_grid=100):

        self.model = model
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.search_type = search_type
        self.n_search_grid = n_search_grid
        self.tree = None

    def get_params(self, deep=True):
        return {
            "model": self.model.get_params() if deep else self.model,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "search_type": self.search_type,
            "n_search_grid": self.n_search_grid,
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def __repr__(self):
        class_name = self.__class__.__name__
        return "{}({})".format(class_name, ', '.join([ "{}={}".format(k,v) for k, v in self.get_params(deep=False).items() ]))

    # ======================
    # Fit
    # ======================
    def fit(self, X, y, verbose=False):

        # Settings
        model = self.model
        min_samples_leaf = self.min_samples_leaf
        max_depth = self.max_depth
        search_type = self.search_type
        n_search_grid = self.n_search_grid

        if verbose:
            print(" max_depth={}, min_samples_leaf={}, search_type={}...".format(max_depth, min_samples_leaf, search_type))

        def _build_tree(X, y):

            global index_node_global

            def _create_node(X, y, depth, container):
                loss_node, model_node = _fit_model(X, y, model)
                node = {"name": "node",
                        "index": container["index_node_global"],
                        "loss": loss_node,
                        "model": model_node,
                        "data": (X, y),
                        "n_samples": len(X),
                        "j_feature": None,
                        "threshold": None,
                        "children": {"left": None, "right": None},
                        "depth": depth}
                container["index_node_global"] += 1
                return node

            # Recursively split node + traverse node until a terminal node is reached
            def _split_traverse_node(node, container):

                # Perform split and collect result
                result = _splitter(node, model,
                                   max_depth=max_depth,
                                   min_samples_leaf=min_samples_leaf,
                                   search_type=search_type,
                                   n_search_grid=n_search_grid)

                # Return terminal node if split is not advised
                if not result["did_split"]:
                    if verbose:
                        depth_spacing_str = " ".join([" "] * node["depth"])
                        print(" {}*leaf {} @ depth {}: loss={:.6f}, N={}".format(depth_spacing_str, node["index"], node["depth"], node["loss"], result["N"]))
                    return

                # Update node information based on splitting result
                node["j_feature"] = result["j_feature"]
                node["threshold"] = result["threshold"]
                del node["data"]  # delete node stored data

                # Extract splitting results
                (X_left, y_left), (X_right, y_right) = result["data"]
                model_left, model_right = result["models"]

                # Report created node to user
                if verbose:
                    depth_spacing_str = " ".join([" "] * node["depth"])
                    print(" {}node {} @ depth {}: loss={:.6f}, j_feature={}, threshold={:.6f}, N=({},{})".format(depth_spacing_str, node["index"], node["depth"], node["loss"], node["j_feature"], node["threshold"], len(X_left), len(X_right)))

                # Create children nodes
                node["children"]["left"] = _create_node(X_left, y_left, node["depth"]+1, container)
                node["children"]["right"] = _create_node(X_right, y_right, node["depth"]+1, container)
                node["children"]["left"]["model"] = model_left
                node["children"]["right"]["model"] = model_right

                # Split nodes
                _split_traverse_node(node["children"]["left"], container)
                _split_traverse_node(node["children"]["right"], container)

            container = {"index_node_global": 0}  # mutatable container
            root = _create_node(X, y, 0, container)  # depth 0 root node
            _split_traverse_node(root, container)  # split and traverse root node

            return root

        # Construct tree
        self.tree = _build_tree(X, y)

    # ======================
    # Predict
    # ======================
    def predict(self, X):
        assert self.tree is not None
        def _predict(node, x):
            no_children = node["children"]["left"] is None and \
                          node["children"]["right"] is None
            if no_children:
                y_pred_x = node["model"].predict([x])[0]
                return y_pred_x
            else:
                if x[node["j_feature"]] <= node["threshold"]:  # x[j] < threshold
                    return _predict(node["children"]["left"], x)
                else:  # x[j] > threshold
                    return _predict(node["children"]["right"], x)
        y_pred = np.array([_predict(self.tree, x) for x in X])
        return y_pred

    # ======================
    # Explain
    # ======================
    def explain(self, X, header):
        assert self.tree is not None
        def _explain(node, x, explanation):
            no_children = node["children"]["left"] is None and \
                          node["children"]["right"] is None
            if no_children:


                return explanation
            else:
                if x[node["j_feature"]] <= node["threshold"]:  # x[j] < threshold
                    explanation.append("{} = {:.6f} <= {:.6f}".format(header[node["j_feature"]], x[node["j_feature"]], node["threshold"]))
                    return _explain(node["children"]["left"], x, explanation)
                else:  # x[j] > threshold
                    explanation.append("{} = {:.6f} > {:.6f}".format(header[node["j_feature"]], x[node["j_feature"]], node["threshold"]))
                    return _explain(node["children"]["right"], x, explanation)

        explanations = [_explain(self.tree, x, []) for x in X]
        return explanations

    # ======================
    # Loss
    # ======================
    def loss(self, X, y, y_pred):
        loss = self.model.loss(X, y, y_pred)
        return loss

    # ======================
    # Tree diagram
    # ======================
    def export_graphviz(self, output_filename, feature_names,
                        export_png=True, export_pdf=False):
        """
         Assumes node structure of:

           node["name"]
           node["n_samples"]
           node["children"]["left"]
           node["children"]["right"]
           node["j_feature"]
           node["threshold"]
           node["loss"]

        """
        g = Digraph('g', node_attr={'shape': 'record', 'height': '.1'})

        def build_graphviz_recurse(node, parent_node_index=0, parent_depth=0, edge_label=""):

            # Empty node
            if node is None:
                return

            # Create node
            node_index = node["index"]
            if node["children"]["left"] is None and node["children"]["right"] is None:

                params = node['model'].get_params()
                if params is not None :
                    threshold_str="y = "
                    for i in range(len(params)):
                        threshold_str += str(round(params[i],2]))
                        threshold_str += "*X"+str(i)
                    threshold_str+="\n"
                else:
                    threshold_str=""

            else:
                threshold_str = "{} <= {:.1f}\\n".format(feature_names[node['j_feature']], node["threshold"])

            label_str = "{} n_samples = {}\\n loss = {:.6f}".format(threshold_str, node["n_samples"], node["loss"])

            # Create node
            nodeshape = "rectangle"
            bordercolor = "black"
            fillcolor = "white"
            fontcolor = "black"
            g.attr('node', label=label_str, shape=nodeshape)
            g.node('node{}'.format(node_index),
                   color=bordercolor, style="filled",
                   fillcolor=fillcolor, fontcolor=fontcolor)

            # Create edge
            if parent_depth > 0:
                g.edge('node{}'.format(parent_node_index),
                       'node{}'.format(node_index), label=edge_label)

            # Traverse child or append leaf value
            build_graphviz_recurse(node["children"]["left"],
                                   parent_node_index=node_index,
                                   parent_depth=parent_depth + 1,
                                   edge_label="")
            build_graphviz_recurse(node["children"]["right"],
                                   parent_node_index=node_index,
                                   parent_depth=parent_depth + 1,
                                   edge_label="")

        # Build graph
        build_graphviz_recurse(self.tree,
                               parent_node_index=0,
                               parent_depth=0,
                               edge_label="")

        # Export pdf
        if export_pdf:
            print("Saving model tree diagram to '{}.pdf'...".format(output_filename))
            g.format = "pdf"
            g.render(filename=output_filename, view=False, cleanup=True)

        # Export png
        if export_png:
            print("Saving model tree diagram to '{}.png'...".format(output_filename))
            g.format = "png"
            g.render(filename=output_filename, view=False, cleanup=True)


# ***********************************
#
# Side functions
#
# ***********************************


def _splitter(node, model,
              max_depth=5, min_samples_leaf=10,
              search_type="greedy", n_search_grid=100):

    # Extract data
    X, y = node["data"]
    depth = node["depth"]
    N, d = X.shape

    # Find feature splits that might improve loss
    did_split = False
    loss_best = node["loss"]
    data_best = None
    models_best = None
    j_feature_best = None
    threshold_best = None

    # Perform threshold split search only if node has not hit max depth
    if (depth >= 0) and (depth < max_depth):

        for j_feature in range(d):

            # If using adaptive search type, decide on one to use
            search_type_use = search_type
            if search_type == "adaptive":
                if N > n_search_grid:
                    search_type_use = "grid"
                else:
                    search_type_use = "greedy"

            # Use decided search type and generate threshold search list (j_feature)
            threshold_search = []
            if search_type_use == "greedy":
                for i in range(N):
                    threshold_search.append(X[i, j_feature])
            elif search_type_use == "grid":
                x_min, x_max = np.min(X[:, j_feature]), np.max(X[:, j_feature])
                dx = (x_max - x_min) / n_search_grid
                for i in range(n_search_grid+1):
                    threshold_search.append(x_min + i*dx)
            else:
                exit("err: invalid search_type = {} given!".format(search_type))

            # Perform threshold split search on j_feature
            for threshold in threshold_search:

                # Split data based on threshold
                (X_left, y_left), (X_right, y_right) = _split_data(j_feature, threshold, X, y)
                N_left, N_right = len(X_left), len(X_right)

                # Splitting conditions
                split_conditions = [N_left >= min_samples_leaf,
                                    N_right >= min_samples_leaf]

                # Do not attempt to split if split conditions not satisfied
                if not all(split_conditions):
                    continue

                # Compute weight loss function
                loss_left, model_left = _fit_model(X_left, y_left, model)
                loss_right, model_right = _fit_model(X_right, y_right, model)
                loss_split = (N_left*loss_left + N_right*loss_right) / N

                # Update best parameters if loss is lower
                if loss_split < loss_best:
                    did_split = True
                    loss_best = loss_split
                    models_best = [model_left, model_right]
                    data_best = [(X_left, y_left), (X_right, y_right)]
                    j_feature_best = j_feature
                    threshold_best = threshold

    # Return the best result
    result = {"did_split": did_split,
              "loss": loss_best,
              "models": models_best,
              "data": data_best,
              "j_feature": j_feature_best,
              "threshold": threshold_best,
              "N": N}

    return result

def _fit_model(X, y, model):
    model_copy = deepcopy(model)  # must deepcopy the model!
    model_copy.fit(X, y)
    y_pred = model_copy.predict(X)
    loss = model_copy.loss(X, y, y_pred)
    assert loss >= 0.0
    return loss, model_copy

def _split_data(j_feature, threshold, X, y):
    idx_left = np.where(X[:, j_feature] <= threshold)[0]
    idx_right = np.delete(np.arange(0, len(X)), idx_left)
    assert len(idx_left) + len(idx_right) == len(X)
    return (X[idx_left], y[idx_left]), (X[idx_right], y[idx_right])
