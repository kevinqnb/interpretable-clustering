import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from typing import Callable, List, Dict, Tuple, Any
from numpy.typing import NDArray
from .node import Node
from .utils import flatten_labels
from .rules import Decision


####################################################################################################


def plot_decision_boundaries(
    model : Callable,
    X : NDArray,
    color_dict : Dict[int, Any],
    ax : Callable = None,
    resolution : int = 100,
    label_array = False
):
    """
    Plots the decision boundaries of a given model.

    Args:
        model (Callable): Prediction object which should have a predict() method.
        X (NDArray): Dataset fitted to the model. 
        color_dict (Dict[int, Any], optional): Dictionary mapping cluster labels to colors.
        ax (matplotlib axes, optional): Axes for plotting. 
        resolution (int, optional): Number of points on the meshgrid, controls the 
            resolution of the contour lines. Defaults to 100.
        label_array (bool, optional): `True` if the output of the model's prediction is a set of 
            labels represented as as 1d array. `False` if the labels should instead be a 2 label
            set. Defaults to False.
    """
    assert X.shape[1] == 2, "X must be a 2D array with shape (n_samples, 2)."

    # Define the axis boundaries of the plot
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    
    # Create a mesh grid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    # Predict the classification for each point in the mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    if not label_array:
        Z = flatten_labels(Z)
    Z = Z.reshape(xx.shape)
    unique_labs = np.unique(Z)

    for l in unique_labs:
        ix = np.where(Z == l)
        if ax is None:
            plt.contourf(xx, yy, Z, levels=[l-0.5, l+0.5], colors=[color_dict[l]], alpha=0.25)
        else:
            ax.contourf(xx, yy, Z, levels=[l-0.5, l+0.5], colors=[color_dict[l]], alpha=0.25)
        

####################################################################################################


def plot_rule_decision_boundaries(
    model : Callable,
    X : NDArray,
    color_dict : Dict[int, Any] = None,
    ax : Callable = None
):
    """
    Plots the decision boundaries of a given model as boxes around the rules.
    This is useful for visualizing the rules of a general decision set, which are often overlapping.

    Args:
        model (Callable): Prediction object which should have a predict() method.
        X (NDArray): Dataset fitted to the model. Must be 2D with shape (n_samples, 2).
        color_dict (Dict[int, Any], optional): Dictionary mapping cluster labels to colors.
        ax (matplotlib axes, optional): Axes for plotting. If None, uses the current axes.
    """
    assert X.shape[1] == 2, "X must be a 2D array with shape (n_samples, 2)."
    decision_set_labels = [{decision.label} for decision in model.decision_set]
    if color_dict is None:
        color_dict = {list(i)[0]: 'grey' for i in decision_set_labels}

    supported_plot = ['PEC', 'IDS']
    if model.__class__.__name__ not in supported_plot:
        raise ValueError(
            f"Plotting for {model.__class__.__name__} is not supported. "
            f"Supported models are: {supported_plot}"
        )
    
    for i,decision in enumerate(model.decision_set):
        x_bounds = [np.min(X[:,0]), np.max(X[:,0])]
        y_bounds = [np.min(X[:,1]), np.max(X[:,1])]

        for condition in decision.rule.conditions:
            if condition.features[0] == 0:
                if condition.direction == -1:
                    x_bounds[1] = min(x_bounds[1], condition.threshold)
                else:
                    x_bounds[0] = max(x_bounds[0], condition.threshold)
            elif condition.features[0] == 1:
                if condition.direction == -1:
                    y_bounds[1] = min(y_bounds[1], condition.threshold)
                else:
                    y_bounds[0] = max(y_bounds[0], condition.threshold)

        if ax is None:
            plt.gca().add_patch(
                plt.Rectangle(
                    (x_bounds[0], y_bounds[0]),
                    x_bounds[1] - x_bounds[0],
                    y_bounds[1] - y_bounds[0],
                    fill=True,
                    facecolor=color_dict[list(decision_set_labels[i])[0]],
                    alpha=0.25,
                    linestyle='solid',
                    edgecolor='black',
                    linewidth=2
                )
            )
        else:
            ax.add_patch(
                plt.Rectangle(
                    (x_bounds[0], y_bounds[0]),
                    x_bounds[1] - x_bounds[0],
                    y_bounds[1] - y_bounds[0],
                    fill=True, 
                    facecolor=color_dict[list(decision_set_labels[i])[0]],
                    alpha=0.25,
                    linestyle='solid',
                    edgecolor='black',
                    linewidth=2
                )
            )
        

####################################################################################################


def build_networkx_graph(graph : Callable, node : Node):
    """
    Constructs a networkx graph by adding edges from the current node object. 

    Args:
        graph (networkx Digraph): Networkx Graph object to add nodes to. Should be initially empty.

        node (Node): Node object.
    """
    if node.type != "leaf":
        graph.add_edge(node, node.left_child)
        build_networkx_graph(graph, node.left_child)

        graph.add_edge(node, node.right_child)
        build_networkx_graph(graph, node.right_child)


####################################################################################################


def plot_tree(
    root : Node,
    feature_labels : List[str] = None,
    leaf_labels : List[str] = None,
    data_scaler : Callable = None,
    color_dict : Dict[int, Any] = None,
    output_file : str = None,
):
    """
    Wrapper function for drawing a Tree object with networkx.

    Args:
        root (Node): Root Node object for the tree.
        
        feature_labels (List[str], optional): List of feature labels used for display.
            Each non-leaf Node object has a feature index attribute, and we use 
            feature_labels[index] to print the label associated with the index. Defaults to None
            which displays basic feature information. 

        data_scaler (Callable): Sklearn data scaler, which will be used to convert
            thresholds, conditions back to their unscaled versions (better interpretability).
            This current supports the StandardScaler or the MinMaxScaler. Defaults to None 
            in which case values are left as is. 
            
        cmap (Callable): Matplotlib colormap. Should be callable so that cmap(i) gives the 
            color for cluster i.
        
        output_file (str, optional): File to save the resulting image. Defaults to None.
    """
    G = nx.DiGraph()
    build_networkx_graph(G, root)
    node_colors = [
        color_dict[node.label] if (node.type == 'leaf' and color_dict is not None)
        else 'white' for node in G.nodes
    ]
    node_labels = {}
    for node in G.nodes:
        if node.type == 'internal':
            node_labels[node] = node.condition.display(
                scaler = data_scaler, 
                feature_labels = feature_labels
            )
        else:
            if leaf_labels is not None:
                node_labels[node] = leaf_labels[node.label]
            else:
                node_labels[node] = "Cluster " + str(node.label)

    node_sizes = [12500 if node.type == "internal" else 7500 for node in G.nodes]

    fig,ax = plt.subplots(figsize = (12,12))
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot")
    nx.draw_networkx(
        G,
        pos,
        labels=node_labels,
        node_color = node_colors,
        node_size = node_sizes,
        edge_color="black",
        edgecolors="black",
        font_size=18,
        linewidths = 2
    )
    plt.axis('off')
    if output_file is not None:
        plt.savefig(output_file, bbox_inches = 'tight', dpi = 300)


####################################################################################################


def plot_decision_set(
    decision_set : List[Decision],
    feature_labels : List[str] = None,
    data_scaler : Callable = None,
    color_dict : Dict[int, Any] = None,
    vertical : bool = True,
    size_factor : float = None,
    filename : str = None
):
    """
    Plots a decision set as a list of rules.
    
    Args:
        decision_set (List[List[Condition]]): A list of rules, where each rule is a 
            list of Condition objects.

        rule_labels (List[List[int]]): List of cluster labels for each rule.
        
        feature_labels (List[str]): List of feature names used for display. Defaults to None
            which displays basic feature information. 

        data_scaler (Callable): Sklearn data scaler, which will be used to convert
            thresholds, conditions back to their unscaled versions (better interpretability).
            This current supports the StandardScaler or the MinMaxScaler. Defaults to None 
            in which case values are left as is. 

        color_dict (Dict[int, Any]): Dictionary mapping cluster labels to colors.
            If None, does not use any colors for plotting.

        vertical (bool): If True, plots the rules vertically. If False, plots horizontally. 
            Defaults to True.

        size_factor (int): Factor to scale the size of the plot based on the length of the rules.
            Defaults to None, which automatically scales based on the maximum rule length.
        
        filename (str, optional): File to save the resulting image. Defaults to None
    """
    max_rule_length = np.max([len(d.rule) for d in decision_set])
    rule_label_array = np.array([d.label for d in decision_set])

    if size_factor is None:
        size_factor = max(1, max_rule_length // 2)

    if vertical:
        fig,ax = plt.subplots(figsize = (4, len(decision_set) * size_factor), dpi = 300)
        ax.set_xlim(0, 5)
        ax.set_ylim(0.0, (len(decision_set) + 0.2) * size_factor)
    else:
        fig,ax = plt.subplots(figsize = (len(decision_set) * size_factor, 4), dpi = 300)
        ax.set_xlim(-2, (len(decision_set) + 0.1) * size_factor)
        ax.set_ylim(0, 0.6)

    ax.axis('off')
    #ax.set_aspect('equal')
    
    # Order rules by cluster labels
    ordering = np.argsort(rule_label_array)
    for i, idx in enumerate(ordering):
        rule = decision_set[idx].rule
        rule_string = 'If '
        
        # Every condition except the last node, which should be a leaf
        for j, condition in enumerate(rule.conditions):            
            rule_string += '('
            rule_string += condition.display(
                scaler=data_scaler,
                feature_labels=feature_labels,
                newline=False
            )
            rule_string += ')'
            
            if j >= len(rule) - 1:
                rule_string += f'\n'
            #elif j % 1 == 0: 
            elif j % 2 == 1:
                #rule_string += r" $\&$ " + f'\n'
                rule_string += r" and " + f'\n'
            else:
                #rule_string += r" $\&$ "
                rule_string += r" and "
                
        rule_string += 'Then cluster ' + str(rule_label_array[idx])

        if color_dict is not None:
            rule_color = color_dict[rule_label_array[idx]]
            if vertical:
                ax.scatter(
                    x = 0.25,
                    y = (len(decision_set) - i) * size_factor,
                    color = rule_color,
                    s = 100, 
                    marker = 's',
                    edgecolors='black'
                )
            else:
                ax.scatter(
                    x = i * size_factor - 0.5,
                    y = 0.5 - 0.075,
                    color = rule_color,
                    s = 2000, 
                    marker = 's',
                    edgecolors='black'
                )
        
        if vertical:
            ax.text(
                s = rule_string,
                x = 0.5,
                y = ((len(decision_set) - i)) * size_factor + 0.1,
                color = 'black',
                alpha = 1,
                fontweight = 'extra bold',
                fontsize = 12,
                va = 'top',
                ha = 'left'
            )
        else:
            ax.text(
                s = rule_string,
                x = i * size_factor + 0.1,
                y = 0.5,
                color = 'black',
                alpha = 1,
                fontweight = 'extra bold',
                fontsize = 64,
                va = 'top',
                ha = 'left'
            )
        
    if filename is not None:
        plt.savefig(filename, bbox_inches = 'tight', dpi = 300)
    

####################################################################################################