import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
#import pygraphviz
#import graphviz as gv
from IPython.display import Image
from typing import Callable, List, Dict, Tuple
from numpy.typing import NDArray
from intercluster.decision_trees import Node
from intercluster.utils import can_flatten, flatten_labels, labels_to_assignment


####################################################################################################


def plot_decision_boundaries(
    model : Callable,
    X : NDArray,
    ax : Callable = None,
    resolution : int = 100,
    label_array = False
):
    """
    Plots the decision boundaries of a given model.

    Args:
        model (Callable): Prediction object which should have a predict() method.
        X (NDArray): Dataset fitted to the model. 
        ax (matplotlib axes, optional): Axes for plotting. 
        resolution (int, optional): Number of points on the meshgrid, controls the 
            resolution of the contour lines. Defaults to 100.
        label_array (bool, optional): `True` if the output of the model's prediction is a set of 
            labels represented as as 1d array. `False` if the labels should instead be a 2 label
            set. Defaults to False.
    """
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
    
    # Plot the decision boundaries
    if ax is None:
        plt.contour(xx, yy, Z, levels = len(np.unique(Z)), colors='k', linestyles='dashed',
                    alpha = 0.8, linewidths = 1.5)
    else:
        ax.contour(xx, yy, Z, levels = len(np.unique(Z)), colors='k', linestyles='dashed',
                   alpha = 0.8, linewidths = 1.5)
        

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


def draw_tree(
    root : Node,
    feature_labels : List[str] = None,
    data_scaler : Callable = None,
    cmap : Callable = None,
    display_node_info : bool = True,
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

        display_node_info (bool): Boolean for deciding whether to display 
            additional node information (size, cost, etc.).
        
        output_file (str, optional): File to save the resulting image. Defaults to None.
    """
    G = nx.DiGraph()
    build_networkx_graph(G, root)
    node_colors = [
        cmap(node.label) if (node.type == 'leaf' and cmap is not None)
        else 'white' for node in G.nodes
    ]
    node_labels = {
        node : node.condition.display(scaler = data_scaler, feature_labels = feature_labels)
        if node.type == 'internal'
        else "Cluster " + str(node.label)
        for node in G.nodes
    }
    #node_labels = {node : "" for node in G.nodes}
    node_sizes = [12500 if node.type == "internal" else 7500 for node in G.nodes]
    #node_sizes = [100 if node.type == "internal" else 100 for node in G.nodes]

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
    decision_set : List[List[Node]],
    rule_labels : List[List[int]],
    feature_labels : List[str] = None,
    data_scaler : Callable = None,
    cmap : Callable = None,
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
        
        cmap (Callable): Matplotlib colormap. Should be callable so that cmap(i) gives the 
            color for cluster i.
        
        filename (str, optional): File to save the resulting image. Defaults to None
    """
    assert can_flatten(rule_labels), "Each rule must have exactly one label."
    rule_label_array = flatten_labels(rule_labels)

    max_rule_length = np.max([len(r) for r in decision_set])
    size_factor = max(1, max_rule_length // 2)

    fig,ax = plt.subplots(figsize = (4, len(decision_set) * size_factor), dpi = 300)
    ax.set_xlim(0, 5)
    ax.set_ylim(0.9, (len(decision_set) + 0.1) * size_factor)
    ax.axis('off')
    #ax.set_aspect('equal')
    
    # Order rules by cluster labels
    ordering = np.argsort(rule_label_array)
    for i, idx in enumerate(ordering):
        rule = decision_set[idx]
        rule_string = 'If '
        
        # Every condition except the last node, which should be a leaf
        for j, condition in enumerate(rule):            
            rule_string += '('
            rule_string += condition.display(
                scaler=data_scaler,
                feature_labels=feature_labels,
                newline = False
            )
            rule_string += ')'
            
            if j >= len(rule) - 1:
                rule_string += f'\n'
            elif j % 2 == 1:
                rule_string += r" $\&$ " + f'\n'
            else:
                rule_string += r" $\&$ "
                
        rule_string += 'Then cluster ' + str(rule_label_array[idx])

        if cmap is not None:
            rule_color = cmap(rule_label_array[idx])
            ax.scatter(
                x = 0.25,
                y = (len(decision_set) - i) * size_factor,
                color = rule_color,
                s = 100, 
                marker = 's',
                edgecolors='black'
            )
        
        ax.text(
            s = rule_string,
            x = 0.5,
            y = ((len(decision_set) - i)) * size_factor + 0.1,
            color = 'black',
            alpha = 1,
            fontweight = 'extra bold',
            fontsize = 18,
            va = 'top',
            ha = 'left'
        )
        
    if filename is not None:
        plt.savefig(filename, bbox_inches = 'tight', dpi = 300)
    

####################################################################################################


def experiment_plotter(
    measurement_df : pd.DataFrame,
    std_df : pd.DataFrame,
    domain_df : pd.DataFrame,
    xlabel : str,
    ylabel : str,
    cmap : Callable,
    baseline_list : List[str] = None,
    legend : bool = True,
    xlim : Tuple[float, float] = None,
    ylim : Tuple[float, float] = None,
    xaxis : bool = True,
    yaxis : bool = True,
    filename : str = None
):
    """
    Plots experiment results (for coverage/cost experiments).

    Args:
        measurement_df (pd.DataFrame): Size (t x m) pandas dataframe where each of the m columns 
            corresponds to an experiment module, containing experimental measurements 
            over t different trial settings. 
        
        std_df (pd.DataFrame): Size (t x m) pandas dataframe where each of the m columns 
            corresponds to an experiment module, containing the standard deviation 
            for experimental measurements over t different trial settings. In other words, 
            this shows std for the measurement dataframe.

        domain_df (pd.DataFrame): Size (t x m) pandas dataframe where each of the m columns 
            corresponds to an experiment module, containing the domain values for each
            of experimental measurements over t different trial settings. In other words, 
            these are the x-axis values ofr the measurement dataframe.

        xlabel (str): Label to plot on the x-axis.

        ylabel (str): Label to plot on the y-axis.

        cmap (Callable): cmap (Callable): Matplotlib colormap. Should be callable so that cmap(i) 
            gives the color for cluster i.

        baseline_list (List[str]): List of names for modules within the measurement/std/domain 
            dataframes to treat like baselines for comparison. This usually means they have 
            values which do not vary over the t trials and can be plotted as horizontal 
            baselines.

        legend (bool): If True, plot the legend. Defaults to False.

        xlim (Tuple[float, float]): Tuple of (low, high) x limit values.

        ylim (Tuple[float, float]): Tuple of (low, high) y limit values.

        filename (str): If given, saves the plot. Defaults to None in which case nothing is saved.

    """
    fig,ax = plt.subplots(figsize = (6,4))
    if baseline_list is None:
        baseline_list = ['KMeans']
    baseline_linestyles = ['-', 'dashed']
    baselines = [_ for _ in measurement_df.columns if _ in baseline_list]
    modules = [_ for _ in measurement_df.columns if _ not in baselines]
    
    for i,b in enumerate(baselines):
        ax.hlines(
            measurement_df[b].iloc[0],
            xmin = domain_df.min().min(),
            xmax = domain_df.max().max(),
            color = 'k',
            label = fr'$\texttt{{{b}}}$',
            linestyle = baseline_linestyles[i],
            linewidth = 3,
            alpha = 0.6
        )
    
    for i,m in enumerate(modules):
        ax.plot(
            np.array(domain_df[m]),
            np.array(measurement_df[m]),
            linewidth = 6,
            marker='o',
            markersize = 8,
            c = cmap(i),
            label = fr'$\texttt{{{m}}}$'
        )
        ax.fill_between(
            np.array(domain_df[m]), 
            np.array(measurement_df[m]) - np.array(std_df[m]),
            np.array(measurement_df[m]) + np.array(std_df[m]),
            color= cmap(i),
            alpha=0.3
        )

    if legend:
        plt.legend(loc = 'upper right', bbox_to_anchor=(2, 1))

    if xlim is not None:
        plt.xlim(xlim)
    
    if ylim is not None:
        plt.ylim(ylim)

    if not xaxis:
        plt.xticks([])
    if not yaxis:
        plt.yticks([])
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if filename is not None:
        plt.savefig(filename, bbox_inches = 'tight', dpi = 300)
        
        
####################################################################################################