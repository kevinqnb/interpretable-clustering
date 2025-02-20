import numpy as np
import matplotlib.pyplot as plt
import graphviz as gv
from IPython.display import Image
from typing import Callable, List, Dict
from numpy.typing import NDArray
from intercluster.rules import Node
from intercluster.utils import flatten_labels, labels_to_assignment


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


def build_graph(
    node : Node,
    graph : Callable = None,
    parent_id : str = None,
    node_id : str = "0",
    feature_labels : List[str] = None, 
    leaf_colors : Dict[int, str] = None,
    data_scaler : Callable = None,
    cost : bool = True
):
    """
    Builds a graph representation of the node tree using 
    graphviz.
    
    Args:
        node (Node): Node object, which is the root node of the tree.
        graph (gv.Digraph, optional): The graph object to add nodes and edges to. Defaults to None.
        parent_id (str, optional): String identifier for parent. Defaults to None.
        node_id (str, optional): String identifier for the node. Defaults to "0".
        feature_labels (List[str], optional): List of feature labels used for display. 
            Defaults to None.
        leaf_colors (Dict[int:str], optional): Dictionary specifying colors for leaf nodes. 
            Defaults to None.
        newline (bool, optional): Whether to add newlines in the node labels. Defaults to True.
        cost (bool, optional): Whether to display the cost in the leaf nodes. Defaults to True.

    Returns:
        gv.Digraph: The graph object with the tree structure.
    """
    if graph is None:
        graph = gv.Digraph(format='png')
    
    # Ensure node_id and parent_id are strings
    node_id = str(node_id)
    if parent_id is not None:
        parent_id = str(parent_id)
    
    # Build the node's label (depending upon what's given as input).
    node_label = ""
    
    # For NON-leaf nodes:
    if node.type == 'internal':
        # Rescale the weights if necessary
        weights = node.condition.weights
        if data_scaler is not None and len(node.condition.features) > 1:
            weights = []
            for l, feature in enumerate(node.condition.features):
                feature_min = data_scaler.data_min_[feature]
                feature_max = data_scaler.data_max_[feature]
                weight_l = np.round(node.condition.weights[l] / (feature_max - feature_min), 2)
                weights.append(weight_l)
        else:
            weights = np.round(node.condition.weights, 2)
            #weights = [weight]
            
        # Rescale thresholds if necessary:
        threshold = node.condition.threshold
        if data_scaler is not None:
            if len(node.condition.features) == 1:
                feature_min = data_scaler.data_min_[node.condition.features[0]]
                feature_max = data_scaler.data_max_[node.condition.features[0]]
                threshold = threshold * (feature_max - feature_min) 
            else:             
                for l,feature in enumerate(node.condition.features):
                    feature_min = data_scaler.data_min_[feature]
                    feature_max = data_scaler.data_max_[feature]
                    threshold += (node.condition.weights[l] * feature_min) / (feature_max - feature_min)
                    
        
        if feature_labels is None:
            node_label += (
                f"Features {node.condition.features} \n Weights {weights}\n\u2264"
                f"{np.round(threshold, 3)}"
            )
        else:
            if all(x == 1 for x in weights):
                sum = ' + \n'.join([f'{feature_labels[f]}' for f in node.condition.features])
            else:
                sum = ' + '.join([f'{w}*{feature_labels[f]}' for w, f in zip(weights,
                                                                             node.condition.features)])
            
            node_label += (f"{sum} \n \u2264 {np.round(threshold, 3)}")
            
    # For leaf nodes:
    else:
        node_label += f"Cluster {node.label}\n"
        node_label += f"Size: {len(node.indices)}"
        if cost:
            node_label += f"\nCost: {np.round(node.cost, 3)}"
        else:
            #node_label += f"\n Cluster {node.label}"
            pass
        #node_label += f"\nLabel: {node.label}"
        
    if node.type == 'leaf' and (leaf_colors is not None):
        graph.node(node_id, label=node_label, fillcolor=leaf_colors[node.label], 
                   style='filled', penwidth='5', fontsize='64', fontname="times-bold")
    else:
        graph.node(node_id, label=node_label, fontsize='64', penwidth='5', fontname="times-bold")

    
    # Add an edge from the parent to the current node
    if parent_id is not None:
        graph.edge(parent_id, node_id, penwidth = '10')
    
    # Recursively add children
    if node.type == 'internal':
        if node.left_child is not None:
            build_graph(
                node = node.left_child,
                graph = graph,
                parent_id = node_id, 
                node_id = str(int(node_id) * 2 + 1),
                feature_labels = feature_labels,
                leaf_colors = leaf_colors,
                data_scaler = data_scaler,
                cost = cost
            )
        if node.right_child is not None:
            build_graph(
                node = node.right_child,
                graph = graph,
                parent_id = node_id, 
                node_id = str(int(node_id) * 2 + 2),
                feature_labels = feature_labels,
                leaf_colors = leaf_colors,
                data_scaler = data_scaler,
                cost = cost
            )
    
    return graph


####################################################################################################


def visualize_tree(
    root : Node,
    feature_labels : List[str] = None,
    leaf_colors : Dict[int, str] = None,
    data_scaler : Callable = None,
    cost : bool = True,
    output_file : str = 'tree',
):
    """
    Wrapper function for visualizing a Tree object by building a graphviz decision tree.

    Args:
        root (Node): Root Node object for the tree.
        
        feature_labels (List[str], optional): List of feature labels used for display.
            Each non-leaf Node object has a feature index attribute, and we use 
            feature_labels[index] to print the label associated with the index. Defaults to None.
            
        leaf_colors (Dict[int:str], optional): Dictionary specifying colors for leaf nodes.
            Each leaf Node object has a integer label attribute which can be used to 
            access the dictionary. Each item in the dictionary should be a 
            RGBA hexadecimal string. Defaults to None.
            
        newline (bool, optional): Whether to add newlines in the node labels. Defaults to True.
        
        cost (bool, optional): Whether to display the cost in the leaf nodes. Defaults to True.
        
        output_file (str, optional): File to save the resulting image. Defaults to 'tree.png'.

    Returns:
        IPython.display.Image: The image object for display in Jupyter notebooks.
    """
    graph = build_graph(root, feature_labels=feature_labels, leaf_colors=leaf_colors,
                        data_scaler=data_scaler, cost=cost)
    graph.attr(size="10,10", dpi="300", ratio="0.75")
    graph.render(output_file, format='png', cleanup=True)
    return Image(output_file + '.png')


####################################################################################################


def plot_decision_set(
    D : List[List[Node]],
    feature_labels : List[str],
    rule_labels : List[List[int]],
    cluster_colors : Dict[int, str],
    data_scaler : Callable = None,
    filename : str = None
):
    """
    Plots a decision set as a list of rules.
    
    Args:
        D (List[List[Condition]]): A list of rules, where each rule is a list of Condition objects.
        
        feature_labels (List[str]): List of feature labels used for display.
        
        rule_labels (List[List[int]]): List of rule labels for each rule.
        
        cluster_colors (Dict[int:str]): Dictionary specifying colors for leaf nodes.
        
        data_scaler (Callable, optional): Scaler object used to scale the data. Defaults to None.
            NOTE: If used, this assumes that the data was scaled to a 0-1 range, and will simply 
                adjust to undo that specific form of scaling.
        
        filename (str, optional): File to save the resulting image. Defaults to None
    """
    fig,ax = plt.subplots(figsize = (4, 6), dpi = 300)
    ax.set_xlim(0, len(D) + 0.1)
    ax.set_ylim(0.9/1.5, len(D) + 0.1)
    ax.axis('off')
    #ax.set_aspect('equal')
    
    # Order rules by cluster labels
    ordering = np.ndarray.flatten(np.argsort(rule_labels, axis = 0))
    for i, idx in enumerate(ordering):
        rule = D[idx]
        rule_string = 'If '
        
        # Every condition except the last node, which should be a leaf
        for j, condition in enumerate(rule):            
            rule_string += '('
            # Sum of weights and features
            for l,feature in enumerate(condition.features):
                if l > 0:
                    rule_string += ' + '

                # Rescale the weights if necessary
                if data_scaler is not None and len(condition.features) > 1:
                    feature_min = data_scaler.data_min_[feature]
                    feature_max = data_scaler.data_max_[feature]
                    weight_l = np.round(condition.weights[l] / (feature_max - feature_min), 2)
                else:
                    weight_l = np.round(condition.weights[l], 2)
                    
                if weight_l != 1:
                    rule_string += str(weight_l) + ' '
                    
                rule_string += str(feature_labels[feature])
                
            
            # Add the threshold
            if condition.direction == -1:
                rule_string += r' $\leq$ '
            else:
                rule_string += r' $>$ '
                
            threshold = condition.threshold
            
            # Rescale if necessary:
            if data_scaler is not None:                
                for l,feature in enumerate(condition.features):
                    feature_min = data_scaler.data_min_[feature]
                    feature_max = data_scaler.data_max_[feature]
                    threshold += (condition.weights[l] * feature_min) / (feature_max - feature_min)
                    
                if len(condition.features) == 1:
                    feature_min = data_scaler.data_min_[condition.features[0]]
                    feature_max = data_scaler.data_max_[condition.features[0]]
                    threshold = threshold * (feature_max - feature_min) 
                
            rule_string += str(np.round(threshold, 3))
            rule_string += ')'
            
            if j >= len(rule) - 2:
                rule_string += f'\n'
            elif j % 2 == 1:
                rule_string += r' $\&$ ' + f'\n'
            else:
                rule_string += r' $\&$ '
                
        rule_string += 'Then cluster ' + str(list(rule_labels[idx]))
        rule_color = cluster_colors[list(rule_labels[idx])[0]]
        ax.scatter(
            x = 0.25,
            y = (len(D) - i)/1.5,
            color = rule_color,
            s = 50, 
            marker = 's',
            edgecolors='black'
        )
        
        ax.text(
            s = rule_string,
            x = 0.5,
            y = ((len(D) - i) + 0.055)/1.5,
            color = 'black',
            alpha = 1,
            fontweight = 'extra bold',
            fontsize = 10,
            va = 'top',
            ha = 'left'
        )
        
    if filename is not None:
        plt.savefig(filename, bbox_inches = 'tight', dpi = 300)
    plt.show()
    

####################################################################################################


def experiment_plotter(
    mean_df,
    std_df,
    xlabel,
    ylabel,
    domain,
    cmap,
    legend = True,
    filename = None,
    baseline_list = None
):
    fig,ax = plt.subplots(figsize = (6,4))
    if baseline_list is None:
        baseline_list = ['KMeans']
    baseline_linestyles = ['-', 'dashed']
    baselines = [_ for _ in mean_df.columns if _ in baseline_list]
    modules = [_ for _ in mean_df.columns if _ not in baselines]
    
    end = -1
    for i,b in enumerate(baselines):
        ax.hlines(
            mean_df[b].iloc[0],
            xmin = domain.min().min(),
            xmax = domain.max().max(),
            color = 'k',
            label = fr'$\texttt{{{b}}}$',
            linestyle = baseline_linestyles[i],
            linewidth = 3,
            alpha = 0.6
        )
    
    for i,m in enumerate(modules):
        ax.plot(
            np.array(domain[m]),
            np.array(mean_df[m]),
            linewidth = 3,
            marker='o',
            markersize = 5,
            c = cmap(i),
            label = fr'$\texttt{{{m}}}$'
        )
        ax.fill_between(
            np.array(domain[m]), 
            np.array(mean_df[m]) - np.array(std_df[m]),
            np.array(mean_df[m]) + np.array(std_df[m]),
            color= cmap(i),
            alpha=0.1
        )

    #ticks = domain[::2]
    #labels = [str(i) for i in ticks] 
    #plt.xticks(ticks, labels)

    if legend:
        plt.legend(loc = 'upper right', bbox_to_anchor=(2, 1))
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if filename is not None:
        plt.savefig(filename, bbox_inches = 'tight', dpi = 300)
    #plt.show()
        
        
####################################################################################################