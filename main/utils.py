import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import graphviz as gv
from IPython.display import Image
from rules import *

####################################################################################################

def kmeans_cost(X, clustering, centers):
    """
    Computes the squared L2 norm cost of a clustering with an associated set of centers

    Args:
        X (np.ndarray): (n x m) Dataset
        
        clustering (List[List[int]]): 2d List of integers representing a clustering.
            clustering[i] is a list of indices j for the items within the cluster with label i. 
            
        centers (np.ndarray): (k x m) Set of representative centers for each of the k clusters.

    Returns:
        cost (float): Total cost of the clustering.
    """
    cost = 0
    for i, cluster in enumerate(clustering):
        center = centers[i,:]
        cost += np.sum(np.linalg.norm(X[cluster,:] - center, axis = 1)**2)
        #cost +=  np.sum((X[cluster,:] - center)**2)
    return cost

####################################################################################################

def kmedians_cost(X, clustering, centers):
    """
    Computes the L1 norm cost of a clustering with an associated set of centers

    Args:
        X (np.ndarray): (n x m) Dataset
        
        clustering (List[List[int]]): 2d List of integers representing a clustering.
            clustering[i] is a list of indices j for the items within the cluster with label i. 
            
        centers (np.ndarray): (k x m) Set of representative centers for each of the k clusters.

    Returns:
        cost (float): Total cost of the clustering.
    """
    cost = 0
    for i, cluster in enumerate(clustering):
        center = centers[i,:]
        cost += np.sum(np.abs(X[cluster,:] - center))
    return cost

####################################################################################################

def clustering_to_labels(clustering, label_array = None):
    """
    Takes an input clustering and returns its associated list of labels.
    NOTE: Assumes clusters are indexed 0 -> (k - 1) and items are indexed 0 -> (n - 1).
    
    Args:
        clustering (List[List[int]]): 2d List of integers representing a clustering.
            clustering[i] is a list of indices j for the items within the cluster with label i. 
            
        label_array (np.ndarray): Input array to modify. Useful if some items are not clustered,
            i.e. will have label NaN.  

    Returns:
        labels (np.ndarray): List of integers where an entry at index i has value j if the 
            item associated with index i is present within cluster j. 
    """
    if label_array is None:
        lens = [len(i) for i in clustering]
        labels = np.zeros(np.sum(lens)) - 1
    else:
        labels = label_array
    
    for i,cluster in enumerate(clustering):
        for j in cluster:
            labels[j] = i
            
    return labels
        
####################################################################################################

def labels_to_clustering(labels):
    """
    Takes an input list of labels and returns its associated clustering.
    NOTE: Assumes clusters are indexed 0 -> (k - 1) and items are indexed 0 -> (n - 1).
    
    Args:
        labels (np.ndarray): List of integers where an entry at index i has value j if the 
            item associated with index i is present within cluster j. 

    Returns:
        clustering (List[List[int]]): 2d List of integers representing a clustering.
            clustering[i] is a list of indices j for the items within the cluster with label i. 
    """
    clustering = [[] for i in range(int(np.max(labels)) + 1)]
    for i,j in enumerate(labels):
        if not np.isnan(j):
            clustering[int(j)].append(i)
        
    return clustering

####################################################################################################

def kmeans_plus_plus_initialization(X, k, random_seed = None):
    """
    Implements KMeans++ initialization to choose k initial cluster centers.

    Args:
    X (np.ndarray): Data points (n_samples, n_features)
    k (int): Number of clusters
    
    Returns:
    centers (np.ndarray): Initialized cluster centers (k, n_features)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    n, m = X.shape
    centers = np.empty((k, m))
    centers[:] = np.inf
    
    # Randomly choose the first center
    first_center = np.random.choice(n)
    centers[0,:] = X[first_center,:]

    for i in range(1, k):
        diffs = X[np.newaxis, :, :] - centers[:, np.newaxis, :]
        distances = np.sum(diffs ** 2, axis=-1)
        min_distances = np.min(distances, axis = 0)
        
        # Choose the next center with probability proportional to the squared distance
        probabilities = min_distances / min_distances.sum()
        next_center_index = np.random.choice(n, p=probabilities)
        centers[i,:] = X[next_center_index,:]

    return centers

####################################################################################################

def plot_decision_boundaries(model, X, ax = None, resolution = 100):
    """
    Plots the decision boundaries of a given model.

    Args:
        model (Object): Object which requires a predict() method.
        X (_type_): Dataset fitted to the model. 
        ax (matplotlib axes, optional): Axes for plotting. 
        resolution (int, optional): Number of points on the meshgrid, controls the 
            resolution of the contour lines. Defaults to 100.
    """
    # Define the axis boundaries of the plot
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    
    # Create a mesh grid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    # Predict the classification for each point in the mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundaries
    if ax is None:
        plt.contour(xx, yy, Z, levels = len(np.unique(Z)), colors='k', linestyles='dashed',
                    alpha = 0.8, linewidths = 1.5)
    else:
        ax.contour(xx, yy, Z, levels = len(np.unique(Z)), colors='k', linestyles='dashed',
                   alpha = 0.8, linewidths = 1.5)

####################################################################################################

def build_graph(custom_node, graph=None, parent_id=None, node_id="0", feature_labels=None, 
                leaf_colors=None, newline=True):
    """
    Builds a graph representation of the custom_node tree using 
    graphviz.
    
    Args:
        custom_node (Node): The root node of the tree.
        graph (gv.Digraph, optional): The graph object to add nodes and edges to. Defaults to None.
        parent_id (str, optional): String identifier for parent. Defaults to None.
        node_id (str, optional): String identifier for the node. Defaults to "0".
        feature_labels (List[str], optional): List of feature labels used for display. 
            Defaults to None.
        leaf_colors (Dict[int:str], optional): Dictionary specifying colors for leaf nodes. 
            Defaults to None.
        newline (bool, optional): Whether to add newlines in the node labels. Defaults to True.

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
    if custom_node.type == 'node':
        if feature_labels is None:
            if newline:
                node_label += (f"Features {custom_node.features} \n Weights {custom_node.weights}\n \u2264 {np.round(custom_node.threshold, 3)}")
            else:
                node_label += (f"Features {custom_node.features} Weights {custom_node.weights} \n \u2264 {np.round(custom_node.threshold, 3)}")
        else:
            if all(x == 1 for x in custom_node.weights):
                sum = ' + \n'.join([f'{feature_labels[f]}' for f in custom_node.features])
            else:
                sum = ' + '.join([f'{w}*{feature_labels[f]}' for w, f in zip(custom_node.weights,
                                                                             custom_node.features)])
            if newline:
                node_label += (f"{sum} \n \u2264 {np.round(custom_node.threshold, 3)}")
            else:
                node_label += (f"{sum} \u2264 {np.round(custom_node.threshold, 3)}")
            
    # For leaf nodes:
    else:
        node_label += f"Size: {custom_node.size}"
        node_label += f"\nCost: {np.round(custom_node.cost, 3)}"
        #node_label += f"\nLabel: {custom_node.label}"
        
    if custom_node.type == 'leaf' and (leaf_colors is not None):
        graph.node(node_id, label=node_label, fillcolor=leaf_colors[custom_node.label], 
                   style='filled', penwidth='5', fontsize='36', fontname="times-bold")
    else:
        graph.node(node_id, label=node_label, fontsize='36', penwidth='5', fontname="times-bold")

    
    # Add an edge from the parent to the current node
    if parent_id is not None:
        graph.edge(parent_id, node_id)
    
    # Recursively add children
    if custom_node.type == 'node':
        if custom_node.left_child is not None:
            build_graph(custom_node.left_child, graph, parent_id=node_id, 
                        node_id=str(int(node_id) * 2 + 1),
                        feature_labels=feature_labels, leaf_colors=leaf_colors, newline=newline)
        if custom_node.right_child is not None:
            build_graph(custom_node.right_child, graph, parent_id=node_id, 
                        node_id=str(int(node_id) * 2 + 2),
                        feature_labels=feature_labels, leaf_colors=leaf_colors, newline=newline)
    
    return graph

####################################################################################################

def visualize_tree(custom_root, output_file='tree', feature_labels=None, leaf_colors=None,
                   newline=True):
    """
    Wrapper function for visualizing a Tree object by building a graphviz decision tree.

    Args:
        custom_root (Node): Root Node object for the tree.
        
        output_file (str, optional): File to save the resulting image. Defaults to 'tree.png'.
        
        feature_labels (List[str], optional): List of feature labels used for display.
            Each non-leaf Node object has a feature index attribute, and we use 
            feature_labels[index] to print the label associated with the index. Defaults to None.
            
        leaf_colors (Dict[int:str], optional): Dictionary specifying colors for leaf nodes.
            Each leaf Node object has a integer label attribute which can be used to 
            access the dictionary. Each item in the dictionary should be a 
            RGBA hexadecimal string. Defaults to None.

    Returns:
        IPython.display.Image: The image object for display in Jupyter notebooks.
    """
    graph = build_graph(custom_root, feature_labels=feature_labels, leaf_colors=leaf_colors,
                        newline=newline)
    graph.attr(size="10,10", dpi="300", ratio="0.75")
    graph.render(output_file, format='png', cleanup=True)
    return Image(output_file + '.png')

####################################################################################################

def rule_grid(X, g):
    """
    Builds a g x g grid of rules around a given dataset. 

    Args:
        data (np.ndarray): Input (n x m) dataset. 
        g (int): _description_

    Returns:
        grid (List[Rule]): List of Rule objects describing the grid.
    """
    # Step 1: Find the min and max of the dataset along both dimensions
    x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
    y_min, y_max = np.min(X[:, 1]), np.max(X[:, 1])

    # Step 2: Calculate the step size for both dimensions
    x_step = (x_max - x_min) / g
    y_step = (y_max - y_min) / g

    # Step 3: Create the grid cells with logical conditions
    grid_cells = []

    for i in range(g):
        for j in range(g):
            x_start = x_min + i * x_step
            x_end = x_start + x_step
            y_start = y_min + j * y_step
            y_end = y_start + y_step
            
            # Logical conditions defining the current cell
            ineq = ['>', '<', '>', '<']
            if i == 0:
                ineq[0] = '>='
            elif i == g - 1:
                ineq[1] = '<='
            if j == 0:
                ineq[2] = '>='
            elif j == g - 1:
                ineq[3] = '<='
            
            cell_conditions = [
                Condition(0, ineq[0], x_start),
                Condition(0, ineq[1], x_end),
                Condition(1, ineq[2], y_start),
                Condition(1, ineq[3], y_end)
            ]
            
            grid_cells.append(Rule([Term(cell_conditions)]))

    return grid_cells

####################################################################################################

def remove_rows_cols(matrix, indices):
    """
    Given a data matrix, and a list of integer valued indices,
    removes the rows and columns corresponding to the indices.

    Args:
        matrix (np.ndarray): (n x m) Dataset in the form of a numpy array.
        indices (_type_): List of indices where each index i corresponds to both a row 
            and a column to be removed from the matrix.

    Returns:
        (np.ndarray): Modified dataset with rows/columns removed.
    """
    
    indices = sorted(indices, reverse=True)
    for idx in indices:
        # Remove the specified row
        matrix = np.delete(matrix, idx, axis=0)
        # Remove the specified column
        matrix = np.delete(matrix, idx, axis=1)
    return matrix

####################################################################################################

def add_row_col(matrix, new_row, new_col):
    """
    Given a data matrix, add a new row and a new column.
    The new column/row is added as the last column/row in the matrix.

    Args:
        matrix (np.ndarray): (n x m) Dataset in the form of a numpy array.
        new_row (np.ndarray): Length m array to be added as a new row.
        new_col (np.ndarray): Length n + 1 array to be added as a new column.

    Returns:
        (np.ndarray): Modified (n + 1 x m + 1) dataset with rows/columns added.
    """
    # Ensure the new row is a 1D array of the correct length
    new_row = np.array(new_row)
    assert new_row.shape[0] == matrix.shape[1], "New row length must match number of columns in the matrix"
    
    # Append the new row to the matrix
    matrix = np.vstack([matrix, new_row])
    
    # Ensure the new column is a 1D array of the correct length
    new_col = np.array(new_col)
    assert new_col.shape[0] == matrix.shape[0], "New column length must match number of rows in the updated matrix"
    
    # Append the new column to the matrix
    matrix = np.hstack([matrix, new_col.reshape(-1, 1)])
    
    return matrix
    
####################################################################################################