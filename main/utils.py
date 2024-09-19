import numpy as np
import pandas as pd 
import pygraphviz as pgv
from IPython.display import Image

####################################################################################################
def kmeans_plus_plus_initialization(X, k):
    """
    Implements KMeans++ initialization to choose `k` initial cluster centers.

    Args:
    X : np.ndarray : Data points (n_samples, n_features)
    k : int : Number of clusters
    
    Returns:
    centers : np.ndarray : Initialized cluster centers (k, n_features)
    """
    n, m = X.shape
    centers = np.empty((k, m))
    
    # Randomly choose the first center
    first_center = np.random.choice(n)
    centers[0] = X[first_center,:]


    distances = np.full(n, np.inf)
    for i in range(1, k):
        distances = np.minimum(distances, np.sum((X - centers[i - 1]) ** 2, axis=1))
        
        # Choose the next center with probability proportional to the squared distance
        probabilities = distances / distances.sum()
        next_center_index = np.random.choice(n, p=probabilities)
        centers[i] = X[next_center_index]

    return centers

####################################################################################################


def build_graph(custom_node, graph=None, parent_id=None, node_id=0,
                feature_labels = None, leaf_colors = None):
    """
    Recursively builds a pygraphviz graph for visualization of a decision tree.
    Given a single node (starting with the root), it will recursively parse the 
    tree, adding nodes and edges as it goes down.

    Args:
        custom_node (Node): Node object 
    
        graph (AGraph, optional): pygraphviz AGraph object to add to. Defaults to None, in which 
            case a graph will be automatically initialized for you.
            
        parent_id (str, optional): String identifier for parent. Defaults to None.
            By convention we take ids as type str(int). For example "5" is a valid 
            identifier.
            
        node_id (str, optional): String identifies for the node. Defaults to "0".
            By convention we take ids as type str(int). For example "5" is a valid 
            identifier.

        feature_labels (List[str], optional): List of feature labels used for display.
            Each non-leaf Node object has a feature index attribute, and we use 
            feature_labels[index] to print the label associated with the index. Defaults to None.
            
        leaf_colors (Dict[int:str], optional): Dictionary specifying colors for leaf nodes.
            Each leaf Node object has a integer label attribute which can be used to 
            access the dictionary. Each item in the dictionary should be a 
            RGBA hexadecimal string. Defaults to None.

    Returns:
        _type_: _description_
    """
    if graph is None:
        graph = pgv.AGraph(directed=True)
    
    # Build the node's label (depending upon what's given as input).
    node_label = ""
    
    # For NON-leaf nodes:
    if custom_node.type == 'node':
        if feature_labels is None:
            node_label += (f"Feature {custom_node.feature} \n\u2264 {np.round(custom_node.threshold, 3)}")
        else:
            node_label += (f"{feature_labels[custom_node.feature]} \n\u2264 {np.round(custom_node.threshold, 3)}")
            
    # For leaf nodes:
    else:
        node_label += f"# Points: {custom_node.points}"
        node_label += f"\nCost: {np.round(custom_node.cost, 3)}"


    # Add the node to the graph:
    # For leaf nodes:
    if custom_node.type == 'leaf' and (leaf_colors is not None):
        graph.add_node(node_id, label=node_label, fillcolor = leaf_colors[custom_node.label], 
                       style = 'filled', penwidth = 5, fontsize = 24, fontname="times-bold")
        
    # For NON-leaf nodes:
    else:
        graph.add_node(node_id, label=node_label, fontsize = 30, penwidth = 5,
                       fontname="times-bold")
        
    # Add an edge connecting to its parent.
    if parent_id is not None:
        graph.add_edge(parent_id, node_id, arrowsize = 2, penwidth = 3)

    # Recursively add left and right children (if any).
    if custom_node.left_child is not None:
        build_graph(custom_node.left_child, graph, parent_id = node_id, 
                    node_id=str(int(node_id) * 2 + 1),
                    feature_labels = feature_labels, leaf_colors = leaf_colors)
        
    if custom_node.right_child is not None:
        build_graph(custom_node.right_child, graph, parent_id = node_id, 
                    node_id= str(int(node_id) * 2 + 2),
                    feature_labels = feature_labels, leaf_colors = leaf_colors)

    return graph


####################################################################################################

def visualize_tree(custom_root, output_file='tree.png', feature_labels = None, leaf_colors = None):
    """
    Wrapper function for visualizing a Tree object by building a pygraphviz decision tree.

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
        _type_: _description_
    """
    graph = build_graph(custom_root, feature_labels = feature_labels, leaf_colors = leaf_colors)
    graph.layout(prog='dot')
    #graph.draw(output_file, args='-Gsize=1 -Gratio=1 -Gdpi=200')
    graph.draw(output_file)
    #return Image(output_file, width = width, height = height)
    return Image(output_file)

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

# Given a data matrix, add a new row and a new column to its end points, i.e. 
# the new column/row is the last column/row in the matrix
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