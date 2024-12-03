import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import graphviz as gv
from IPython.display import Image
from rules import *
from collections.abc import Iterable


####################################################################################################

def kmeans_cost(X, assignment, centers):
    """
    Computes the squared L2 norm cost of a clustering with an associated set of centers

    Args:
        X (np.ndarray): (n x m) Dataset
        
        assignment (np.ndarray: bool): n x k boolean (or binary) matrix with entry (i,j) 
            being True (1) if point i belongs to cluster j and False (0) otherwise. 
            
        centers (np.ndarray): (k x m) Set of representative centers for each of the k clusters.

    Returns:
        cost (float): Total cost of the clustering.
    """
    
    k = assignment.shape[1]
    cost = 0
    for i in range(k):
        points = X[assignment[:,i] == 1]
        center = centers[i,:]
        cost += np.sum(np.linalg.norm(points - center, axis = 1)**2)
    return cost
    
####################################################################################################

def kmedians_cost(X, assignment, centers):
    """
    Computes the L1 norm cost of a clustering with an associated set of centers

    Args:
        X (np.ndarray): (n x m) Dataset
        
        assignment (np.ndarray: bool): n x k boolean (or binary) matrix with entry (i,j) 
            being True (1) if point i belongs to cluster j and False (0) otherwise. 
            
        centers (np.ndarray): (k x m) Set of representative centers for each of the k clusters.

    Returns:
        cost (float): Total cost of the clustering.
    """
    
    k = assignment.shape[1]
    cost = 0
    for i in range(k):
        points = X[assignment[:,i] == 1]
        center = centers[i,:]
        cost += np.sum(np.abs(points - center))
    return cost

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

'''
def clustering_to_labels(clustering, n = None):
    """
    Takes an input clustering and returns its associated list of labels.
    NOTE: Assumes clusters are indexed 0 -> (k - 1) and items are indexed 0 -> (n - 1).
    
    Args:
        clustering (List[List[int]]): 2d List of integers representing a clustering.
            clustering[i] is a list of indices j for the items within the cluster with label i.  
            
        n (int, optional): Number of items. Defaults to None, in which case the number of
            of points is inferred from the given clustering.

    Returns:
        labels (List[int] OR List[List[int]]): List of integers where an entry at index i has value 
            j if the item associated with index i is present within cluster j. Alternatively,
            in a soft clustering where points have multiple labels, labels[i] is a list of 
            cluster labels j.
    """
    if n is None:
        lens = [len(i) for i in clustering]
        n = np.sum(lens)
        
    labels = [-1 for _ in range(n)]   
    
    for i,cluster in enumerate(clustering):
        for j in cluster:
            if labels[j] != -1:
                if isinstance(labels[j], (int, float)):
                    labels[j] = [labels[j], i]
                else:
                    labels[j].append(i)
            else:
                labels[j] = i
            
    return labels
        
####################################################################################################

def labels_to_clustering(labels, k = None):
    """
    Takes an input list of labels and returns its associated clustering.
    NOTE: Assumes clusters are indexed 0 -> (k - 1) and items are indexed 0 -> (n - 1).
    
    Args:
        labels (List[int] OR List[List[int]]): List of integers where an entry at index i has value 
            j if the item associated with index i is present within cluster j. Alternatively,
            in a soft clustering where points have multiple labels, labels[i] is a list of 
            cluster labels j.
            
        k (int, optional): Number of clusters. Defaults to None, in which case the number of
            clusters is inferred from the input labels.

    Returns:
        clustering (List[List[int]]): 2d List of integers representing a clustering.
            clustering[i] is a list of indices j for the items within the cluster with label i. 
    """
    if k is None:
        s = -1
        for l in labels:
            if np.max(l) > s:
                s = np.max(l)
                
        k = int(s) + 1
        
    clustering = [[] for _ in range(k)]
    for i,j in enumerate(labels):
        if isinstance(j, (int, float, np.integer)):
            if not np.isnan(j):
                clustering[int(j)].append(i)
        elif isinstance(j, Iterable) and not isinstance(j, (str, bytes)):
            for l in j:
                clustering[int(l)].append(i)
        else:
            raise ValueError("Invalid label type")
        
    return clustering
'''

####################################################################################################

def labels_to_assignment(labels, k = None):
    """
    Takes an input list of labels and returns its associated clustering matrix.
    NOTE: By convention, clusters are indexed [0,k) and items are indexed [0, n).
    
    Args:
        labels (List[int] OR List[List[int]]): List of integers where an entry at index i has value 
            j if the item associated with index i is present within cluster j. Alternatively,
            in a soft clustering where points have multiple labels, labels[i] is a list of 
            cluster labels j.
            
        k (int, optional): Number of clusters. Defaults to None, in which case the number of
            clusters is inferred from the input labels.

    Returns:
        assignment_matrix (np.ndarray): n x k boolean matrix with entry (i,j) being True
            if point i belongs to cluster j and False otherwise.
    """
    # Infer if k is not provided
    if k is None:
        s = -1
        for l in labels:
            if np.max(l) > s:
                s = np.max(l) 
        k = int(s) + 1
        
    assignment_matrix = np.zeros((len(labels), k), dtype = bool)
    for i,j in enumerate(labels):
        if isinstance(j, (int, float, np.integer)):
            if not np.isnan(j):
                assignment_matrix[i, int(j)] = True
        elif isinstance(j, Iterable) and not isinstance(j, (str, bytes)):
            for l in j:
                assignment_matrix[i, int(l)] = True
        else:
            raise ValueError("Invalid label type")
        
    return assignment_matrix


####################################################################################################

def assignment_to_labels(assignment):
    """
    Takes an input n x k boolean assignment matrix, and outputs a list of labels for the 
    datapoints.
     
    NOTE: By convention, clusters are indexed [0,k) and items are indexed [0, n).
    
    Args:
        assignment_matrix (np.ndarray): n x k boolean matrix with entry (i,j) being True
            if point i belongs to cluster j and False otherwise.

    Returns:
        labels (List[int] OR List[List[int]]): List of integers where an entry at index i has value 
            j if the item associated with index i is present within cluster j. Alternatively,
            in a soft clustering where points have multiple labels, labels[i] is a list of 
            cluster labels j.
    """
    labels = []
    for i, assign in enumerate(assignment):
        i_labels = np.where(assign)[0]
        
        if len(i_labels) == 1:
            labels.append(i_labels[0])
        else:
            labels.append(list(i_labels))
            
    return labels

####################################################################################################

def traverse(node, path=None):
    """
    Traverses a binary tree in a depth-first manner, yielding nodes as they are visited.
    
    Args:
        node (Node): Root node of the tree.
    
    Yields:
        path (List[(Node, str)]): List of node objects visited on the current path.
            If the path followed a left child, the corresponding string is 'left'.
            Otherwise, the string is 'right'.
    """
    if path is None:
        path = []
    
    path_update = path + [node]
    
    yield path_update
    
    if node.left_child is not None:
        left_path = path + [(node, 'left')]
        yield from traverse(node.left_child, left_path)
    if node.right_child is not None:
        right_path = path + [(node, 'right')]
        yield from traverse(node.right_child, right_path)

####################################################################################################

def find_leaves(root):
    """
    Given the root of a tree, finds all leaf nodes in the tree.
    
    Args:
        root (Node): Root of the tree.
    
    Returns:
        leaves (dict[int: List[Node]]): Dictionary where the key is the label of the leaf node,
            and the value is a list of nodes along the path from root to leaf (inclusive). 
    """
    
    leaves = {}
    
    for path in traverse(root):
        last_node = path[-1]
        if last_node.type == 'leaf':
            leaves[last_node.label] = path
            
    return leaves

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

def plot_multiclass_decision_boundaries(model, X, ax = None, resolution = 100, cmap = None):
    """
    Plots the decision boundaries of a given model. In contrast to 
    plot_decision_boundaries, this function is specifically designed to handle 
    situations where points may belong to multiple classes. The .predict() method of 
    the model should return a 0/1 matrix with size n x k (where n is the number of 
    points and k is the number of classes), indicating the class membership of a point. 

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
    
    if isinstance(Z, list):
        Z = labels_to_assignment(Z)
    
    # Plot the decision boundaries for each class
    if ax is None:
        ax = plt.gca()
    
    for i in range(Z.shape[1]):
        Z_class = Z[:, i].reshape(xx.shape)
        if cmap is None:
            ax.contour(xx, yy, Z_class, levels=[0.5], colors='k', linestyles='dashed',
                    alpha=0.8, linewidths=1.5)
        else:
            c = cmap(i)
            ax.contour(xx, yy, Z_class, levels=[0.5], colors=[c], linestyles='dashed',
                    alpha=1, linewidths=4)

####################################################################################################

def build_graph(custom_node, graph=None, parent_id=None, node_id="0", feature_labels=None, 
                leaf_colors=None, newline=True, cost = True):
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
        node_label += f"Cluster {custom_node.label}\n"
        node_label += f"Size: {custom_node.size}"
        if cost:
            node_label += f"\nCost: {np.round(custom_node.cost, 3)}"
        else:
            #node_label += f"\n Cluster {custom_node.label}"
            pass
        #node_label += f"\nLabel: {custom_node.label}"
        
    if custom_node.type == 'leaf' and (leaf_colors is not None):
        graph.node(node_id, label=node_label, fillcolor=leaf_colors[custom_node.label], 
                   style='filled', penwidth='5', fontsize='64', fontname="times-bold")
    else:
        graph.node(node_id, label=node_label, fontsize='64', penwidth='5', fontname="times-bold")

    
    # Add an edge from the parent to the current node
    if parent_id is not None:
        graph.edge(parent_id, node_id, penwidth = '10')
    
    # Recursively add children
    if custom_node.type == 'node':
        if custom_node.left_child is not None:
            build_graph(custom_node.left_child, graph, parent_id=node_id, 
                        node_id=str(int(node_id) * 2 + 1),
                        feature_labels=feature_labels, leaf_colors=leaf_colors, newline=newline, cost = cost)
        if custom_node.right_child is not None:
            build_graph(custom_node.right_child, graph, parent_id=node_id, 
                        node_id=str(int(node_id) * 2 + 2),
                        feature_labels=feature_labels, leaf_colors=leaf_colors, newline=newline, cost = cost)
    
    return graph

####################################################################################################

def visualize_tree(custom_root, output_file='tree', feature_labels=None, leaf_colors=None,
                   newline=True, cost = True):
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
                        newline=newline, cost=cost)
    graph.attr(size="10,10", dpi="300", ratio="0.75")
    graph.render(output_file, format='png', cleanup=True)
    return Image(output_file + '.png')

####################################################################################################

def plot_decision_set(D, feature_labels, rule_labels, cluster_colors, filename = None):
    fig,ax = plt.subplots(dpi = 300)
    ax.axis('off')
    
    for i,rule in enumerate(D):
        rule_string = ''
        for j, condition in enumerate(rule[:-1]):
            node = condition[0]
            rule_string += '(' + str(feature_labels[node.features[0]])
            if condition[1] == 'left':
                rule_string += r' $\leq$ '
            else:
                rule_string += r' $>$ '
                
            rule_string += str(node.threshold) + ')'
            
            if j < len(rule) - 2:
                if j > 0 and j % 2 == 0:
                    rule_string += r' $\&$ ' + f'\n'
                else:
                    rule_string += r' $\&$ '
        
        text_color = cluster_colors[rule_labels[i]]
        ax.text(s = rule_string, x = 0, y = 2*(len(D) - i)/len(D), color = text_color, alpha = 1,  fontweight = 'extra bold')
        
    if filename is not None:
        plt.savefig(filename, bbox_inches = 'tight', dpi = 300)
    plt.show()

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