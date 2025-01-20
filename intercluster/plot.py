import numpy as np
import matplotlib.pyplot as plt
import graphviz as gv
from IPython.display import Image

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
    if custom_node.type == 'internal':
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
        node_label += f"Size: {len(custom_node.indices)}"
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
    if custom_node.type == 'internal':
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

def plot_decision_set(
    D,
    feature_labels,
    rule_labels,
    cluster_colors,
    data_scaler = None,
    filename = None
):
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
                
            if data_scaler is not None:
                feature_min = data_scaler.data_min_[node.features[0]]
                feature_max = data_scaler.data_max_[node.features[0]]
                threshold = node.threshold * (feature_max - feature_min) + feature_min
            else:
                threshold = node.threshold
                
            rule_string += str(np.round(threshold, 3)) + ')'
            
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