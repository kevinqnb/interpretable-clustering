from tree import *
from rules import *

####################################################################################################
class tree_to_rules:
    """
    Transforms a tree to a list of Rules by traversing the tree and examining 
    the conditions required to get to its leaf nodes. See tree.py or rules.py for
    more info on either of the Tree or Rules classes.
    """
    
    def __init__(self):
        """
        Args:
            rule_list (list): Maintained list of rules from the tree. 
        """
        self.term_list = []
        self.leaf_node_labels = []

    def traverser(self, node, condition_list):
        """
        Recursively traverses the tree from an input node, and 
        builds a rule DNF as it does so.

        Args:
            node (Node): Current node in the tree.
            rule_dnf (list): List of (feature, comparison, threshold) 
                            conjunctive conditions seen so far.
        """
        if node.type == 'leaf':
            term = Term(condition_list)
            self.term_list.append(term)
            self.leaf_node_labels.append(node.label)
        else:
            left_rule = condition_list + [Condition(node.feature, '<=', node.threshold, 
                                                    feature_label = node.feature_label)]
            self.traverser(node.left_child, left_rule)
    
            right_rule = condition_list + [Condition(node.feature, '>', node.threshold, 
                                                     feature_label = node.feature_label)]
            self.traverser(node.right_child, right_rule)
    
    def traverse(self, root_node):
        """
        Traverses the entire tree to build a rule list, starting from the root.

        Args:
            root_node (Node): Root node of some input tree.

        Returns:
            (list): List of Rule objects found at leaf nodes in the tree.
        """
        self.traverser(root_node, [])
        return [Rule([T]) for T in self.term_list]
    
    
####################################################################################################