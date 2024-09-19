import numpy as np 
import copy

####################################################################################################

class Condition:
    """
    Logical axis aligned formula with boolean evaluations. Designed to be 
    tested on data points, its typically of the form: 
    feature <= value, where feature is the name or index of a feature variable, <= is the 
    comparison operator being used, and value is some threshold value. 
    
    Supported operators include '==', '!=', '<', '>', '<=', or '>='. 
    """
    def __init__(self, feature, operator, value, feature_label = None):
        """
        Args:
            feature (int): Feature index.
            comparison (str): comparison operator.
            value (_type_): Threshold value.
            feature_label (str): Name for the given feature (for printing and display). 
        """
        self.feature = feature
        self.value = value
        
        if operator not in ['==', '!=', '<', '>', '<=', '>=']:
            raise ValueError('Invalid operator')
        
        self.operator = operator
        self.feature_label = feature_label
        
    
    def __repr__(self):
        """
        Returns:
            str: String representation of the Condition.
        """
        if self.feature_label is None:
            return f'x{self.feature} {self.operator} {self.value}'
        else:
            return f'{self.feature_label} {self.operator} {self.value}'
    
    
    def __eq__(self, other):
        """
        Tests condition equality.

        Args:
            other (Condition): Other Condition object to test with

        Returns:
            bool: Equality evaluation. 
        """
        if (self.feature == other.feature and
            self.operator == other.operator and
            self.value == other.value):
            return True
        else:
            return False
        
    def boundary(self, other):
        """
        Tests if another condition is at the boundary of the current.
        For example, conditions x <= 5 and x > 5 are boundary conditions.
        
        Args:
            other (Condition): Other Condition object to test with

        Returns:
            bool: Boundary evaluation. 
        """
        if (self.feature == other.feature and
            self.value == other.value):
            
            if self.operator == '>=' and other.operator == '<':
                return True
            elif self.operator == '<=' and other.operator == '>':
                return True
            elif self.operator == '>' and other.operator == '<=':
                return True
            elif self.operator == '<' and other.operator == '>=':
                return True
            
        else:
            return False
        
    
    def evaluate(self, data_point):
        """
        Evaluates the condition on a single input data point.

        Args:
            data_point (np.ndarray): Data point.

        Returns:
            bool: Boolean evaluation.
        """
        
        if self.operator == '==':
            return data_point[self.feature] == self.value
        elif self.operator == '!=':
            return data_point[self.feature] != self.value
        elif self.operator == '<':
            return data_point[self.feature] < self.value
        elif self.operator == '>':
            return data_point[self.feature] > self.value
        elif self.operator == '<=':
            return data_point[self.feature] <= self.value
        elif self.operator == '>=':
            return data_point[self.feature] >= self.value
        

####################################################################################################

class Term:
    """
    Initializes a Term as a conjunction of Conditions.
    """
    def __init__(self, condition_list):
        """
        Args:
            condition_list (List[Condition]): List of Condition objects.
            
        Attributes:
            q (int): Number of conditions in the conjunction.
            satisfied_points (np.ndarray): Subset array of data points which satisfy the term.
            satisfied_indices (List[int]): List of indices corresponding to satisfied points 
                                            in a dataset.  
        """
        self.condition_list = condition_list
        self.q = len(condition_list)
        self.satisfied_points = None
        self.satisfied_indices = None
        
    
    def __repr__(self):
        """
        Returns:
            str: String representation of the Term.
        """
        return '(' + '  ∧  '.join([repr(cond) for cond in self.condition_list]) + ')'
    
    def __eq__(self, other):
        """
        Tests term equality.

        Args:
            other (Term): Other Term object to test with

        Returns:
            bool: Equality evaluation. 
        """
        for cond1 in self.condition_list:
            present = False
            for cond2 in other.condition_list:
                if cond1 == cond2:
                    present = True
                    
            if present == False:
                return False
            
        return True
    
    
    def get_boundary(self, other):
        """
        Finds a feature/value that two Terms share a 'boundary' on (if any). 
        For example two terms (x < 5 ∧ y < 10) and (x >= 5 ∧ y < 10) share 
        a boundary at x = 5.

        Args:
            other (Term): Term object to compare with.
            
        Returns:
            feature, value (int, float): Feature, value pair for the boundary.
                                        If no boundary exists, return None.
        """
        boundary_pair = None
        for cond1 in self.condition_list:
            found = False
            for cond2 in other.condition_list:
                if cond1.boundary(cond2):
                    found = True
                    break
                    
            if found:
                boundary_pair = (cond1.feature, cond1.value)
                break
            
        return boundary_pair
    
    
    def boundary(self, other):
        """
        Determines if two Terms are 'boundary' terms meaning 
        that they 1) share boundary conditions on exactly one feature
        and 2) share equal conditions on every other feature.

        Args:
            other (Term): Term object to compare with.
        """
    
        # 1) Determine if a boundary exists:
        boundary_pair = self.get_boundary(other)
        
        if boundary_pair is None:
            return False
        else:
            boundary_feature, boundary_value = boundary_pair
        
        # 2) Determine if they share equal conditions elsewhere:
        is_boundary = True
        for cond1 in self.condition_list:
            if cond1.feature != boundary_feature:
                found = False
                for cond2 in other.condition_list:
                    if cond1 == cond2:
                        found = True
                        break
                        
                if not found:
                    is_boundary = False
                    break
                
        for cond2 in other.condition_list:
            if cond2.feature != boundary_feature:
                found = False
                for cond1 in self.condition_list:
                    if cond2 == cond1:
                        found = True
                        break
                        
                if not found:
                    is_boundary = False
                    break
                
        return is_boundary
    
    def simplify(self):
        """
        Removes any redundant conditions from the condition list. The idea is that if there 
        are two conditions such as, x <= 10 and x <= 5, then the former condition is redundant. 
        """
        new_condition_list = []
        for i, cond1 in enumerate(self.condition_list):
            unique = True
            for j, cond2 in enumerate(self.condition_list):
                if j != i:
                    if cond1.feature == cond2.feature and cond1.operator == cond2.operator:
                        if cond2.value == cond1.value and j in new_condition_list:
                            unique = False
                        elif cond1.operator == '<' and cond2.value < cond1.value:
                            unique = False
                        elif cond1.operator == '<=' and cond2.value < cond1.value:
                            unique = False
                        elif cond1.operator == '>' and cond2.value > cond1.value:
                            unique = False
                        elif cond1.operator == '>=' and cond2.value > cond1.value:
                            unique = False  
                        
            if unique:
                new_condition_list.append(i)
                
        self.condition_list = [self.condition_list[i] for i in new_condition_list]
            
        

    def evaluate(self, data_point):
        """
        Evaluate the conjunction of conditions on a data point.
        
        Args:
            data_point (np.ndarray): A numpy array representing a single data point.
        
        Returns:
            bool: Evaluation of the conjunction.
        """
        return all(cond.evaluate(data_point) for cond in self.condition_list)
        


####################################################################################################

class Rule:
    """
    Initializes a Rule as a disjunction of conjunctive Terms (DNF).
    """
    def __init__(self, term_list):
        """
        Args: 
            term_list (List[Term]): List of Term objects.
            
        Attributes:
            r (int): Number of terms in the disjunction.
            satisfied_points (np.ndarray): Subset array of data points which satisfy the term.
            satisfied_indices (List[int]): List of indices corresponding to satisfied points 
                                            in a dataset.  
        """
        self.term_list = term_list
        self.r = len(term_list)
        self.satisfied_points = None
        self.satisfied_indices = None
        
        
    def __repr__(self):
        """
        Returns:
            str: String representation of the Term.
        """
        return '  ∨  \n'.join([repr(T) for T in self.term_list])
        
    def simplify(self):
        """
        Simplifies the Rule by merging terms which share boundaries.
        """
        # Simplify conditions in the term list:
        for ter in self.term_list:
            ter.simplify()
            
        # Greedily merge terms until none can be merged anymore:
        new_term_list = copy.copy(self.term_list)
        full_pass = False
        while not full_pass:
            count = 0
            for i, ter1 in enumerate(new_term_list):
                found_merge = False
                for j, ter2 in enumerate(new_term_list):
                    if j != i:
                        if ter1.boundary(ter2):
                            found_merge = True
                            boundary_feature, boundary_value = ter1.get_boundary(ter2)
                            
                            conds1 = [c for c in ter1.condition_list if 
                                    (c.feature != boundary_feature or c.value != boundary_value)]
                            conds2 = [c for c in ter2.condition_list if 
                                    (c.feature != boundary_feature or c.value != boundary_value)]
                            
                            new_term = Term(conds1 + conds2)
                            new_term.simplify()
                            
                            del new_term_list[max(i,j)]
                            del new_term_list[min(i,j)]
                            new_term_list += [new_term]
                            
                            break
                    
                if found_merge:
                    break
                else:
                    count += 1
                    
            if count == len(new_term_list):
                full_pass = True
                
        self.term_list = new_term_list
                        
                        
        
            

    def evaluate(self, data_point):
        """
        Test if a data point satisfies the rule.
        
        Args:
            data_point (np.ndarray): A numpy array representing the single data point.
        
        Returns:
            bool: Evaluation of the rule.
        """
        return any(T.evaluate(data_point) for T in self.term_list)
    
    
    def find_satisfied_indices(self,X):
        """
        Find the indices of all points from X which satisfy this rule.

        Args:
            X (np.ndarray): Dataset.

        Returns:
            List[int]: List of indices representing points which satisfy the rule.
        """
        satisfies = []
        for i in range(len(X)):
            if self.evaluate(X[i,:]):
                satisfies.append(i)
                
        return satisfies

    def fit(self, X):
        """
        Update the satisfied data points with dataset X.
        
        Args:
            X (np.ndarray): A numpy array representing a dataset.
        """
        satisfies = self.find_satisfied_indices(X)
        self.satisfied_points = X[satisfies,:]
        self.satisfied_indices = satisfies
        


####################################################################################################
    
