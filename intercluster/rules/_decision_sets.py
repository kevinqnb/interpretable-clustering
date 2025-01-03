from typing import List, Dict, Any, Tuple
from numpy.typing import NDArray
from ._tree import Tree
from ..utils import *
from .prune import *

class DecisionSet:
    """
    Base class for a decision set.
    """
    def __init__(
        self,
        #norm : int = 2,
        #center_init : str = None,
        #centers : NDArray = None,
        random_seed : int = None,
        feature_labels : List[str] = None
    ):
        
        """
        Args:   
            norm (int, optional): Takes values 1 or 2. If norm = 1, compute distances using the 
                1 norm. If 2, compute distances with the squared two norm. Defaults to 2.
                
            center_init (str, optional): Center initialization method. Included options are 
                'k-means' which runs a k-means algorithm and uses the output centers,
                'random++' which uses a randomized k-means++ initialization, or 
                'manual' which assumes an input array of centers (in the next parameter). 
                Defaults to None in which case no centers are initialized.
                
            centers (np.ndarray, optional): Input list of reference centers to calculate cost with. 
                Defaults to None.
                
            random_seed (int, optional): Random seed. In the Tree object randomness is only ever 
                used for breaking ties between nodes, or if you are using a RandomTree!
                
            feature_labels (List[str]): Iterable object with strings representing feature names. 
        """
        '''
        if norm in [1,2]:
            self.norm = norm
        else:
            raise ValueError('Norm must either be 1 or 2.')
        
        if center_init in [None, 'k-means', 'random++', 'manual']:
            self.center_init = center_init
        else:
            raise ValueError('Unsupported initialization method.')
        
        
        if self.center_init == 'manual' and (centers is not None):            
            self.centers = copy.deepcopy(centers)
            self.n_centers = len(self.centers)
            
        elif self.center_init == 'manual':
            raise ValueError('Must provide an input array of centers for manual initialization.')
        
        else:
            self.centers = centers
            self.n_centers = None
            
        self.feature_labels = feature_labels
        '''
        
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
            
        self.feature_labels = feature_labels
        
        self.all_rules = None
        self.decision_set = None
        
        
    def _fitting(self, X : NDArray) -> List[Any]:
        """
        Privately used, custom fitting function.
        Fits a decision set to an input dataset. 
        
        Args:
            X (np.ndarray): Input dataset.
            
        returns:
            decision_set (List[Any]): List of rules.
        """
        raise NotImplementedError('Method not implemented.')
        
        
    def fit(self, X : NDArray):
        """
        Public fit function. 
        Fits a decision set to an input dataset. 
        
        Args:
            X (np.ndarray): Input dataset.
        """
        self.decision_set = self._fitting(X)
        self.covers = self.get_covers(X)
        #self.costs = self.get_costs(X)
        
        
    def get_covers(self, X : NDArray) -> Dict[int, List[int]]:
        """
        Finds data points of X covered by each rule in the decision set.
        
        Args:
            X (np.ndarray): Input dataset.
            
        Returns:
            covers (dict[rule, List[int]]): Dictionary with rules indices 
                as keys and list of data point indices as values.
        """
        raise NotImplementedError('Method not implemented.')
    
    '''
    def get_costs(self, X : NDArray) -> Dict[int, float]:
        """
        Finds the cost associated with each rule in the decision set.
        
        Args:
            X (np.ndarray): Input dataset.
            
        Returns:
            covers (dict[rule, float]): Dictionary with rule indices as keys and costs as values
        """
        raise NotImplementedError('Method not implemented.')
    '''
    
    
    def predict(self, X : NDArray) -> List[List[int]]:
        """
        Predicts the label(s) of each data point in X.
        
        Args:
            X (np.ndarray): Input dataset.
            
        Returns:
            labels (List[List[int]]): 2d list of predicted labels, with the internal list 
                at index i representing the group of decision rules which satisfy X[i,:].
        """
        
        set_covers = self.get_covers(X)
        
        labels = [[] for _ in range(len(X))]
        
        for i,r_covers in set_covers.items():
            for j in r_covers:
                labels[j].append(i)
        
        return labels