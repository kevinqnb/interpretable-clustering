import numpy as np
import itertools

class Grid:
    """
    Builds a g x g grid of rules around a given dataset. 
    """
    
    def __init__(self, g, features):
        """
        Args:
            X (np.ndarray): Input (n x m) dataset. 
            g (int): grid size
        """
        
        self.g = g
        self.features = features
        
        dtuples = list(itertools.product(list(range(self.g+2)), repeat = len(features)))
        self.label_dict = {tup:i for i,tup in enumerate(dtuples)}
        
        self.hist = None
        self.edges = None
        
    def fit(self, X):
        """
        Fits a grid to an input dataset. 

        Args:
            X (np.ndarray): Input dataset.
        """
        X_= X[:, self.features]
                
        self.hist, self.edges = np.histogramdd(X_, bins=self.g)
            
            
    def predict(self, X):
        X_= X[:, self.features]
        
        binned = np.zeros(X_.shape, dtype = int)
        for d in range(X_.shape[1]):
            binned[:, d] = np.digitize(X_[:, d], bins=self.edges[d])
            #binned[:, d] = np.clip(binned[:, d], 0, self.g + 1)
            
        labels = np.array([self.label_dict[tuple(binned[i, :])] for i in range(len(binned))])
        return labels