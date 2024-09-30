import numpy as np

class Grid:
    """
    Builds a g x g grid of rules around a given dataset. 
    """
    
    def __init__(self, g):
        """
        Args:
            X (np.ndarray): Input (n x m) dataset. 
            g (int): grid size
        """
        
        self.g = g
        
    def fit(self, X):
        """
        Fits a grid to an input dataset. 

        Args:
            X (np.ndarray): Input dataset.
        """
        # Step 1: Find the min and max of the dataset along both dimensions
        x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
        y_min, y_max = np.min(X[:, 1]), np.max(X[:, 1])

        # Step 2: Calculate the step size for both dimensions
        x_step = (x_max - x_min) / self.g
        y_step = (y_max - y_min) / self.g

        # Step 3: Create grid cells with logical conditions
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