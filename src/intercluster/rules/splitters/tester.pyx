cimport cython
import numpy as np
cimport numpy as cnp
cnp.import_array()

# Typing
from typing import Tuple, List, Callable, Union
from numpy.typing import NDArray
DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t

def euclidean_distance(
    x : Union[float, int, NDArray],
    y : Union[float, int, NDArray]
) -> float:
    """
    Euclidean Distance between points x and y.

    Args:
        x (np.ndarray): point in euclidean space
        y (np.ndarray): point in euclidean space

    Returns:
        (float): computed distance
    """
    if isinstance(x, float) or isinstance(x, int):
        x = np.array([x])
    if isinstance(y, float) or isinstance(y, int):
        y = np.array([y])

    assert x.shape == y.shape, "x and y must have the same shape."

    return np.linalg.norm(x - y, ord=2)

    
def dtw_distance(
    cnp.ndarray[DTYPE_t, ndim = 1] x,
    cnp.ndarray[DTYPE_t, ndim = 1] y,
    mult_penalty : List[float] = [1.0,1.0,1.0],
    add_penalty : List[float] = [0.0,0.0,0.0],
    distance_fn : Callable = euclidean_distance
) -> Tuple[float, List[Tuple[int]]]:
    """
    Computes the dynamic time warp distance between two sequences x and y.
    The dtw distance works by creating a matching or alignment between the 
    two sequences. The distance or cost of the alignment is computed by summing the
    distances between the aligned elements.

    This implementation allows for the use of different penalties for different types of
    sequence alignment 'moves.' Specifically we consider the following three scenarios 
    when aligning the i-th element of x with the j-th element of y. For each, 
    we can apply a different multiplicative penalty and a different additive penalty to 
    the objective. The dynamic program works by taking the minimum of these three options.

    1. The previous pair in the alignment matched x[i-1] with y[j] (vertical move).
        dtw(i,j) = mult_penalty[0] * distance_fn(x[i], y[j]) + dtw(i-1,j) + add_penalty[0]
    2. The previous pair in the alignment matched x[i] with y[j-1] (horizontal move).
        dtw(i,j) = mult_penalty[1] * distance_fn(x[i], y[j]) + dtw(i,j-1) + add_penalty[1]
    3. The previous pair in the alignment matched x[i] with y[j] (diagonal move).
        dtw(i,j) = mult_penalty[2] * distance_fn(x[i], y[j]) + dtw(i-1,j-1) + add_penalty[2]


    Args:
        x (np.ndarray[float64]): First sequence.
        y (np.ndarray[float64]): Second sequence.
        mult_penalty (List[float]): List of length 3 which describe multiplicative penalties 
            for vertical, horizontal, and diagonal moves respectively.
        add_penalty (List[float]): List of length 3 which describe additive penalties 
            for vertical, horizontal, and diagonal moves respectively.
        distance_fn (function): Function to compute distance between individual elements.

    Returns:
        distance (float): The dtw distance between the two sequences.
        alignment (List[Tuple[int]]): A list of tuples describing the alignment between 
            the two sequences. Each tuple is of the form (i,j) indicating
            that x[i] has been matched with y[j].
    """
    # Initialize variables
    cdef int i, j
    cdef int n = len(x)
    cdef int m = len(y)
    cdef cnp.ndarray[DTYPE_t, ndim = 2] cost_array = np.zeros((n,m), dtype = DTYPE)
    cost_array[:] = np.nan
    cdef int min_move
    cdef DTYPE_t min_cost
    cdef cnp.ndarray[DTYPE_t, ndim=1] costs = np.zeros(3, dtype=DTYPE)
    
    # Compute entries for cost array and track predecessors for alignment
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                cost_array[i,j] = distance_fn(x[i], y[j])
            elif i == 0:
                cost_array[i,j] = (
                    mult_penalty[1] * distance_fn(x[i], y[j]) +
                    cost_array[i,j-1] +
                    add_penalty[1]
                )
            elif j == 0:
                cost_array[i,j] = (
                    mult_penalty[0] * distance_fn(x[i], y[j]) + 
                    cost_array[i-1,j] + 
                    add_penalty[1]
                )
            else:
                costs[0] = (
                    mult_penalty[0] * distance_fn(x[i], y[j]) + cost_array[i-1, j] + add_penalty[0]
                )
                costs[1] = (
                    mult_penalty[1] * distance_fn(x[i], y[j]) + cost_array[i, j-1] + add_penalty[1]
                )
                costs[2] = (
                    mult_penalty[2] * distance_fn(x[i], y[j]) + cost_array[i-1, j-1] + add_penalty[2]
                )
                
                min_move = np.argmin(costs)

                # Always prefer a diagonal move in the event of ties.
                if costs[min_move] == costs[2]:
                    min_move = 2

                cost_array[i,j] = costs[min_move]
                
            
            
    # Backtrack to find the optimal alignment
    i = n - 1 
    j = m - 1
    cdef DTYPE_t current_cost
    alignment = [(i, j)]
    while i > 0 or j > 0:
        current_cost = cost_array[i, j]
        vertical_cost = (
            mult_penalty[0] * distance_fn(x[i], y[j]) + cost_array[i-1, j] + add_penalty[0]
        )
        horizontal_cost = (
            mult_penalty[1] * distance_fn(x[i], y[j]) + cost_array[i, j-1] + add_penalty[1]
        )
        diagonal_cost = (
            mult_penalty[2] * distance_fn(x[i], y[j]) + cost_array[i-1, j-1] + add_penalty[2]
        )
        
        if current_cost == diagonal_cost:
            i -= 1
            j -= 1
            alignment = [(i, j)] + alignment
        elif current_cost == vertical_cost:
            i -= 1
            alignment = [(i, j)] + alignment
        else:
            j -= 1
            alignment = [(i, j)] + alignment
        
    return cost_array[n-1,m-1], alignment