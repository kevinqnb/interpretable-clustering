cimport cython
import numpy as np
cimport numpy as cnp
cnp.import_array()
from . import isotonic_error_table
import time
from typing import Tuple

# Typing
from numpy.typing import NDArray
DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t


def dynamic_segment(
    cnp.ndarray[DTYPE_t, ndim = 1] data,
    float penalty = 0.0,
    int penalty_by_length = 0,
    normalize : bool = False,
) -> Tuple[cnp.ndarray[int], cnp.ndarray[int]]:
    """
    Given an input data vector, find the minimum cost segmentation boundaries 
    between fitted isotonic curves. 

    Args:
        data (np.ndarray[float64]): Size n input data array.

        penalty (float): Per-segment penalty. Larger values penalize solutions with more segments.

        penalty_by_length (int): If greater than 0, uses a penalty which is set as the average 
            error among segments of a given legnth. NOTE: If this is greater 0, it will 
            override the normal penalty parameter. Defaults to 0 in which case the 
            standard penalty parameter is used.

        normalize (bool): If True, normalize the error tables so that values fall 
            in the range [0,1]. Default is True.

    Returns:
        inflections (np.ndarray[int64]): Array of indices for which consecutive 
            entries (i, i+1) describe the boundaries of segmentation for the data array. 

        directions (np.ndarray[int64]): Binary array with directional information for 
            each fitted segment, where 1s indicate increasing segments and 0s indicate decreasing.
            For the unimodal model, consecutive segments always alternate between 
            increasing and decreasing isotonic curves. 

            For example, we may want to describe an inflections array [0, 10, 20, 30, 40] in 
            which [0,10) is an increasing segment, [10,20) is decreasing, [20,30) is increasing,
            and [30,40) is decreasing. The directions array would then be [1, 0, 1, 0].
    """
    if not data.ndim == 1:
        raise ValueError("Input data must be a 1d array.")

    if len(data) == 0:
        raise ValueError("Input data cannot be empty.")

    cdef int n = data.shape[0]

    if not penalty_by_length <= n:
        raise ValueError(
            "Penalty by length parameter cannot be greater than the length of the array."
        )

    cdef cnp.ndarray[DTYPE_t, ndim=2] memo_table = np.full((2, n + 1), np.nan, dtype=DTYPE)
    cdef cnp.ndarray[DTYPE_t, ndim=2] inc_error_table, dec_error_table
    inc_error_table, dec_error_table = (
        isotonic_error_table.error_tables(data, normalize = normalize)
    )
    cdef int it
    if penalty_by_length > 0:
        errors = []
        for it in range(n - penalty_by_length):
            errors.append(inc_error_table[it, it + penalty_by_length])
            errors.append(dec_error_table[it, it + penalty_by_length])
        penalty = np.mean(errors)

    # Fill memo table:
    cdef DTYPE_t error, min_error
    cdef int direction,i,j

    # Base case:
    memo_table[0, 0] = 0
    memo_table[1, 0] = 0

    for i in range(1, n + 1):
        for direction in range(2):
            min_error = np.inf
            for j in range(i):
                if direction == 0:
                    error = dec_error_table[j,i] + memo_table[1, j] + penalty
                else:
                    error = inc_error_table[j,i] + memo_table[0, j] + penalty

                if error < min_error:
                    min_error = error

            memo_table[direction, i] = min_error


    # Backtrack:
    cdef int current_direction, current_idx
    cdef DTYPE_t current_error
    current_idx = n
    current_direction = 1 if memo_table[1, current_idx] < memo_table[0, current_idx] else 0
    current_error = memo_table[current_direction, current_idx]

    inflections = [current_idx]
    directions = []
    while current_idx > 0:
        for j in range(current_idx - 1, -1, -1):
            if current_direction == 0:
                est_error = dec_error_table[j,current_idx] + memo_table[1, j] + penalty
            else:
                est_error = inc_error_table[j,current_idx] + memo_table[0, j] + penalty

            if est_error == current_error:
                inflections = [j] + inflections
                directions = [current_direction] + directions
                current_idx = j
                current_direction = 1 - current_direction
                current_error = memo_table[current_direction, current_idx]
                break 

    return np.array(inflections), np.array(directions)