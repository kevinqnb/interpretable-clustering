# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

from cython.parallel import prange
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def clustering_distance_cython(list labels1,
                               list labels2,
                               bint percentage=False,
                               set ignore=None) -> float:
    """
    Efficient Cython version of clustering_distance.
    """

    cdef Py_ssize_t n = len(labels1)
    if n != len(labels2):
        raise ValueError("Label arrays must have the same length.")

    # Predeclare variables (Cython doesn't allow cdef inside loops)
    cdef Py_ssize_t i, j
    cdef list non_outliers = []
    cdef double n_pairs
    cdef long differences = 0
    cdef set s1i, s1j, s2i, s2j
    cdef bint same1, same2

    # Build non-outlier index list
    for i in range(n):
        if ignore is None or (labels1[i] != ignore and labels2[i] != ignore):
            non_outliers.append(i)

    n = len(non_outliers)
    if n < 2:
        raise ValueError("Not enough non-outlier points to compute clustering distance.")

    n_pairs = n * (n - 1) / 2.0

    # Main loop
    for i in range(n - 1):
        s1i = labels1[non_outliers[i]]
        s2i = labels2[non_outliers[i]]
        for j in range(i + 1, n):
            s1j = labels1[non_outliers[j]]
            s2j = labels2[non_outliers[j]]

            # Compute overlap (Python-level set ops, but less overhead)
            same1 = bool(s1i & s1j)
            same2 = bool(s2i & s2j)
            if same1 != same2:
                differences += 1

    if percentage:
        return differences / n_pairs
    else:
        return differences

