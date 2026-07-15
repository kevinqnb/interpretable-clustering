import numpy as np
import pytest
from intercluster.kinetic_heap import KineticHeap, KineticNode


####################################################################################################
# Test helpers:
####################################################################################################


def key_wins(i, j, t):
    """
    Restates the heap's key comparison independently of the implementation: `True` if leaf i's item
    has a strictly larger key than leaf j's item at time t, ties therefore going to j.
    """
    if t == -np.inf:
        return (-i.a, i.b) > (-j.a, j.b)
    if t == np.inf:
        return (i.a, i.b) > (j.a, j.b)
    return i.a * t + i.b > j.a * t + j.b


def collect_leaves(x):
    """
    Returns the leaves of the subtree rooted at x, in left to right order.
    """
    if x is None:
        return []
    if x.type == 'leaf':
        return [x]
    return collect_leaves(x.left) + collect_leaves(x.right)


def check_node(x, parent, t_c):
    """
    Verifies, bottom up, that every cached field of the subtree rooted at x is exactly equal to the
    value recomputed from scratch at the current time t_c, and that the AVL balance condition holds.

    Exact equality is demanded rather than approximate: the cached fields of a subtree that was
    pruned during an advance of the current time are stale, but they should nonetheless be bit for
    bit what a recomputation would produce, since a subtree with no swap in the elapsed interval has
    the same winner afterwards as before.
    """
    assert x.parent is parent, "Parent pointer does not agree with the tree structure."

    if x.type == 'leaf':
        assert x.left is None and x.right is None, "A leaf must not have children."
        assert x.height == 0, "A leaf must have height zero."
        assert x.winner is x, "A leaf must be its own winner."
        assert x.max_leaf is x, "A leaf must be its own rightmost leaf."
        assert x.predecessor is None, "A leaf has no predecessor."
        assert x.swap_time == -np.inf, "A leaf holds no swap."
        assert x.min_future_swap == np.inf, "A leaf holds no future swap."
        return

    assert x.type == 'internal', "A node must be either internal or a leaf."
    assert x.left is not None and x.right is not None, "An internal node must have two children."
    assert x.item is None, "An internal node must not hold an item."

    check_node(x.left, x, t_c)
    check_node(x.right, x, t_c)
    y = x.left
    z = x.right

    assert x.height == 1 + max(y.height, z.height), "Cached height is stale."
    assert abs(y.height - z.height) <= 1, "AVL balance condition is violated."
    assert x.max_leaf is z.max_leaf, "Cached rightmost leaf is stale."
    assert x.predecessor is y.max_leaf, "Cached predecessor is stale."

    i = y.winner
    j = z.winner
    assert x.winner is (i if key_wins(i, j, t_c) else j), "Cached winner is stale."

    if i.a != j.a:
        swap_time = (i.b - j.b) / (j.a - i.a)
    else:
        swap_time = -np.inf
    assert x.swap_time == swap_time, "Cached swap time is stale."

    own_swap = swap_time if swap_time > t_c else np.inf
    f = min(own_swap, y.min_future_swap, z.min_future_swap)
    assert x.min_future_swap == f, "Cached minimum future swap time is stale."
    assert x.min_future_swap > t_c, "A pending swap time must lie strictly in the future."


def check_invariants(heap):
    """
    Verifies every structural invariant of the heap: the leaves are strictly ordered, the item to
    leaf map is a bijection onto the leaves, the tree is a balanced AVL tree, and every cached field
    equals its recomputation from scratch at the current time.
    """
    leaves = collect_leaves(heap.root)

    # The leaf ordering is a strict total order, so no two leaves may compare equal.
    orders = [(leaf.a, leaf.b, leaf.seq) for leaf in leaves]
    for prev, curr in zip(orders, orders[1:]):
        assert prev < curr, "Leaves are not in strictly increasing order."

    # The item to leaf map is a bijection onto the leaves of the tree.
    assert len(leaves) == len(heap.leaves), "Item map and tree hold different numbers of items."
    assert len(heap) == len(leaves), "Reported length does not match the number of leaves."
    for leaf in leaves:
        assert heap.leaves[leaf.item] is leaf, "Item map does not point at the item's own leaf."

    if heap.root is not None:
        assert heap.root.parent is None, "The root must not have a parent."
        check_node(heap.root, None, heap.current_time)


def oracle_find_max(items, t):
    """
    Brute force reference for `find_max`. Takes a plain dict of item -> (a, b, seq) and computes the
    item of maximum key at time t directly, breaking ties toward the largest (a, b, seq), which is
    the heap's documented rule of returning the item rightmost in the leaf ordering.
    """
    if len(items) == 0:
        return None
    keys = {item : a * t + b for item, (a, b, seq) in items.items()}
    best = max(keys.values())
    tied = [item for item in items if keys[item] == best]
    return max(tied, key = lambda item : items[item])


####################################################################################################
# Basic behavior:
####################################################################################################


class TestKineticHeapBasics:
    """Test construction, size reporting, and the empty heap."""

    def test_make_heap_is_empty(self):
        """Test that a new heap is empty and starts at a current time of negative infinity."""
        heap = KineticHeap.make_heap()
        assert heap.is_empty()
        assert len(heap) == 0
        assert heap.current_time == -np.inf
        check_invariants(heap)

    def test_empty_heap_find_max_returns_none(self):
        """Test that find_max on an empty heap returns None rather than raising."""
        heap = KineticHeap()
        assert heap.find_max(0.0) is None
        assert heap.delete_max(1.0) is None
        assert heap.current_time == 1.0

    def test_single_item(self):
        """Test that a heap of one item always returns that item."""
        heap = KineticHeap()
        heap.insert('a', 2.0, -1.0)
        check_invariants(heap)
        assert len(heap) == 1
        assert 'a' in heap
        for t in [-10.0, 0.0, 0.5, 100.0]:
            assert heap.find_max(t) == 'a'
            check_invariants(heap)

    def test_delete_last_item_empties_heap(self):
        """Test that deleting the only item leaves a valid empty heap."""
        heap = KineticHeap()
        heap.insert('a', 1.0, 1.0)
        heap.delete('a')
        assert heap.is_empty()
        assert 'a' not in heap
        assert heap.find_max(0.0) is None
        check_invariants(heap)

    def test_delete_max_returns_and_removes(self):
        """Test that delete_max returns the maximum and removes it from the heap."""
        heap = KineticHeap()
        heap.insert('flat', 0.0, 10.0)
        heap.insert('steep', 1.0, 0.0)
        assert heap.delete_max(0.0) == 'flat'
        check_invariants(heap)
        assert 'flat' not in heap
        assert heap.find_max(0.0) == 'steep'
        assert heap.delete_max(0.0) == 'steep'
        assert heap.is_empty()

    def test_duplicate_item_raises_error(self):
        """Test that inserting an item already in the heap raises ValueError."""
        heap = KineticHeap()
        heap.insert('a', 1.0, 0.0)
        with pytest.raises(ValueError, match="already in the heap"):
            heap.insert('a', 2.0, 3.0)

    def test_delete_missing_item_raises_error(self):
        """Test that deleting an item not in the heap raises ValueError."""
        heap = KineticHeap()
        heap.insert('a', 1.0, 0.0)
        with pytest.raises(ValueError, match="not in the heap"):
            heap.delete('b')

    def test_reinsertion_after_deletion(self):
        """Test that an item may be reinserted with a new key after being deleted."""
        heap = KineticHeap()
        heap.insert('a', 1.0, 0.0)
        heap.insert('b', 2.0, 0.0)
        heap.delete('a')
        heap.insert('a', 5.0, 0.0)
        check_invariants(heap)
        assert heap.find_max(1.0) == 'a'


####################################################################################################
# peek_max:
####################################################################################################


class TestPeekMax:
    """Test the read-only peek_max operation."""

    def test_peek_max_empty_heap_returns_none(self):
        """Test that peek_max on an empty heap returns None rather than raising."""
        heap = KineticHeap()
        assert heap.peek_max() is None

    def test_peek_max_item_matches_find_max_before_any_query(self):
        """Test that peek_max's item agrees with find_max at the initial current_time of -inf.

        The key value itself is not checked here: at t=-inf, a*t+b is -inf for any a != 0,
        so the raw value is degenerate (uninformative) even though the winning item, resolved
        via the heap's -inf tie-break convention, is well defined.
        """
        heap = KineticHeap()
        heap.insert('a', 2.0, -1.0)
        heap.insert('b', 1.0, 5.0)
        item, _ = heap.peek_max()
        assert item == heap.find_max(-np.inf)

    def test_peek_max_matches_find_max_after_query(self):
        """Test that peek_max reports the same item and value as the preceding find_max."""
        heap = KineticHeap()
        heap.insert('a', 2.0, -1.0)
        heap.insert('b', 1.0, 5.0)
        heap.insert('c', -1.0, 10.0)
        winner = heap.find_max(3.0)
        item, value = heap.peek_max()
        assert item == winner
        assert value == item_a_times_t_plus_b(heap, item, 3.0)

    def test_peek_max_after_delete_max(self):
        """Test that peek_max reflects the new winner after delete_max removes the old one."""
        heap = KineticHeap()
        heap.insert('flat', 0.0, 10.0)
        heap.insert('steep', 1.0, 0.0)
        assert heap.delete_max(0.0) == 'flat'
        item, value = heap.peek_max()
        assert item == 'steep'
        assert value == pytest.approx(1.0 * 0.0 + 0.0)

    def test_peek_max_does_not_mutate_heap(self):
        """Test that peek_max leaves current_time, length, and structural invariants untouched."""
        heap = KineticHeap()
        heap.insert('a', 2.0, -1.0)
        heap.insert('b', 1.0, 5.0)
        heap.insert('c', -1.0, 10.0)
        heap.find_max(3.0)

        before_time = heap.current_time
        before_len = len(heap)
        heap.peek_max()
        heap.peek_max()

        assert heap.current_time == before_time
        assert len(heap) == before_len
        assert 'a' in heap and 'b' in heap and 'c' in heap
        check_invariants(heap)


def item_a_times_t_plus_b(heap, item, t):
    """Independently recomputes an item's key value at time t from the item to leaf map."""
    leaf = heap.leaves[item]
    return leaf.a * t + leaf.b


####################################################################################################
# Query time monotonicity:
####################################################################################################


class TestMonotonicity:
    """Test that query times are required to be nondecreasing."""

    def test_decreasing_query_time_raises_error(self):
        """Test that a query preceding the current time raises ValueError."""
        heap = KineticHeap()
        heap.insert('a', 1.0, 0.0)
        heap.find_max(5.0)
        with pytest.raises(ValueError, match="nondecreasing"):
            heap.find_max(4.999)

    def test_decreasing_query_time_raises_on_empty_heap(self):
        """Test that monotonicity is enforced even when the heap holds no items."""
        heap = KineticHeap()
        heap.find_max(3.0)
        with pytest.raises(ValueError, match="nondecreasing"):
            heap.find_max(2.0)

    def test_repeated_query_time_allowed(self):
        """Test that querying the same time twice is permitted and stable."""
        heap = KineticHeap()
        heap.insert('a', 1.0, 0.0)
        heap.insert('b', -1.0, 0.0)
        assert heap.find_max(2.0) == 'a'
        assert heap.find_max(2.0) == 'a'
        check_invariants(heap)

    def test_delete_max_does_not_rewind_time(self):
        """Test that a delete_max advances the current time like any other query."""
        heap = KineticHeap()
        heap.insert('a', 1.0, 0.0)
        heap.insert('b', 2.0, 0.0)
        heap.delete_max(7.0)
        assert heap.current_time == 7.0
        with pytest.raises(ValueError, match="nondecreasing"):
            heap.find_max(6.0)


####################################################################################################
# Swap correctness:
####################################################################################################


class TestSwapCorrectness:
    """Test that winners switch at exactly the time their key lines cross."""

    def test_winner_flips_at_crossing_time(self):
        """
        Test two lines crossing at a known time. The flat line leads strictly before the crossing,
        and the steeper line leads at and after it, the swap being treated as having fired once the
        current time reaches it.
        """
        # flat(t) = 0, steep(t) = t - 1. These cross at t = 1.
        heap = KineticHeap()
        heap.insert('flat', 0.0, 0.0)
        heap.insert('steep', 1.0, -1.0)
        check_invariants(heap)

        assert heap.find_max(0.0) == 'flat'
        assert heap.find_max(0.999) == 'flat'
        assert heap.find_max(1.0) == 'steep'
        check_invariants(heap)
        assert heap.find_max(1.001) == 'steep'
        assert heap.find_max(50.0) == 'steep'
        check_invariants(heap)

    def test_winner_flips_in_max_heap_direction(self):
        """Test that it is the steeper line, not the flatter one, that wins after the crossing."""
        heap = KineticHeap()
        heap.insert('shallow', -2.0, 10.0)
        heap.insert('steep', 3.0, -5.0)
        # -2t + 10 = 3t - 5  =>  t = 3.
        assert heap.find_max(2.9) == 'shallow'
        assert heap.find_max(3.0) == 'steep'
        assert heap.find_max(3.1) == 'steep'

    def test_winner_never_switches_back(self):
        """Test that once the steeper line takes the lead it keeps it forever."""
        heap = KineticHeap()
        heap.insert('flat', 0.0, 0.0)
        heap.insert('steep', 1.0, -1.0)
        assert heap.find_max(1.0) == 'steep'
        for t in [1.0, 2.0, 10.0, 1e6, 1e12]:
            assert heap.find_max(t) == 'steep'
            check_invariants(heap)

    def test_crossing_skipped_over_in_one_advance(self):
        """Test that a swap is applied even when the current time jumps straight past it."""
        heap = KineticHeap()
        heap.insert('flat', 0.0, 0.0)
        heap.insert('steep', 1.0, -1.0)
        # Never query near the crossing at t = 1; jump from before it to well past it.
        assert heap.find_max(-5.0) == 'flat'
        assert heap.find_max(100.0) == 'steep'
        check_invariants(heap)

    def test_many_crossings_in_one_advance(self):
        """Test a fan of lines through a common point, all of which swap in a single advance."""
        heap = KineticHeap()
        # Every line passes through (2, 4), so all of them cross one another at t = 2.
        for n in range(8):
            slope = float(n - 4)
            heap.insert(n, slope, 4.0 - 2.0 * slope)
        check_invariants(heap)

        # Before the common crossing the shallowest line leads; after it, the steepest.
        assert heap.find_max(0.0) == 0
        check_invariants(heap)
        assert heap.find_max(10.0) == 7
        check_invariants(heap)


####################################################################################################
# Asymptotics and degenerate keys:
####################################################################################################


class TestAsymptotics:
    """Test the behavior of the heap at extreme times."""

    def test_large_time_returns_largest_coefficient(self):
        """
        Test that for large times the item of largest key coefficient wins. This is what the leaf
        ordering exists to arrange, and would fail were the ordering left in its min-heap direction.
        """
        heap = KineticHeap()
        heap.insert('steepest', 5.0, -1000.0)
        heap.insert('middle', 1.0, 0.0)
        heap.insert('flattest', -3.0, 1000.0)
        check_invariants(heap)

        assert heap.find_max(-1000.0) == 'flattest'
        assert heap.find_max(1e9) == 'steepest'
        check_invariants(heap)

    def test_large_time_ties_broken_by_largest_constant(self):
        """Test that among items of equal largest coefficient, the largest constant wins."""
        heap = KineticHeap()
        heap.insert('low', 2.0, 0.0)
        heap.insert('high', 2.0, 1.0)
        heap.insert('flat', 0.0, 500.0)
        assert heap.find_max(1e6) == 'high'
        check_invariants(heap)

    def test_initial_current_time_is_negative_infinity(self):
        """
        Test that before any query the winner is the item that leads as time tends to negative
        infinity, namely the one of smallest coefficient, ties going to the largest constant.
        """
        heap = KineticHeap()
        heap.insert('a', 1.0, 0.0)
        heap.insert('b', -1.0, 0.0)
        heap.insert('c', -1.0, 5.0)
        check_invariants(heap)
        # The cached winner at the initial time of -inf, read without advancing the clock.
        assert heap.root.winner.item == 'c'


class TestDegenerateKeys:
    """Test keys that are parallel, constant, or otherwise degenerate."""

    def test_all_coefficients_equal(self):
        """Test parallel lines, which never cross and so never swap."""
        heap = KineticHeap()
        for n in range(6):
            heap.insert(n, 2.0, float(n))
        check_invariants(heap)

        # Parallel lines never cross, so no internal node holds a future swap.
        assert heap.root.min_future_swap == np.inf
        for t in [-100.0, 0.0, 100.0]:
            assert heap.find_max(t) == 5
            check_invariants(heap)

    def test_all_constants_equal(self):
        """Test a pencil of lines through a common intercept."""
        heap = KineticHeap()
        for n in range(6):
            heap.insert(n, float(n) - 3.0, 4.0)
        check_invariants(heap)

        # All lines cross at t = 0, so the smallest coefficient leads before it and the largest
        # leads at and after it.
        assert heap.find_max(-1.0) == 0
        assert heap.find_max(0.0) == 5
        assert heap.find_max(1.0) == 5
        check_invariants(heap)

    def test_all_keys_constant(self):
        """Test items whose keys do not vary with time at all."""
        heap = KineticHeap()
        heap.insert('a', 0.0, 3.0)
        heap.insert('b', 0.0, 7.0)
        heap.insert('c', 0.0, 5.0)
        check_invariants(heap)
        for t in [-50.0, 0.0, 50.0]:
            assert heap.find_max(t) == 'b'
        check_invariants(heap)

    def test_zero_coefficient_at_initial_time(self):
        """
        Test that an item of zero coefficient is handled at the initial current time of -inf, where
        evaluating its key as a * t + b would produce a nan.
        """
        heap = KineticHeap()
        heap.insert('zero', 0.0, 1.0)
        heap.insert('positive', 1.0, 1.0)
        check_invariants(heap)
        assert heap.root.winner.item == 'zero'

        # The two lines share an intercept and so cross at exactly t = 0, where the tie is resolved
        # in favor of the steeper of the two.
        assert heap.find_max(-1.0) == 'zero'
        assert heap.find_max(0.0) == 'positive'
        assert heap.find_max(1.0) == 'positive'
        check_invariants(heap)


####################################################################################################
# Tie breaking:
####################################################################################################


class TestTieBreaking:
    """Test items whose keys are identical or momentarily tied."""

    def test_identical_keys_are_all_retained(self):
        """Test that items with identical keys remain a valid tree and none are lost."""
        heap = KineticHeap()
        for n in range(10):
            heap.insert(n, 1.0, 2.0)
            check_invariants(heap)

        assert len(heap) == 10
        # Ties go to the item rightmost in the leaf ordering, i.e. the latest inserted.
        assert heap.find_max(0.0) == 9

        # Every item is still reachable and deletable.
        for n in range(10):
            assert n in heap
        for n in range(10):
            heap.delete(n)
            check_invariants(heap)
        assert heap.is_empty()

    def test_ties_broken_by_largest_coefficient(self):
        """Test that among items momentarily tied, the largest coefficient is returned."""
        heap = KineticHeap()
        # At t = 0 all three keys equal 5, but the coefficients differ.
        heap.insert('flat', 0.0, 5.0)
        heap.insert('steep', 4.0, 5.0)
        heap.insert('middle', 2.0, 5.0)
        assert heap.find_max(0.0) == 'steep'
        check_invariants(heap)

    def test_ties_broken_by_largest_constant_then_latest_insertion(self):
        """Test the second and third components of the tie breaking rule."""
        heap = KineticHeap()
        heap.insert('first', 1.0, 0.0)
        heap.insert('second', 1.0, 0.0)
        # Identical keys: the later insertion sits further right and so wins.
        assert heap.find_max(3.0) == 'second'

        # A larger constant outranks insertion order.
        heap.insert('larger', 1.0, 0.1)
        assert heap.find_max(3.0) == 'larger'
        check_invariants(heap)


####################################################################################################
# Numerical edge cases:
####################################################################################################


class TestNumericalEdgeCases:
    """Test near degenerate arithmetic in the swap times."""

    def test_near_equal_coefficients(self):
        """
        Test coefficients differing by a very small amount. The crossing is real but enormously far
        in the future, and must simply be pruned rather than cause a failure.
        """
        heap = KineticHeap()
        heap.insert('a', 1.0, 1.0)
        heap.insert('b', 1.0 + 1e-12, 0.0)
        check_invariants(heap)

        # The lines cross at t = 1e12, so before that the larger constant leads.
        assert heap.find_max(0.0) == 'a'
        assert heap.find_max(1e6) == 'a'
        check_invariants(heap)
        assert heap.find_max(1e13) == 'b'
        check_invariants(heap)

    def test_overflowing_swap_time(self):
        """Test that a swap time too large to represent is treated as no swap at all."""
        heap = KineticHeap()
        heap.insert('a', 1.0, 1e308)
        heap.insert('b', 1.0 + 1e-10, -1e308)
        check_invariants(heap)
        # The difference of the constants overflows, so the crossing is infinitely far away and is
        # never reached and never applied.
        assert heap.root.swap_time == np.inf
        assert heap.root.min_future_swap == np.inf
        assert heap.find_max(1e300) == 'a'
        check_invariants(heap)

    def test_query_exactly_at_swap_time_repeatedly(self):
        """Test that querying repeatedly at exactly a swap time is stable and consumes the swap."""
        heap = KineticHeap()
        heap.insert('flat', 0.0, 0.0)
        heap.insert('steep', 1.0, -1.0)

        assert heap.find_max(1.0) == 'steep'
        # The swap has now fired, so it is no longer pending.
        assert heap.root.min_future_swap == np.inf
        assert heap.find_max(1.0) == 'steep'
        check_invariants(heap)
        assert heap.find_max(1.0 + 1e-15) == 'steep'
        check_invariants(heap)

    def test_swap_time_coincident_with_initial_time(self):
        """Test parallel lines, whose swap time is negative infinity like the initial time."""
        heap = KineticHeap()
        heap.insert('low', 1.0, 0.0)
        heap.insert('high', 1.0, 1.0)
        # A swap time of -inf is not in the future, so nothing is pending even at the initial time.
        assert heap.root.swap_time == -np.inf
        assert heap.root.min_future_swap == np.inf
        assert heap.root.winner.item == 'high'
        assert heap.find_max(-np.inf) == 'high'
        check_invariants(heap)


####################################################################################################
# Brute force differential testing:
####################################################################################################


class TestAgainstBruteForce:
    """Test the heap against a brute force oracle over randomized operation sequences."""

    def test_random_operations_match_oracle(self):
        """
        Test randomized sequences of inserts, deletes and queries against an oracle that computes
        the maximum key directly. Every structural invariant is checked after every operation.

        Half of the seeds draw small integer keys and step the query time in whole units, so that
        query times land exactly on crossing times often; the other half draw continuous keys and
        times, where crossings are almost never hit exactly.
        """
        samples = 200
        for i in range(samples):
            np.random.seed(i)
            integral = (i % 2 == 0)

            heap = KineticHeap()
            items = {}
            seq = 0
            next_item = 0
            t = -20.0

            for _ in range(60):
                choice = np.random.rand()

                if choice < 0.45 or len(items) == 0:
                    if integral:
                        a = float(np.random.randint(-3, 4))
                        b = float(np.random.randint(-3, 4))
                    else:
                        a = float(np.random.uniform(-3, 3))
                        b = float(np.random.uniform(-3, 3))
                    heap.insert(next_item, a, b)
                    items[next_item] = (a, b, seq)
                    seq += 1
                    next_item += 1

                elif choice < 0.70:
                    item = int(np.random.choice(list(items.keys())))
                    heap.delete(item)
                    del items[item]

                else:
                    if integral:
                        t = t + float(np.random.randint(0, 3))
                    else:
                        t = t + float(np.random.uniform(0, 2))
                    assert heap.find_max(t) == oracle_find_max(items, t)

                check_invariants(heap)

    def test_random_delete_max_drains_in_key_order(self):
        """
        Test that repeatedly deleting the maximum at a fixed time drains the heap in decreasing
        order of key, agreeing with the oracle at every step.
        """
        samples = 50
        for i in range(samples):
            np.random.seed(1000 + i)

            heap = KineticHeap()
            items = {}
            n = int(np.random.randint(1, 40))
            for n_item in range(n):
                a = float(np.random.uniform(-3, 3))
                b = float(np.random.uniform(-3, 3))
                heap.insert(n_item, a, b)
                items[n_item] = (a, b, n_item)

            t = float(np.random.uniform(-5, 5))
            while len(items) > 0:
                expected = oracle_find_max(items, t)
                assert heap.delete_max(t) == expected
                del items[expected]
                check_invariants(heap)

            assert heap.is_empty()

    def test_large_heap_stays_balanced(self):
        """Test that the tree stays balanced and correct at a size where its height matters."""
        np.random.seed(7)
        heap = KineticHeap()
        items = {}
        n = 500

        for n_item in range(n):
            a = float(np.random.uniform(-10, 10))
            b = float(np.random.uniform(-10, 10))
            heap.insert(n_item, a, b)
            items[n_item] = (a, b, n_item)
        check_invariants(heap)

        # An AVL tree of n leaves has 2n - 1 nodes and so height at most about 1.44 * log2(2n).
        assert heap.root.height <= 1.4405 * np.log2(2 * n) + 1

        t = -50.0
        for _ in range(100):
            t = t + float(np.random.uniform(0, 1))
            assert heap.find_max(t) == oracle_find_max(items, t)
        check_invariants(heap)

    def test_sorted_insertion_stays_balanced(self):
        """Test the worst case for an unbalanced tree: items inserted in increasing key order."""
        heap = KineticHeap()
        items = {}
        n = 200

        for n_item in range(n):
            a = float(n_item)
            b = 0.0
            heap.insert(n_item, a, b)
            items[n_item] = (a, b, n_item)
            check_invariants(heap)

        assert heap.root.height <= 1.4405 * np.log2(2 * n) + 1
        assert heap.find_max(1.0) == n - 1
        assert heap.find_max(1e6) == n - 1
