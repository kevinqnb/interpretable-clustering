import numpy as np
from typing import Any, Optional, Tuple


####################################################################################################


class KineticNode():
    """
    Node object to be used within a kinetic heap. Nodes are either external (leaf) nodes, which
    carry an item and its linear key, or internal nodes, which carry no item and serve only to
    route searches and to cache the state of their subtree.

    Args:
        None
    Attributes:
        type (str): Internally set to 'internal' or 'leaf' depending on if the node is a
            routing node or an external node holding an item.

        item (Any): (For leaf nodes only) The identifier of the item held by this node.

        a (float): (For leaf nodes only) The key coefficient of this node's item, i.e. the slope
            of its key line.

        b (float): (For leaf nodes only) The key constant of this node's item, i.e. the intercept
            of its key line.

        seq (int): (For leaf nodes only) Insertion sequence number. Used as the final tiebreaker
            of the leaf ordering so that items with identical keys still receive a strict order.

        left (KineticNode): (For internal nodes only) Pointer to the left child of this node.

        right (KineticNode): (For internal nodes only) Pointer to the right child of this node.

        parent (KineticNode): Pointer to the parent of this node, or `None` at the root. Needed to
            retrace from a leaf back up to the root during deletion.

        height (int): Height of this node's subtree, with leaves at height zero. Used to maintain
            the AVL balance condition.

        winner (KineticNode): The leaf of this node's subtree whose item has maximum key at the
            heap's current time. For a leaf, this is the leaf itself.

        swap_time (float): (For internal nodes only) The time at which this node's winner switches
            from its left child's winner to its right child's winner. `-inf` when the two key lines
            are parallel and therefore never cross.

        min_future_swap (float): The smallest swap time strictly greater than the heap's current
            time among this node and its descendants, or `inf` if there is none. This is what
            allows a subtree with no upcoming swap to be pruned when time advances.

        predecessor (KineticNode): (For internal nodes only) The rightmost leaf of this node's left
            subtree. Its key is the largest in the left subtree, which makes it the router used to
            guide an insertion search down from the root.

        max_leaf (KineticNode): The rightmost leaf of this node's subtree. Cached so that
            `predecessor` can be pulled up from a node's children in constant time rather than
            found by a downward walk.
    """

    def __init__(self):
        self.type = None
        self.item = None
        self.a = None
        self.b = None
        self.seq = None
        self.left = None
        self.right = None
        self.parent = None
        self.height = None
        self.winner = None
        self.swap_time = None
        self.min_future_swap = None
        self.predecessor = None
        self.max_leaf = None

    def leaf_node(
        self,
        item : Any,
        a : float,
        b : float,
        seq : int
    ):
        """
        Initializes this to be an external (leaf) node of the heap, holding a single item.

        All of a leaf's cached fields are fixed at construction: a leaf's winner is its own item,
        a leaf contains no swap, and a leaf is its own rightmost leaf.

        Args:
            item (Any): The identifier of the item held by this node.

            a (float): The key coefficient of the item, i.e. the slope of its key line.

            b (float): The key constant of the item, i.e. the intercept of its key line.

            seq (int): Insertion sequence number, used as the final tiebreaker of the leaf ordering.
        """
        self.type = 'leaf'
        self.item = item
        self.a = a
        self.b = b
        self.seq = seq
        self.left = None
        self.right = None
        self.height = 0
        self.winner = self
        self.swap_time = -np.inf
        self.min_future_swap = np.inf
        self.predecessor = None
        self.max_leaf = self

    def internal_node(
        self,
        left,
        right
    ):
        """
        Initializes this to be an internal (routing) node of the heap with the given children.

        NOTE: This only wires up the child and parent pointers. The cached fields (`winner`,
        `swap_time`, `min_future_swap`, `predecessor`, `max_leaf` and `height`) depend on the
        heap's current time and are filled in by `KineticHeap._update`.

        Args:
            left (KineticNode): Pointer to the left child of this node.

            right (KineticNode): Pointer to the right child of this node.
        """
        self.type = 'internal'
        self.item = None
        self.a = None
        self.b = None
        self.seq = None
        self.left = left
        self.right = right
        left.parent = self
        right.parent = self


####################################################################################################


class KineticHeap():
    """
    A simple kinetic max-heap, following Kaplan, Tarjan and Tsioutsiouliklis, "Faster Kinetic Heaps
    and Their Use in Broadcast Scheduling" (Section 2). The paper presents the structure as a
    min-heap; this is the max-heap obtained by negating all keys.

    Each item `i` in the heap has a key that is a linear function of time,

        key_i(t) = a_i * t + b_i,

    with `a_i` the key coefficient and `b_i` the key constant. The heap answers `find_max(t)`, the
    item of maximum key at time `t`, for a nondecreasing sequence of query times, without ever
    re-sorting the items. The current time `t_c` is the time of the most recent query, or `-inf`
    before any query has occurred.

    Structure. The heap is an AVL tree that is simultaneously two tournaments on the same nodes.
    Items live in external (leaf) nodes; internal nodes hold no item and instead cache (i) the
    winner of their subtree, the item of maximum key at `t_c`, and (ii) the minimum future swap
    time in their subtree. AVL is used rather than a red-black tree because its balance condition
    is a single integer height per node, which is cheap to maintain alongside the other cached
    fields and easy to verify exactly.

    Leaf ordering. Leaves are ordered left to right by increasing key coefficient, ties broken by
    increasing key constant, and further ties broken by increasing insertion sequence number. The
    coefficient ordering is what makes the structure a max-heap: the line that eventually dominates
    is the one of largest slope, so the maximum-key item migrates monotonically rightward as time
    advances, and the winner of an internal node therefore switches from its left child to its
    right child at most once and never switches back. (This is the mirror image of the paper's
    min-heap, which orders by decreasing coefficient.) The sequence number is not needed for
    correctness of the swap times, which depend only on the coefficients; it is there to make the
    leaf ordering a strict total order even when two items have identical keys, so that duplicate
    keys cannot collide and tie-breaking is deterministic.

    Ties. Where several items are tied for the maximum key at `t_c`, `find_max` returns the one
    that is rightmost in the leaf ordering: largest key coefficient, then largest key constant,
    then latest insertion. Consequently a query made at exactly the time two lines cross returns
    the steeper of the two, i.e. the swap is treated as having already occurred. This is the
    convention that keeps the winners and the swap times consistent with one another: a swap
    scheduled at time `s` has fired once `t_c >= s`.

    Numerical policy. Keys are ordinary floats, and coefficients are compared for equality exactly,
    with no tolerance. The leaf ordering and the swap time formula therefore agree by construction
    about which pairs of lines are parallel, which is what guarantees that the left child of a node
    never has a larger coefficient than the right child. Two coefficients that differ by a very
    small amount produce a correspondingly enormous swap time, which is the correct answer: the
    lines do cross, but so far in the future that the swap is pruned away.

    Args:
        None
    Attributes:
        root (KineticNode): The root of the tree, or `None` when the heap is empty.

        current_time (float): The current time `t_c`, i.e. the time of the most recent query, or
            `-inf` if no query has occurred yet.

        leaves (Dict[Any, KineticNode]): Maps each item in the heap to the leaf holding it, so that
            `delete` can locate an item in constant time.
    """

    def __init__(self):
        self.root = None
        self.current_time = -np.inf
        self.leaves = {}
        self._next_seq = 0

    @staticmethod
    def make_heap():
        """
        Creates a new, empty kinetic heap. Runs in O(1) worst case time.

        Returns:
            (KineticHeap): An empty heap, with a current time of `-inf`.
        """
        return KineticHeap()

    def __len__(self) -> int:
        """
        Returns:
            (int): The number of items currently in the heap.
        """
        return len(self.leaves)

    def __contains__(self, item : Any) -> bool:
        """
        Args:
            item (Any): The item to look for.

        Returns:
            (bool): `True` if the item is currently in the heap, and `False` otherwise.
        """
        return item in self.leaves

    def is_empty(self) -> bool:
        """
        Returns:
            (bool): `True` if the heap holds no items, and `False` otherwise.
        """
        return self.root is None


    ################################################################################################
    # Cached field maintenance:
    ################################################################################################

    def _order(self, leaf : KineticNode) -> Tuple[float, float, int]:
        """
        The sort key defining the left to right ordering of the leaves. This is a strict total
        order: no two leaves in the heap compare equal, since insertion sequence numbers are
        distinct.

        Args:
            leaf (KineticNode): The leaf to take the ordering key of.

        Returns:
            (tuple): The triple of key coefficient, key constant and insertion sequence number.
        """
        return (leaf.a, leaf.b, leaf.seq)

    def _wins(self, i : KineticNode, j : KineticNode, t : float) -> bool:
        """
        Determines whether leaf `i`'s item has a strictly larger key than leaf `j`'s item at time
        `t`. Ties are therefore resolved in favor of `j`, which is why a node's winner is taken to
        be its right child's winner whenever the two are tied.

        NOTE: Keys are compared here rather than evaluated, because the current time begins at
        `-inf`, where `a * t + b` would be `nan` for an item whose coefficient is zero. At an
        infinite time the comparison is taken in the limit, where the item with the smaller
        coefficient wins as `t` tends to `-inf` (and the larger coefficient wins as `t` tends to
        `inf`), ties in either case going to the larger constant.

        Args:
            i (KineticNode): The leaf challenging for the win.

            j (KineticNode): The leaf being challenged.

            t (float): The time at which to compare the two keys. May be infinite.

        Returns:
            (bool): `True` if `i`'s key is strictly larger than `j`'s key at time `t`.
        """
        if t == -np.inf:
            return (-i.a, i.b) > (-j.a, j.b)
        if t == np.inf:
            return (i.a, i.b) > (j.a, j.b)
        return i.a * t + i.b > j.a * t + j.b

    def _update(self, x : KineticNode):
        """
        Recomputes every cached field of the internal node `x` from those of its two children, in
        constant time, at the heap's current time.

        This single routine is what restores the structure after any change. Because a rotation
        preserves the left to right order of the leaves, and because every field of `x` is a
        function of the corresponding fields of its children, replaying this along the path from a
        change up to the root is enough to repair the whole tree, at rotation sites included.

        Args:
            x (KineticNode): The internal node to recompute. Its children must already be correct.
        """
        assert x.type == 'internal', "Only internal nodes carry recomputable cached fields."
        y = x.left
        z = x.right
        t_c = self.current_time

        x.height = 1 + max(y.height, z.height)
        x.max_leaf = z.max_leaf
        x.predecessor = y.max_leaf

        # The tournament on items: the winner is the item of maximum key at the current time, ties
        # going to the right child, whose line is the steeper of the two.
        i = y.winner
        j = z.winner
        x.winner = i if self._wins(i, j, t_c) else j

        # The time at which the winner switches from i to j, i.e. where the two lines cross. The
        # leaf ordering guarantees that i's coefficient is no larger than j's, so the denominator
        # here is positive and j is the steeper line: before the crossing i is on top, after it j
        # is. Parallel lines never cross and so never swap.
        if i.a != j.a:
            x.swap_time = (i.b - j.b) / (j.a - i.a)
        else:
            x.swap_time = -np.inf

        # The heap on swap times. Only x's own swap time may lie in the past here, and it is
        # discarded if so, having already fired. The children's values are by induction already
        # either strictly in the future or infinite, since the same rule was applied when they were
        # last recomputed, and no node whose value falls at or below the current time survives an
        # advance of the clock unrecomputed.
        own_swap = x.swap_time if x.swap_time > t_c else np.inf
        x.min_future_swap = min(own_swap, y.min_future_swap, z.min_future_swap)


    ################################################################################################
    # Rebalancing:
    ################################################################################################

    def _replace_child(self, parent : Optional[KineticNode], old : KineticNode, new : KineticNode):
        """
        Points `parent` at `new` wherever it currently points at `old`, updating the root of the
        heap instead if `old` has no parent.

        Args:
            parent (KineticNode): The parent whose child pointer is to be redirected, or `None` if
                `old` is the root of the heap.

            old (KineticNode): The child currently pointed at.

            new (KineticNode): The child to point at in its place.
        """
        new.parent = parent
        if parent is None:
            self.root = new
        elif parent.left is old:
            parent.left = new
        else:
            parent.right = new

    def _rotate_right(self, x : KineticNode) -> KineticNode:
        """
        Rotates the internal node `x` to the right, promoting its left child in its place, and
        recomputes the cached fields of both, the demoted node first.

        Args:
            x (KineticNode): The internal node to rotate. Its left child must also be internal.

        Returns:
            (KineticNode): The node now occupying `x`'s former position, i.e. its old left child.
        """
        y = x.left
        self._replace_child(x.parent, x, y)
        x.left = y.right
        x.left.parent = x
        y.right = x
        x.parent = y
        self._update(x)
        self._update(y)
        return y

    def _rotate_left(self, x : KineticNode) -> KineticNode:
        """
        Rotates the internal node `x` to the left, promoting its right child in its place, and
        recomputes the cached fields of both, the demoted node first.

        Args:
            x (KineticNode): The internal node to rotate. Its right child must also be internal.

        Returns:
            (KineticNode): The node now occupying `x`'s former position, i.e. its old right child.
        """
        z = x.right
        self._replace_child(x.parent, x, z)
        x.right = z.left
        x.right.parent = x
        z.left = x
        x.parent = z
        self._update(x)
        self._update(z)
        return z

    def _rebalance(self, x : KineticNode) -> KineticNode:
        """
        Restores the AVL balance condition at the internal node `x`, whose children are assumed to
        be balanced and to have correct cached fields.

        Args:
            x (KineticNode): The internal node to rebalance.

        Returns:
            (KineticNode): The node now occupying `x`'s position, which is `x` itself if no
                rotation was needed.
        """
        balance = x.left.height - x.right.height
        if balance > 1:
            # Left heavy. A left-right case is first reduced to a left-left case.
            if x.left.left.height < x.left.right.height:
                self._rotate_left(x.left)
            return self._rotate_right(x)
        if balance < -1:
            # Right heavy. A right-left case is first reduced to a right-right case.
            if x.right.right.height < x.right.left.height:
                self._rotate_right(x.right)
            return self._rotate_left(x)
        return x

    def _retrace(self, x : Optional[KineticNode]):
        """
        Walks from the internal node `x` up to the root, recomputing cached fields and rebalancing
        at every node along the way.

        NOTE: Unlike a plain AVL tree, this cannot stop early once the heights settle. A structural
        change moves the winner and the minimum future swap time of every node on the path back to
        the root, whether or not any rotation takes place, so the whole path is always recomputed.

        Args:
            x (KineticNode): The lowest node whose cached fields need recomputing, or `None`, in
                which case there is nothing to do.
        """
        while x is not None:
            self._update(x)
            x = self._rebalance(x)
            x = x.parent


    ################################################################################################
    # Operations:
    ################################################################################################

    def insert(self, item : Any, a : float, b : float):
        """
        Inserts an item with key `a * t + b` into the heap. Runs in O(log n) actual time, and
        O(log^2 n) amortized time once the cost of advancing the current time is charged back to
        the insertions and deletions that paid for it.

        The search for the new leaf's position is guided from the root by the `predecessor` of each
        internal node, the largest leaf of its left subtree, so no auxiliary search structure is
        needed. The landing leaf is then replaced by a new internal node holding it and the new
        leaf, in the correct order.

        Args:
            item (Any): The identifier of the item to insert. Must be hashable, and must not
                already be in the heap.

            a (float): The key coefficient of the item, i.e. the slope of its key line.

            b (float): The key constant of the item, i.e. the intercept of its key line.

        Raises:
            ValueError: If the item is already in the heap.
        """
        if item in self.leaves:
            raise ValueError("Item is already in the heap, and cannot be inserted twice.")

        leaf = KineticNode()
        leaf.leaf_node(item, float(a), float(b), self._next_seq)
        self._next_seq += 1
        self.leaves[item] = leaf

        if self.root is None:
            self.root = leaf
            return

        # Descend to the leaf that the new item belongs next to, routing on each internal node's
        # predecessor: every leaf of the left subtree orders at or below it, and every leaf of the
        # right subtree orders above it.
        x = self.root
        while x.type == 'internal':
            if self._order(leaf) <= self._order(x.predecessor):
                x = x.left
            else:
                x = x.right

        # Split the landing leaf, replacing it with an internal node holding both leaves in order.
        # The landing leaf's old parent is taken first, since adopting it below overwrites it.
        grandparent = x.parent
        parent = KineticNode()
        if self._order(leaf) < self._order(x):
            parent.internal_node(leaf, x)
        else:
            parent.internal_node(x, leaf)
        self._replace_child(grandparent, x, parent)
        self._retrace(parent)

    def delete(self, item : Any):
        """
        Deletes an item from the heap. Runs in O(log n) actual time, and O(log^2 n) amortized time
        once the cost of advancing the current time is charged back to the insertions and deletions
        that paid for it.

        The item's leaf is found in constant time through the item to leaf map, so no comparison
        search is needed. The leaf's parent is then contracted away, its sibling taking its place.

        Args:
            item (Any): The identifier of the item to delete.

        Raises:
            ValueError: If the item is not in the heap.
        """
        if item not in self.leaves:
            raise ValueError("Item is not in the heap, and so cannot be deleted.")

        leaf = self.leaves.pop(item)
        parent = leaf.parent
        if parent is None:
            self.root = None
            return

        sibling = parent.right if parent.left is leaf else parent.left
        self._replace_child(parent.parent, parent, sibling)
        self._retrace(sibling.parent)

    def find_max(self, t : float) -> Optional[Any]:
        """
        Advances the current time to `t` and returns the item of maximum key at that time.

        Query times must be nondecreasing. Where several items are tied for the maximum, the one
        that is rightmost in the leaf ordering is returned; see the class docstring.

        Args:
            t (float): The time at which to find the maximum. Must be at least the current time.

        Returns:
            (Any): The identifier of the item of maximum key at time `t`, or `None` if the heap is
                empty.

        Raises:
            ValueError: If `t` precedes the current time.
        """
        self._advance(t)
        if self.root is None:
            return None
        return self.root.winner.item

    def delete_max(self, t : float) -> Optional[Any]:
        """
        Advances the current time to `t`, then finds and deletes the item of maximum key at that
        time.

        Args:
            t (float): The time at which to find the maximum. Must be at least the current time.

        Returns:
            (Any): The identifier of the deleted item, or `None` if the heap is empty.

        Raises:
            ValueError: If `t` precedes the current time.
        """
        item = self.find_max(t)
        if item is None:
            return None
        self.delete(item)
        return item

    def peek_max(self) -> Optional[Tuple[Any, float]]:
        """
        Returns the item of maximum key and its key value at the current time, without advancing
        the current time and without removing the item from the heap.

        Unlike `find_max`, this takes no time argument: it reads the already-cached winner at the
        heap's current `current_time`, which is what a caller who just called `find_max` or
        `delete_max` at time `t` wants next, without a redundant no-op re-query at the same `t`.

        Returns:
            (tuple): A pair `(item, value)` of the item of maximum key and its key value at the
                current time, or `None` if the heap is empty.
        """
        if self.root is None:
            return None
        w = self.root.winner
        return w.item, w.a * self.current_time + w.b

    def _advance(self, t : float):
        """
        Advances the current time of the heap from `t_c` to `t`, applying every swap that falls in
        the interval `(t_c, t]`.

        The tree is searched from the root, pruning any node whose minimum future swap time lies
        beyond `t`, since no swap anywhere in such a subtree falls in the interval. The nodes that
        survive the pruning form a connected subtree containing the root, and their cached fields
        are recomputed bottom up at the new current time.

        NOTE: The current time is set before the search begins. This is safe, and is what lets the
        search and the recomputation share a single pass: the pruning test reads the stored minimum
        future swap times, which the search does not touch, while the recomputation on the way back
        up sees the new current time. The subtrees that are pruned keep cached fields that are
        stale but nonetheless exactly correct, since a subtree with no swap in `(t_c, t]` has the
        same winner at `t` as it had at `t_c`.

        A single advance takes O(n) time in the worst case. The O(1) amortized bound comes from a
        potential function proportional to log n times the number of nodes holding a future swap
        time, which the insertions and deletions pay into; the potential itself is not maintained
        here.

        Args:
            t (float): The time to advance to. Must be at least the current time.

        Raises:
            ValueError: If `t` precedes the current time.
        """
        t = float(t)
        if t < self.current_time:
            raise ValueError(
                "Query times must be nondecreasing, and the given time precedes the current time "
                "of the heap."
            )
        if t == self.current_time or self.root is None:
            self.current_time = t
            return

        root = self.root
        self.current_time = t
        self._advance_node(root, t)

    def _advance_node(self, x : KineticNode, t : float):
        """
        Applies every swap in the interval `(t_c, t]` within the subtree rooted at `x`, where `t_c`
        is the current time as it stood before the advance began.

        Args:
            x (KineticNode): The root of the subtree to advance.

            t (float): The time being advanced to, which is already the heap's current time.
        """
        if x.type == 'leaf':
            return
        if x.min_future_swap > t:
            # No swap anywhere in this subtree falls in the interval, so its cached fields, and
            # its winner in particular, are unchanged.
            return
        self._advance_node(x.left, t)
        self._advance_node(x.right, t)
        self._update(x)
