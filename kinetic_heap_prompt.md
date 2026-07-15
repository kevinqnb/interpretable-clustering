# Task: Implement the Simple Kinetic Max-Heap

Implement the **simple kinetic heap** from Kaplan, Tarjan & Tsioutsiouliklis, *"Faster Kinetic Heaps and Their Use in Broadcast Scheduling"* (Section 2), in `src/intercluster/kinetic_heap.py`.

The paper presents this structure as a **min-heap**. We want a **max-heap**. The paper notes you convert between the two by negating all keys; the instructions below have already carried that negation through, so implement exactly what is written here — do not re-derive from the paper's min-heap formulas.

Integrate it seamlessly into the existing repository: match the current typing conventions, docstring style, error-handling patterns, `__init__.py` exports, and formatter/linter config. Read a few neighboring modules first and follow what you find there.

## Definitions

Each item `i` has a key that is a **linear function of time**: `key_i(t) = a_i * t + b_i`, where `a_i` is the *key coefficient* and `b_i` the *key constant*.

The **current time** `t_c` is the `t`-value of the most recent `find_max` / `delete_max`, or `-inf` if none has occurred. Queries are **monotone**: successive query times are nondecreasing. Raise on a query with `t < t_c`.

This is a **max-heap**: the winner of a subtree is its item of **maximum** key at `t_c`.

## Structure

A balanced binary search tree (red-black or AVL; pick one and say why) that is *simultaneously* two tournaments on the same tree:

1. **Tournament on items** — items live in *external* (leaf) nodes; each internal node caches the winner of its subtree (maximum key at `t_c`).
2. **Heap on swap times** — the same internal nodes cache the minimum future swap time in their subtree.

### Leaf ordering (critical — note the direction)

Leaves are ordered left-to-right by **increasing key coefficient**, ties broken by **increasing key constant**:

> `i` is left of `j` iff `a_i < a_j`, or (`a_i == a_j` and `b_i < b_j`).

Rationale: for a max-heap, the line that eventually dominates is the one with the *largest* slope. This ordering makes the maximum-key item migrate monotonically *rightward* as `t` increases, which is what makes the swap-time heap correct. (This is the mirror image of the paper's min-heap, which sorts by *decreasing* coefficient.)

### Per-node cached fields

Each internal node `x` with left child `y` and right child `z` stores:

- **`predecessor`** — pointer to the rightmost external node in `x`'s left subtree. Used to guide insertion search from the root without an auxiliary structure.

- **`winner`** — item of **maximum** key at `t_c` in `x`'s subtree. If `x` is a leaf, the winner is its item. Otherwise, with `i = winner(y)` and `j = winner(z)`:
  ```
  winner(x) = i  if  a_i * t_c + b_i >= a_j * t_c + b_j  else  j
  ```

- **`swap_time` `s_x`** — the time the winner of `x` switches from `i` (left) to `j` (right):
  ```
  s_x = (b_i - b_j) / (a_j - a_i)   if a_i != a_j
  s_x = -inf                        if a_i == a_j   (parallel lines, never cross)
  ```
  This is just the crossing point of the two lines; the sign arrangement above keeps the denominator positive under the max-heap ordering. The leaf ordering guarantees `a_i <= a_j`, so `j`'s line is steeper: before the crossing `i` is on top, after it `j` is. The winner therefore switches left→right at most once and **never switches back**.

- **`min_future_swap` `f_x`** — smallest swap time strictly greater than `t_c` among `x` and its descendants; `inf` if none. For a leaf, `f_x = inf`. For an internal node:
  ```
  f_x = min(max(t_c, s_x), max(t_c, f_y), max(t_c, f_z))
  if f_x <= t_c:  f_x = inf
  ```
  This stays a `min` even in the max-heap: we always want the *earliest upcoming* swap. The clamp-then-check is what filters out already-past swaps.

## Operations

| Operation | Behavior | Target |
|---|---|---|
| `make_heap()` | new empty heap | `O(1)` worst-case |
| `insert(i, a, b)` | BST insert at a leaf using the ordering above; recompute cached fields bottom-up along the search path, including at rotation sites | `O(log n)` actual, `O(log^2 n)` amortized |
| `delete(i)` | BST delete; same bottom-up recomputation | `O(log n)` actual, `O(log^2 n)` amortized |
| `find_max(t)` | advance current time to `t`, then return root's winner | `O(1)` amortized |
| `delete_max(t)` | `find_max(t)` followed by `delete` | — |

`delete` requires locating an item in `O(1)`: keep a `dict` from item id → leaf node.

### Advancing the current time `t_c -> t'` (the only nontrivial part)

Unchanged from the min-heap version — the swap-time heap is orientation-agnostic.

1. **Search.** Descend from the root. At node `x`, if `f_x > t'`, the whole subtree contains no swap in `(t_c, t']` — **prune it**. Otherwise recurse into both children and also test `s_x` itself. This locates exactly the nodes whose swap time lies in `(t_c, t']`.
2. **Recompute.** The root-to-node paths of all found nodes form a connected subtree. Set `t_c = t'`, then recompute `winner`, `swap_time`, `min_future_swap` for every node in that subtree, **bottom-up**.

Worst case for a single advance is `Θ(n)`; the cost is amortized against insert/delete via a potential function proportional to `log n` times the number of nodes with a future swap time. Do not implement the potential function — just note the bound in the docstring.

## Tests

Add tests alongside the repo's existing suite, following its conventions (pytest, fixtures, naming). Cover:

- **Brute-force differential test.** Reference oracle: keep a plain list of `(a, b)` and compute `argmax(a*t + b)` directly. Run randomized sequences of `insert` / `delete` / `find_max` with **nondecreasing** query times, and assert the heap's `find_max` matches the oracle at every step. This is the main correctness test — run it over many seeds and sizes.
- **Degenerate keys.** All coefficients equal (parallel lines, `s_x = -inf`); all constants equal; a single item; empty heap (`find_max` should raise or return `None` — pick one and document it).
- **Tie-breaking.** Items with identical `(a, b)` — the structure must remain a valid BST under the ordering and not lose items. Where several items tie for the maximum, document which one is returned and make it deterministic.
- **Monotonicity violation.** `find_max(t)` with `t < t_c` raises.
- **Swap correctness.** Two lines that cross at a known time `t*`; assert the winner flips exactly at `t*` and check both sides of the boundary. Verify the flip goes in the max-heap direction (the steeper line wins *after* `t*`).
- **Asymptotic sanity.** For large `t`, `find_max` returns the item of largest coefficient (ties broken by largest constant) — this directly exercises the reversed leaf ordering and would catch a min-heap left in by mistake.
- **Structural invariants** (test-only helper): after every mutation, verify BST leaf ordering holds (increasing coefficient), every cached `winner` / `swap_time` / `min_future_swap` equals its recomputed-from-scratch value, and the tree is balanced.
- **Numerical edge cases.** Swap times near-coincident with `t_c`; very close coefficients (`a_j - a_i` tiny) — decide on and document a tolerance policy, or use exact rational arithmetic if the repo already does elsewhere.

Run the full test suite plus the repo's linter/formatter/type-checker before reporting done.
