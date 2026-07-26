# Provenance

Vendored, patched copy of [ExKMC](https://github.com/navefr/ExKMC) (Frost, Moshkovitz,
Rashtchian, "ExKMC: Expanding Explainable k-Means Clustering", ICML 2020), MIT licensed
(see `LICENSE`).

Vendored instead of installed from PyPI/git because the upstream package (last released
as `0.0.3`) does not build/run under modern NumPy, scikit-learn, and Python. Local
modifications relative to the `0.0.3` PyPI release:

- `ExKMC/Tree.py`: drop the `n_jobs` keyword to `sklearn.cluster.KMeans`, which sklearn
  removed.
- `ExKMC/Tree.py`: `Tree.plot()` takes a `view` argument controlling whether the
  rendered graph is opened in a viewer, instead of always calling `Source.view()`.
- `setup.py`: always builds `cut_finder` from `cut_finder.pyx` via `cythonize()` (the
  upstream setup.py only did this behind a `--cython` flag that build tools like `uv`
  never pass, and otherwise compiled a pre-generated `cut_finder.c` that isn't checked
  into this repo, per the repo-wide `*.c` `.gitignore` rule). No behavior change.

These are the same fixes present in a couple of still-open/unmerged pull requests
against the upstream repository.
