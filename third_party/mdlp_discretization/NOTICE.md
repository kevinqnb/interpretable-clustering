# Provenance

Vendored, patched copy of [mdlp-discretization](https://github.com/hlin117/mdlp-discretization)
(an implementation of Fayyad & Irani's MDLP discretization criterion), BSD-3-Clause
licensed (see `LICENSE`).

Vendored instead of installed from PyPI/git because the upstream package (last released
as `0.3.3`) does not build/run under modern NumPy and scikit-learn. Local modifications
relative to the `0.3.3` release:

- `mdlp/_mdlp.pyx`: `np.int_t` -> `np.int64_t` (NumPy removed the `np.int_t`-backing
  alias); `mdlp/_mdlp.cpp` regenerated from the `.pyx` accordingly. No behavior change.
- `mdlp/discretization.py`: `sklearn.utils.check_array`'s `force_all_finite` keyword was
  renamed to `ensure_all_finite`.
- `setup.py`: build_ext now injects NumPy's include dir for every extension via
  `build_extensions` instead of a now-removed `build_ext.run` override; added a
  `pyproject.toml` declaring build requirements for isolated builds.
