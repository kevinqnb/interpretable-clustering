# Interpretable Clustering :deciduous_tree:

## Getting Started
To install dependencies, you may use a package manager of your choice. We 
provide the necessary `pyproject.toml` and `setup.py` files to manage installation. 
We reccomend using UV, which makes installation simple:
```
uv sync
```

Note that since some of our dependencies are outdated, you may need to first run the following command 
to fix installation issues with sklearn.
```
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
```

While not strictly necessary for installation, this package uses pygraphviz to 
visualize decision trees. This requires installing graphviz, which can be tricky. 
For MacOS, the following seems somewhat robust.
```
brew install graphviz

env \
  CFLAGS="-I$(brew --prefix graphviz)/include" \
  LDFLAGS="-L$(brew --prefix graphviz)/lib" \
  uv pip install pygraphviz
```

## Examples + Experiments
Example notebooks are provided to showcase the inner workings of the repository. 
Specifically, the `examples/` directory contains notebooks, including a case study for the 
climate dataset. 

Likewise, our experiments are easily reproducible using the information 
and code provided in the `experiments/` directory. 

## Datasets 

Most experiments may be run by downloading datasets with sklearn and our preprocessing functions 
defined in `data/preprocessing.py`. We also include a NOAA climate dataset 
within `data/climate` and the anuran dataset in `data/anuran`.

* NOAA National Centers for Environmental information, Climate at a Glance: Divisional Mapping, published March 2025,
  retrieved on March 14, 2025 from https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/divisional/mapping

* Colonna, J., Nakamura, E., Cristo, M., & Gordo, M. (2015). Anuran Calls (MFCCs) [Dataset]. 
UCI Machine Learning Repository. https://doi.org/10.24432/C5CC9H.

## Notes

Please note that there are some naming differences between things in the code, and 
how they are referred to in the paper. This is just a result of having iterated 
on the paper, while trying to leave the code structurally intact. Here are a few 
cases, although there may be more (we are working on updating this):
* `ScaledGreedy` is referred to as lazy greedy in the code. 
* `PEC` may sometimes be referred to as `DSCluster` in the experiment results 
