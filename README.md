# Interpretable Clustering :deciduous_tree:

## Getting Started
To build a minimal installation, first ensure that poetry is installed 
as a package manager. If you do not have poetry installed, 
instructions and basic usage  may be found [here](https://python-poetry.org/docs/). 
Importantly, this uses python version 3.9 in order to 
satisfy all dependencies and we would recommend the stable 3.9.20. 

Once poetry is installed, clone the repository
and run:

```
poetry install
```

This creates a virtual environment 
with all the required dependencies and may be 
activated witht the command `poetry shell`. 
If you run into any issues with the `poetry.lock` file, it may be 
because you are using an outdated version of poetry. In that case, 
you may either consider updating, or deleting the lock file and 
regenerating a new one with the `poetry lock` command.

Additionally, you should
install and use the original [ExKMC library](https://github.com/navefr/ExKMC/tree/master).
Add this to your virtual environment by running

```
poetry run pip install ExKMC==0.0.3
```
For visualization of decision trees, this package uses pygraphviz 
(which also requires installing graphviz), which can be tricky to install. 
For MacOS, the following seems somewhat robust.
```
brew install graphviz
poetry run python -m pip install \
    --global-option=build_ext \
    --global-option="-I$(brew --prefix graphviz)/include/" \
    --global-option="-L$(brew --prefix graphviz)/lib/" \
    pygraphviz
```
The environment may then be used and activated by running 
```
eval $(poetry env activate)
```

## Examples + Experiments
Example notebooks are provided to showcase the inner workings of the repository. 
Specifically, the `examples/` folder contains notebooks, including a case study for the 
climate dataset. 

Likewise, our experiments are easily reproducible using the files provided in the `experiments`
folder. For each dataset we include a `relative_coverage.py` file to run an experiment which measures the 
result of changing coverage requirements. Similarly, the `interpretable_measurements.py` and `explanation_tree.py`
files contain code for computing and plotting interpretability metrics for both our algorithms as well as 
the explanation tree algorithm of Bandyapadhyay et al [2]. Pre-computed data and visualizations are available within 
the `data/experiments/` and `figures` folders.
## Datasets 

Most experiments may be run by downloading datasets with sklearn and our preprocessing functions 
defined in `intercluster/experiments/preprocessing.py`. We also include a NOAA climate dataset 
within `data/climate`. The exception is the Anuran dataset, which may be downloaded 
from the UCI machine learning repository. 

* NOAA National Centers for Environmental information, Climate at a Glance: Divisional Mapping, published March 2025,
  retrieved on March 14, 2025 from https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/divisional/mapping

* Colonna, J., Nakamura, E., Cristo, M., & Gordo, M. (2015). Anuran Calls (MFCCs) [Dataset]. 
UCI Machine Learning Repository. https://doi.org/10.24432/C5CC9H.

## References
1. Bandyapadhyay, S., Fomin, F.V., Golovach, P.A., Lochet, W., Purohit, N., Si-
monov, K.: How to find a good explanation for clustering? Artificial Intelligence
322, 103948 (2023)
   
2. Frost,N.,Moshkovitz,M.,Rashtchian,C.: ExKMC: Expanding Explainable K-means
Clustering. arXiv preprint arXiv:2006.02399 (2020)

2. Moshkovitz, M., Dasgupta, S., Rashtchian, C., Frost, N.: Explainable k-means and
k-medians clustering. In: International conference on machine learning. pp. 7055–7065. PMLR (2020)


