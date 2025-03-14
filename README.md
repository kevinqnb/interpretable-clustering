# Interpretable Clustering

## Getting Started
To build a minimal installation, first ensure that poetry is installed 
as a package manager. If you do not have poetry installed, 
instructions and basic usage  may be found [here](https://python-poetry.org/docs/). 

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
python -m pip install \
    --global-option=build_ext \
    --global-option="-I$(brew --prefix graphviz)/include/" \
    --global-option="-L$(brew --prefix graphviz)/lib/" \
    pygraphviz
```


# Datasets 

Most experiments may be run by downloading datasets with sklearn and our preprocessing functions 
defined in `intercluster/experiments/preprocessing.py`. We also include a NOAA climate dataset 
within `data/climate`. The exception is the Anuran dataset, which may be downloaded 
from the UCI machine learning repository. 

NOAA National Centers for Environmental information, Climate at a Glance: Regional Time Series, 
published March 2025, retrieved on March 13, 2025 from 
https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/regional/time-series

Colonna, J., Nakamura, E., Cristo, M., & Gordo, M. (2015). Anuran Calls (MFCCs) [Dataset]. 
UCI Machine Learning Repository. https://doi.org/10.24432/C5CC9H.
