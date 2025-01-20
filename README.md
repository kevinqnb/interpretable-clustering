# Interpretable Clustering

To build a minimal installation clone the repository
and then run:
`poetry install`

To install and use the original [ExKMC library](https://github.com/navefr/ExKMC/tree/master) run
 `poetry run pip install ExKMC==0.0.3`

For visualization of decision trees, this package uses graphviz, which must be 
downloaded and installed separately. For linux users 
this may be done with:
`sudo apt install graphviz`
Likewise for MacOS:
`brew install graphviz`
Windows users will need to download from the [source](https://graphviz.org/download/), taking care
to ensure that the appropriate folders are added to the path.

The packages and python version have been set 
mainly to work with requirements for ExKMC.
