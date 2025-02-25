# Interpretable Clustering

To build a minimal installation, first ensure that poetry is installed 
as a package manager. If you do not have poetry installed, 
instructions and basic usage  may be found [here](https://python-poetry.org/docs/). 

Once poetry is installed, clone the repository
and run:`poetry install`

This creates a virtual environment 
with all the required dependencies and may be 
activated witht the command `poetry shell`. 
If you run into any issues with the `poetry.lock` file, it may be 
because you are using an outdated version of poetry. In that case, 
you may either consider updating, or deleting the lock file and 
regenerating a new one with the `poetry lock` command.

Additionally, you may want to 
 install and use the original [ExKMC library](https://github.com/navefr/ExKMC/tree/master).
Add this to your virtual environment by running
 `poetry run pip install ExKMC==0.0.3`.

For visualization of decision trees, this package uses graphviz, which must be 
downloaded and installed separately. For linux users 
this may be done with:
`sudo apt install graphviz`
Likewise for MacOS:
`brew install graphviz`
Windows users will need to download from the [source](https://graphviz.org/download/), taking care
to ensure that the appropriate folders are added to the path.

Reminders for myself:
Plotting python with latex requires some minimal installation of latex. For linux 
this may be installed as:

```
sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super
```

On Mac OS a working installation can be built with:

```
brew install --cask basictex
sudo tlmgr install type1cm dvipng cm-super
```

The packages and python version have been set 
mainly to work with requirements for ExKMC.
