"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="intercluster.rules._splitter",
        sources=["intercluster/rules/_splitter.pyx"],
        include_dirs=[np.get_include()]
    )
]

setup(
    name="intercluster",
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()]
)
"""