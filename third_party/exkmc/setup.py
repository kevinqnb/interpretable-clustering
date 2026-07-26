from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize
import numpy
import sys
from ExKMC import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()


# Apple's system clang doesn't support -fopenmp; only pass it on platforms
# (Linux/gcc) where the OpenMP-parallel prange() sections in cut_finder.pyx
# can actually be compiled. Elsewhere the pragmas are silently ignored and
# the loops just run single-threaded.
openmp_args = ['-fopenmp'] if sys.platform.startswith('linux') else []

extensions = cythonize([
    Extension(
        "cut_finder",
        ["ExKMC/splitters/cut_finder.pyx"],
        extra_compile_args=openmp_args,
        extra_link_args=openmp_args,
    )
])


setup(
    name="ExKMC",
    version=__version__,
    author="Nave Frost",
    author_email="navefrost@mail.tau.edu",
    liceanse="MIT",
    description="Expanding Explainable K-Means Clustering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/navefr/ExKMC",
    packages=find_packages(),
    ext_modules=extensions,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_dirs=[numpy.get_include()],
    python_requires='>=3.0',
)