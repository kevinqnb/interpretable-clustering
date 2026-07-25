#!/usr/bin/env python
"""
Implements the MDLP discretization criterion from Usama Fayyad's paper
"Multi-Interval Discretization of Continuous-Valued Attributes for
Classification Learning."
"""

from setuptools import Extension, find_packages, setup

if __name__ == '__main__':
    # see https://stackoverflow.com/a/42163080 for the approach to pushing
    # numpy and cython dependencies into extension building only
    try:
        # if Cython is available, we will rebuild from the pyx file directly
        from Cython.Distutils import build_ext
        print("Using cython to build the extension")
        sources = ['mdlp/_mdlp.pyx']
    except ImportError:
        # else we build from the cpp file included in the distribution
        print("Cannot find available cython installation. Using cpp to build the extension")
        from setuptools.command.build_ext import build_ext
        sources = ['mdlp/_mdlp.cpp']

    class CustomBuildExt(build_ext):
        """Custom build_ext class to defer numpy imports until needed.

        Overrides the build process for an extension and adds numpy include dirs
        to each extension build.
        """

        def build_extensions(self):
            import numpy

            numpy_include = numpy.get_include()
            for ext in self.extensions:
                ext.include_dirs = list(ext.include_dirs or [])
                if numpy_include not in ext.include_dirs:
                    ext.include_dirs.append(numpy_include)

            super().build_extensions()

    cpp_ext = Extension(
        'mdlp._mdlp',
        sources=sources,
        language='c++',
        include_dirs=[],
        libraries=[],
    )

    setup(
        name='mdlp-discretization',
        version='0.3.3',
        description=__doc__,
        license='BSD 3 Clause',
        url='https://github.com/hlin117/mdlp-discretization',
        author='Henry Lin',
        author_email='hlin117@gmail.com',
        # NumPy is required at build time to compile the extension (for headers)
        # and at runtime.
        setup_requires=[
            'numpy>=1.11.2',
        ],
        install_requires=[
            'numpy>=1.11.2',
            'scipy>=0.18.1',
            'scikit-learn>=0.18.1',
        ],
        packages=find_packages(exclude=('tests', 'build')),
        ext_modules=[cpp_ext],
        cmdclass={'build_ext': CustomBuildExt},
    )
