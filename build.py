from __future__ import annotations
import os
import shutil
import platform
from pathlib import Path
from glob import glob

from Cython.Build import cythonize
from setuptools import Distribution
from setuptools import Extension
from setuptools.command.build_ext import build_ext
import numpy


COMPILE_ARGS = ["-march=native", "-O3"]
if platform.machine() == "x86_64":
    COMPILE_ARGS += ["-msse", "-msse2", "-mfma", "-mfpmath=sse"]

LINK_ARGS = []
INCLUDE_DIRS = [numpy.get_include()]
LIBRARIES = ["m"]
DEFINE_MACROS = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]


def build() -> None:
    sources = glob("src/intercluster/**/*.pyx", recursive=True)
    if not sources:
        raise FileNotFoundError("No .pyx files found under src/intercluster/")

    extensions = [
        Extension(
            "*",
            #["src/intercluster/***/*.pyx", "src/intercluster/*.pyx"],
            sources,
            extra_compile_args=COMPILE_ARGS,
            extra_link_args=LINK_ARGS,
            include_dirs=INCLUDE_DIRS,
            libraries=LIBRARIES,
            define_macros=DEFINE_MACROS,
        )
    ]
   
    ext_modules = cythonize(
        extensions,
        include_path=INCLUDE_DIRS,
        compiler_directives={"binding": True, "language_level": 3},
    )

    distribution = Distribution({
        "name": "intercluster",
        "ext_modules": ext_modules
    })

    cmd = build_ext(distribution)
    cmd.ensure_finalized()
    cmd.run()

    # Copy built extensions back to the project
    for output in cmd.get_outputs():
        output = Path(output)
        relative_extension = Path("src") / output.relative_to(cmd.build_lib)

        shutil.copyfile(output, relative_extension)
        mode = os.stat(relative_extension).st_mode
        mode |= (mode & 0o444) >> 2
        os.chmod(relative_extension, mode)


if __name__ == "__main__":
    build()