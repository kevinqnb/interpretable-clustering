from setuptools import setup, Extension
from Cython.Build import cythonize
from pathlib import Path
import numpy as np

src_dir = Path("src/intercluster")
extensions = []

for pyx_file in src_dir.rglob("*.pyx"):
    # Convert path to module name: src/my_project/utils/helpers.pyx -> my_project.utils.helpers
    relative_path = pyx_file.relative_to("src")
    module_name = str(relative_path.with_suffix("")).replace("/", ".")
    
    extensions.append(
        Extension(
            module_name,
            [str(pyx_file)],
            include_dirs=[np.get_include()]
        )
    )

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"}
    )
)