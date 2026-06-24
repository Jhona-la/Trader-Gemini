from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

extensions = [
    Extension(
        "utils.c_math_kernel",
        ["utils/c_math_kernel.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["/O2", "/fp:fast", "/GL"] if os.name == 'nt' else ["-O3", "-ffast-math"],
        extra_link_args=["/LTCG"] if os.name == 'nt' else [],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
    )
]

setup(
    name="c_math_kernel",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'nonecheck': False,
        }
    ),
    zip_safe=False,
)
