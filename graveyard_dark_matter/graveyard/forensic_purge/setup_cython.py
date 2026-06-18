from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Definir la extensión Cython
extensions = [
    Extension(
        "core.c_orderbook",
        ["core/c_orderbook.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"], # Optimización nivel 3
    )
]

setup(
    name="FastLOB",
    ext_modules=cythonize(extensions, annotate=False, compiler_directives={'language_level': "3"}),
)

# Instrucciones de compilación en Windows:
# python setup_cython.py build_ext --inplace
