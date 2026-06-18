from setuptools import setup, Extension
try:
    from Cython.Build import cythonize
except ImportError:
    exit(1)
import numpy as np

extensions = [
    Extension(
        "risk.c_risk",
        ["risk/c_risk.pyx"],
        language="c",
        include_dirs=[np.get_include()],
        extra_compile_args=["-O2"]
    ),
    Extension(
        "core.c_portfolio",
        ["core/c_portfolio.pyx"],
        language="c",
        include_dirs=[np.get_include()],
        extra_compile_args=["-O2"]
    ),
    Extension(
        "core.c_queue",
        ["core/c_queue.pyx"],
        language="c",
        include_dirs=[np.get_include()],
        extra_compile_args=["-O2"]
    )
]

setup(
    name="GlobalNanoCore",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)
