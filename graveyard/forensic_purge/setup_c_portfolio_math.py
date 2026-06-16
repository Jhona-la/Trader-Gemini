from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "c_portfolio_math",
        ["core/c_portfolio_math.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native", "-ffast-math"]
    )
]

setup(
    name="c_portfolio_math",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"})
)
