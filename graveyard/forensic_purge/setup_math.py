from setuptools import setup, Extension
try:
    from Cython.Build import cythonize
except ImportError:
    exit(1)
import numpy as np

extensions = [
    Extension(
        "strategies.math_core",
        ["strategies/math_core.pyx"],
        language="c",
        include_dirs=[np.get_include()],
        extra_compile_args=["-O2"]
    )
]

setup(
    name="MathCore",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)
