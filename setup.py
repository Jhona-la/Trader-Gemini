from setuptools import setup, Extension
try:
    from Cython.Build import cythonize
except ImportError:
    print("❌ Cython not installed. Install with: pip install cython")
    exit(1)

import numpy as np

extensions = [
    Extension(
        "core.c_orderbook",
        ["core/c_orderbook.pyx"],
        language="c++",
        include_dirs=[np.get_include()]
    )
]

setup(
    name="TraderGeminiExtensions",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)
