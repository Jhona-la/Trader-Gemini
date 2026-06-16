from setuptools import setup, Extension
try:
    from Cython.Build import cythonize
except ImportError:
    print("❌ Cython not installed. Install with: pip install cython")
    exit(1)

import numpy as np

extensions = [
    # Extension removed since core/c_orderbook.pyx doesn't exist
    Extension(
        "strategies.math_core",
        ["strategies/math_core.pyx"],
        language="c",
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"]
    ),
    Extension(
        "core.dark_alpha_queue",
        ["core/dark_alpha_queue.pyx"],
        language="c++",
        include_dirs=[np.get_include()],
    ),
    Extension(
        "core.nano_core",
        ["core/nano_core.pyx"],
        language="c",
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"]
    ),
    Extension(
        "core.mev_rbf_engine",
        ["core/mev_rbf_engine.pyx"],
        language="c++",
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"]
    )
]

setup(
    name="TraderGeminiExtensions",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)
