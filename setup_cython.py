from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="Trader Gemini C++ Kernels",
    ext_modules=cythonize([
        "utils/c_math_kernel.pyx",
        "core/fast_event_bus.pyx"
    ], language_level=3, annotate=True),
    include_dirs=[np.get_include()]
)
