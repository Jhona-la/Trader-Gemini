import os
import glob
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define all the ultra-low latency Cython extensions we need
extensions = [
    Extension("core.nano_core", ["core/nano_core.pyx"], language="c++"),
    Extension("core.nano_consensus", ["core/nano_consensus.pyx"]),
    Extension("core.mev_rbf_engine", ["core/mev_rbf_engine.pyx"], language="c++"),
    Extension("core.nano_portfolio", ["core/nano_portfolio.pyx"]),
    Extension("core.nano_regime", ["core/nano_regime.pyx"]),
    Extension("core.dark_alpha_queue", ["core/dark_alpha_queue.pyx"], language="c++"),
]

# Configure cythonize to bypass checks for maximum execution speed
setup(
    name="Trader Gemini Quantum Core",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'initializedcheck': False,
            'nonecheck': False,
        }
    ),
    include_dirs=[numpy.get_include()]
)
