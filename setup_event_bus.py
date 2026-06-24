from setuptools import setup
from Cython.Build import cythonize
import os

setup(
    name='fast_event_bus',
    ext_modules=cythonize(
        ["core/fast_event_bus.pyx"],
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'nonecheck': False,
            'cdivision': True,
        }
    ),
)
