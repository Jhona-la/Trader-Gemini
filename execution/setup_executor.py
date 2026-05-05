from setuptools import setup
from Cython.Build import cythonize
import os

# Ensure the build happens in the correct directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

setup(
    name='c_executor',
    ext_modules=cythonize("c_executor.pyx", compiler_directives={'language_level': "3"}),
    zip_safe=False,
)
