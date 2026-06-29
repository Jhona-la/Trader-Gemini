from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

rust_lib_dir = os.path.abspath(os.path.join("..", "rust_engine", "target", "release"))

ext = Extension(
    name="nano_ffi",
    sources=["nano_ffi.pyx"],
    include_dirs=[np.get_include()],
    library_dirs=[rust_lib_dir],
    libraries=["quantum_engine.dll"]
)

setup(
    name='nano_ffi',
    ext_modules=cythonize([ext], compiler_directives={'language_level': "3"})
)
