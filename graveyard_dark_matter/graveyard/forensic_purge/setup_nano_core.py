import os
from setuptools import setup, Extension
from Cython.Build import cythonize

# Unified Nano-Core Extension
extensions = [
    Extension(
        "core.nano_core",
        ["core/nano_core.pyx"],
        extra_compile_args=['-O3', '-ffast-math', '-march=native'] if os.name != 'nt' else ['/O2', '/fp:fast', '/arch:AVX2'],
    )
]

setup(
    name="Nano Core",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)
