from setuptools import setup, Extension
from Cython.Build import cythonize
import os

# Arch flags for max performance
extra_compile_args = ['-O3']
if os.name == 'nt':  # Windows
    extra_compile_args.append('/arch:AVX2')
else:
    extra_compile_args.append('-mavx2')

extensions = [
    Extension(
        "core.hyper_kernel",
        sources=["core/hyper_kernel.pyx", "core/physics_engine.c"],
        include_dirs=["core"],
        extra_compile_args=extra_compile_args
    )
]

setup(
    name="Quantum Nano HyperKernel",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"})
)
