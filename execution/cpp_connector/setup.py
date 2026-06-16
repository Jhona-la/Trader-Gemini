from setuptools import setup, Extension
from Cython.Build import cythonize
import os

# ═══════════════════════════════════════════════════════════════
# ⚙️ BUILD SCRIPT PARA COMPONENTE C++ NATIVO
# Ejecutar: python setup.py build_ext --inplace
# Requiere: Microsoft Visual Studio Build Tools (Windows) o GCC (Linux)
# ═══════════════════════════════════════════════════════════════

extensions = [
    Extension(
        name="cpp_executor_wrapper",
        sources=["cpp_executor_wrapper.pyx"],
        language="c++",
        extra_compile_args=["/std:c++17", "/O2"] if os.name == 'nt' else ["-std=c++17", "-O3"],
    )
]

setup(
    name="FastBinanceSocket",
    ext_modules=cythonize(extensions, language_level=3)
)
