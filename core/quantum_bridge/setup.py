from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import platform

# For Windows we need to link against the compiled quantum_engine.dll / .lib
# Assume the rust library was built and output to core/rust_engine/target/release/
is_windows = platform.system() == "Windows"
lib_name = "quantum_engine"
lib_dirs = ["../rust_engine/target/release"]

extensions = [
    Extension(
        "quantum_bridge",
        sources=["quantum_bridge.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=lib_dirs,
        libraries=[lib_name, "ws2_32", "userenv", "ntdll", "advapi32", "bcrypt"],
        # Add necessary compiler flags here if needed
    )
]

setup(
    name="quantum_bridge",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)
