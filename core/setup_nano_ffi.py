from setuptools import setup, Extension
from Cython.Build import cythonize
import os

# Define the path to the compiled Rust library
rust_target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'rust_engine', 'target', 'release'))

extensions = [
    Extension(
        "core.cython_bridge.nano_ffi",
        ["core/cython_bridge/nano_ffi.pyx"],
        # Link to the generated Rust import library (.dll.lib on Windows MSVC)
        extra_objects=[os.path.join(rust_target_dir, "quantum_engine.dll.lib")],
        # Or you can use libraries=["quantum_engine"], library_dirs=[rust_target_dir]
        include_dirs=[],
    )
]

setup(
    name="nano_ffi",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"})
)
