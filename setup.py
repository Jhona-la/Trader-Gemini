
from setuptools import setup, Extension
try:
    from Cython.Build import cythonize
except ImportError:
    print("❌ Cython not installed. Install with: pip install cython")
    exit(1)

extensions = [
    Extension(
        "core.c_orderbook",
        ["core/c_orderbook.pyx"],
        language="c++"
    )
]

setup(
    name="TraderGeminiExtensions",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)
