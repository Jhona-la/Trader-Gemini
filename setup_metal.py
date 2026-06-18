from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("core.metal.technical_fast", ["core/metal/technical_fast.pyx"]),
    Extension("core.dark_alpha_queue", ["core/dark_alpha_queue.pyx"], language="c++"),
    Extension("data.fast_lob", ["data/fast_lob.pyx"]),
    Extension("core.nano_portfolio", ["core/nano_portfolio.pyx"]),
    Extension("core.metal.macro_engine", ["core/metal/macro_engine.pyx"], language="c++")
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"}
    ),
    include_dirs=[numpy.get_include()]
)
