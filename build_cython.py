import os
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Find all .pyx files in the project directories
dirs_to_search = ['core', 'execution', 'risk', 'strategies', 'utils', 'data']
pyx_files = []

for d in dirs_to_search:
    for root, dirs, files in os.walk(d):
        for file in files:
            if file.endswith('.pyx'):
                # Avoid cython bridge or c++ wrapper right now if they require specific C++ headers not present
                if 'cpp' not in file:
                    pyx_files.append(os.path.join(root, file))

print(f"Encontrados {len(pyx_files)} archivos Cython para compilar a binario C:")
for p in pyx_files:
    print(f" - {p}")

extensions = []
for pyx in pyx_files:
    # Generar el nombre del modulo: core.hyper_kernel
    module_name = pyx.replace('.pyx', '').replace(os.path.sep, '.')
    sources = [pyx]
    if 'binance_executor_c.pyx' in pyx:
        sources.append(os.path.join(os.path.dirname(pyx), 'cpp_executor.cpp'))
    elif 'hyper_kernel.pyx' in pyx:
        sources.append(os.path.join(os.path.dirname(pyx), 'physics_engine.cpp'))

    ext = Extension(
        module_name,
        sources,
        include_dirs=[np.get_include(), os.path.join('core', 'metal'), 'execution'],
        language='c++',
        extra_compile_args=['-O3', '-ffast-math', '-march=native'] if os.name != 'nt' else ['/O2', '/fp:fast', '/arch:AVX2']
    )
    extensions.append(ext)

setup(
    name='TraderGeminiNanoCore',
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3", 'boundscheck': False, 'wraparound': False, 'cdivision': True}
    ),
    script_args=['build_ext', '--inplace']
)
