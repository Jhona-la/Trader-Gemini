import os
import glob
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import Cython.Compiler.Options

# Optimize Cython compilation
Cython.Compiler.Options.annotate = False
Cython.Compiler.Options.docstrings = False
Cython.Compiler.Options.fast_fail = True

# Target directories to compile
TARGET_DIRS = [
    "core",
    "risk",
    "strategies",
    "strategies/nano_technical.pyx",
    "execution",
    "utils",
    "analysis"
]

EXCLUDE_FILES = [
    "__init__.py",
    "atomic_guard.py",
    "evolution_kernels.py",
    "feature_engineering.py",
    "first_breath.py",
    "fused_strategy_kernel.py",
    "hft_buffer.py",
    "kinematic_strategy.py",
    "main.py",
    "market_regime_hmm.py",
    "math_kernel.py",
    "nano_backtester.py",
    "nano_core.py",
    "nano_risk_engine.py",
    "nano_stop_checker.py",
    "online_learning_kernels.py",
    "phalanx.py",
    "quant_math.py",
    "quantum_engine.py",
    "rl_buffer.py",
    "setup.py",
    "setup_compiler.py",
    "simulation_numba.py",
    "statistical.py",
    "statistics_pro.py",
    "structure.py",
    "token_bucket.py",
    "trend.py",
    "vectorized_backtest.py",
    "xgboost_compiler.py"
]

def get_ext_paths():
    extensions = []
    for directory in TARGET_DIRS:
        # Recursive glob for all .py and .pyx files
        search_pattern_py = os.path.join(directory, "**", "*.py")
        search_pattern_pyx = os.path.join(directory, "**", "*.pyx")
        
        all_files = glob.glob(search_pattern_py, recursive=True) + glob.glob(search_pattern_pyx, recursive=True)
        for filepath in all_files:
            filename = os.path.basename(filepath)
            
            # Skip excluded files
            if filename in EXCLUDE_FILES:
                continue
                
            # Convert file path to module name format
            # e.g. "core/engine.py" -> "core.engine"
            module_path = os.path.splitext(filepath)[0]
            module_name = module_path.replace(os.path.sep, ".")
            
            print(f"[CYTHONIZER] Prepared for compilation: {module_name} -> {filepath}")
            
            # Special case for hyper_kernel which needs physics_engine.c
            sources = [filepath]
            if module_name == "core.hyper_kernel":
                sources.append("core/physics_engine.c")
                
            extensions.append(
                Extension(
                    name=module_name,
                    sources=sources,
                    extra_compile_args=["/O2", "/fp:fast", "/GL"] if os.name == 'nt' else ["-O3", "-ffast-math"],
                    extra_link_args=["/LTCG"] if os.name == 'nt' else [],
                )
            )
    return extensions

extensions = get_ext_paths()

setup(
    name="TraderGemini_AOT",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,       # Turn off bounds checking for speed
            'wraparound': False,        # Turn off negative index wrapping
            'cdivision': True,          # Turn off divide-by-zero checks
            'initializedcheck': False,  # Turn off uninitialized attribute checks
            'nonecheck': False,         # Turn off none checks
        },
        nthreads=0, # Disable multithreaded compilation on Windows to prevent BrokenProcessPool
        build_dir="build_cython" # Keep root clean from .c files
    ),
    zip_safe=False,
)
