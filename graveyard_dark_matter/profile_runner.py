import cProfile
import pstats
import os
import sys

# Import the backtest
sys.path.insert(0, r"c:\Users\jhona\Documents\Proyectos\Trader Gemini")
import scratch_run_local_backtest

# Profiling is wrapped inside the script itself now, or we can just run it.
# Actually scratch_run_local_backtest.py executes when imported.
