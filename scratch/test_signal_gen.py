"""Quick test: does generate_signals work as sync function?"""
import os, sys, inspect
os.environ['TRADER_GEMINI_BACKTEST'] = 'true'
sys.path.insert(0, '.')

from strategies.technical import HybridScalpingStrategy

# Check the function signature
method = HybridScalpingStrategy.generate_signals
print(f"generate_signals is coroutine: {inspect.iscoroutinefunction(method)}")
print(f"generate_signals type: {type(method)}")

calc_method = HybridScalpingStrategy.calculate_signals
print(f"calculate_signals source:")
import dis
dis.dis(calc_method)
