import os
import sys

# Inject arguments for backtest
sys.argv = ['scripts/run_god_mode_backtest.py', '--days', '1']

with open("scripts/run_god_mode_backtest.py", "r", encoding="utf-8") as f:
    code = f.read()

exec(code, {'__name__': '__main__'})
