import cProfile
import pstats
import sys

from scripts.run_god_mode_backtest import main

# Patch sys.argv to run for 1 day, but we'll modify the God Mode to stop early if we could.
# Since we just want to run it shortly, we'll patch the BacktestDataProvider or just let it run 20 epochs.

if __name__ == '__main__':
    sys.argv = ['run_god_mode_backtest.py', '--days', '0.01', '--symbol', 'BTC/USDT']
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        main()
    except Exception as e:
        print("Error:", e)
    
    profiler.disable()
    
    with open("god_mode_profile.txt", "w") as f:
        ps = pstats.Stats(profiler, stream=f).sort_stats('tottime')
        ps.print_stats()
