import sys
import cProfile
import pstats
import time
import runpy

sys.argv = ['run_god_mode_backtest.py', '--days', '3']

def run_profile():
    pr = cProfile.Profile()
    pr.enable()
    
    start = time.time()
    def trace_calls(frame, event, arg):
        if time.time() - start > 90: # 90 seconds timeout
            raise KeyboardInterrupt("KILL SWITCH HIT")
        return trace_calls
        
    try:
        sys.settrace(trace_calls)
        runpy.run_path('scripts/run_god_mode_backtest.py', run_name='__main__')
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sys.settrace(None)
        pr.disable()
        ps = pstats.Stats(pr).sort_stats('tottime')
        ps.print_stats(30)

if __name__ == '__main__':
    run_profile()
