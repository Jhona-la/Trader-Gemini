import cProfile
import pstats
import threading
import sys
import tracemalloc
import time
import os

def kill_and_dump(profiler):
    print("\n\n[KILL-SWITCH] 60 SECONDS TIMEOUT REACHED! ACTIVATING FORENSIC AUTOPSY...")
    profiler.disable()
    
    # Dump cProfile stats
    stats = pstats.Stats(profiler)
    stats.dump_stats('scratch_autopsy.prof')
    with open('scratch_autopsy_stats_cumtime.txt', 'w') as f:
        stats.sort_stats('cumtime').print_stats(50)
    with open('scratch_autopsy_stats_tottime.txt', 'w') as f:
        stats.sort_stats('tottime').print_stats(50)
    
    # Dump memory stats
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    with open('scratch_autopsy_memory.txt', 'w') as f:
        f.write("[ Top 20 memory allocations ]\n")
        for stat in top_stats[:20]:
            f.write(str(stat) + "\n")
            
    print("Forensic autopsy saved. Terminating process.")
    os._exit(1)

if __name__ == "__main__":
    print("[AUTOPSY RUNNER] Starting tracemalloc and cProfile...")
    tracemalloc.start()
    profiler = cProfile.Profile()
    
    timer = threading.Timer(60.0, kill_and_dump, args=[profiler])
    timer.daemon = True
    timer.start()
    
    profiler.enable()
    start_time = time.time()
    try:
        # Patch sys.argv
        sys.argv = ['scripts/run_god_mode_backtest.py', '--days', '1']
        with open('scripts/run_god_mode_backtest.py', 'r', encoding='utf-8') as f:
            code = compile(f.read(), 'scripts/run_god_mode_backtest.py', 'exec')
            exec(code, {'__name__': '__main__', '__file__': 'scripts/run_god_mode_backtest.py'})
    except Exception as e:
        print(f"\n[AUTOPSY RUNNER] Exception caught: {e}")
    finally:
        timer.cancel()
        profiler.disable()
        elapsed = time.time() - start_time
        print(f"\n[AUTOPSY RUNNER] Process completed naturally in {elapsed:.2f}s before timeout.")
        
        # Dump stats normally
        stats = pstats.Stats(profiler)
        stats.dump_stats('scratch_autopsy.prof')
        with open('scratch_autopsy_stats_cumtime.txt', 'w') as f:
            stats.sort_stats('cumtime').print_stats(50)
        with open('scratch_autopsy_stats_tottime.txt', 'w') as f:
            stats.sort_stats('tottime').print_stats(50)
            
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        with open('scratch_autopsy_memory.txt', 'w') as f:
            f.write("[ Top 20 memory allocations ]\n")
            for stat in top_stats[:20]:
                f.write(str(stat) + "\n")
