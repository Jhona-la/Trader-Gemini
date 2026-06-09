import subprocess
import time
import signal
import os
import sys

def main():
    print("Launching backtest...")
    p = subprocess.Popen([sys.executable, "scratch/temp_backtest_run.py"], 
                         creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    
    print(f"Process PID: {p.pid}. Waiting 75 seconds...")
    time.sleep(75)
    
    if p.poll() is None:
        print("Process is still running. Sending CTRL_C_EVENT to dump traceback...")
        try:
            os.kill(p.pid, signal.CTRL_C_EVENT)
        except AttributeError:
            os.kill(p.pid, signal.SIGINT)
        
        # Wait a bit for faulthandler to print
        time.sleep(5)
        
        if p.poll() is None:
            print("Process still alive, killing forcefully.")
            p.kill()
    else:
        print(f"Process finished early with code {p.returncode}.")

if __name__ == "__main__":
    main()
