import psutil
import os
import sys

target = "main.py"
current_dir = os.getcwd().lower()
print(f"Current Directory (Me): {current_dir}")
print("-" * 30)

found_any = False
for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cwd']):
    try:
        info = proc.info
        cmd = info.get('cmdline')
        if not cmd: continue
        
        cmd_str = " ".join(cmd).lower()
        if target in cmd_str and "supervisor" not in cmd_str:
            found_any = True
            print(f"Found Candidate: PID {info['pid']}")
            print(f"  Name: {info['name']}")
            print(f"  CMD: {cmd}")
            print(f"  CWD (Raw): {info['cwd']}")
            
            proc_cwd = (info.get('cwd') or '').lower()
            
            # Test Logic
            match_a = (proc_cwd and proc_cwd == current_dir)
            match_b = (proc_cwd and current_dir in proc_cwd)
            match_c = (current_dir in cmd_str)
            
            print(f"  Match A (Exact CWD): {match_a}")
            print(f"  Match B (Subdir):    {match_b}")
            print(f"  Match C (CmdPath):   {match_c}")
            
            if match_a or match_b or match_c:
                print("  ✅ VERDICT: MATCH!")
            else:
                print("  ❌ VERDICT: NO MATCH")
            print("-" * 30)
            
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        print(f"  ❌ Access Denied to PID {proc.pid}")

if not found_any:
    print("❌ No 'main.py' processes found at all.")
