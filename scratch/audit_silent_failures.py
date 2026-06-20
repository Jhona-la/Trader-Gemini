import os
import re
import json

root_dir = r"c:\Users\jhona\Documents\Proyectos\Trader Gemini"
target_exts = {".py", ".pyx", ".c", ".cpp", ".rs"}

def scan_files():
    results = {
        "silent_get": [],
        "return_0_0": [],
        "except_pass": [],
        "time_sleep": []
    }
    
    get_pattern = re.compile(r'\.get\([^,]+,\s*[^\)]+\)')
    return_0_pattern = re.compile(r'return\s+0\.0')
    except_pass_pattern = re.compile(r'except(\s+Exception(\s+as\s+\w+)?)?:\s*pass')
    time_sleep_pattern = re.compile(r'time\.sleep')
    
    for root, dirs, files in os.walk(root_dir):
        if any(ignored in root for ignored in ['.git', '.venv', 'build', '__pycache__', 'scratch', 'graveyard_dark_matter', 'graveyard', 'compiled_core', '.vscode', '.models']):
            continue
            
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in target_exts:
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        
                    for i, line in enumerate(lines):
                        line_num = i + 1
                        if get_pattern.search(line):
                            results["silent_get"].append({"file": filepath, "line": line_num, "content": line.strip()})
                        if return_0_pattern.search(line):
                            results["return_0_0"].append({"file": filepath, "line": line_num, "content": line.strip()})
                        if except_pass_pattern.search(line):
                            results["except_pass"].append({"file": filepath, "line": line_num, "content": line.strip()})
                        if time_sleep_pattern.search(line):
                            results["time_sleep"].append({"file": filepath, "line": line_num, "content": line.strip()})
                except Exception as e:
                    pass

    with open(os.path.join(root_dir, "scratch", "silent_failures.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("Silent failures scan completed.")

if __name__ == "__main__":
    scan_files()
