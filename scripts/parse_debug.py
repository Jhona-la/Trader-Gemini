import re

def analyze_log():
    with open('debug_stdout.txt', 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        if 'Dur: 60s' in line:
            print(f"Line {i}: {line.strip()}")
            
if __name__ == '__main__':
    analyze_log()
