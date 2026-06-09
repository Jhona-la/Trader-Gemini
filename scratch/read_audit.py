import os

def find_mentions(file_path, keywords):
    print(f"=== Mentions in {os.path.basename(file_path)} ===")
    if not os.path.exists(file_path):
        print("File does not exist.")
        return
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        for kw in keywords:
            if kw.lower() in line.lower():
                print(f"Line {i+1}: {line}")
                # Print a few lines around it
                start = max(0, i - 5)
                end = min(len(lines), i + 10)
                print("\n".join(lines[start:end]))
                print("-" * 50)
                break

scratch_dir = r"C:\Users\jhona\.gemini\antigravity\brain\6b0bf5e2-4c5f-42eb-9c0a-cf861eb08d00\scratch"
strategy_report = os.path.join(scratch_dir, "strategy_auditor_report.txt")
core_report = os.path.join(scratch_dir, "core_auditor_report.txt")

find_mentions(strategy_report, ["1065", "confluence_score", "setup_type"])
find_mentions(core_report, ["1065", "confluence_score"])
