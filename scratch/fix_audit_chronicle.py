import os
import json
import shutil

file_path = "dashboard/data/futures/audit_chronicle.json"
backup_path = "dashboard/data/futures/audit_chronicle.json.bak"

print(f"Checking if {file_path} exists...")
if not os.path.exists(file_path):
    print("File does not exist.")
    exit(1)

# Create backup
shutil.copyfile(file_path, backup_path)
print(f"Created backup at {backup_path}")

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")
target_idx = 42852  # 0-indexed line 42853
print(f"Line 42853 content: {lines[target_idx].strip()}")

# Replace with correct content
lines[target_idx] = '                            "se\\u00f1al_t\\u00e9cnica": "Precio en media + %B cercano a 0.5 + CVD plano"\n'
print("Replaced line.")

content = "".join(lines)
try:
    data = json.loads(content)
    print("Success! JSON parsed successfully after patch.")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)
    print("Fixed JSON saved.")
except json.JSONDecodeError as e:
    print(f"JSON still invalid: {e}")
    # Print lines around the new error location
    err_line = e.lineno
    print(f"Error line: {err_line}")
    start = max(0, err_line - 5)
    end = min(len(lines), err_line + 5)
    for idx in range(start, end):
        prefix = "-> " if idx + 1 == err_line else "   "
        print(f"{prefix}{idx+1}: {lines[idx].strip()}")
