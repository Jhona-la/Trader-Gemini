import sys

filename = 'core/portfolio.py'
with open(filename, 'r', encoding='utf-8') as f:
    lines = f.readlines()

in_docstring = False
docstring_start_line = -1

print("--- EXHAUSTIVE DOCSTRING AUDIT ---")
for i, line in enumerate(lines):
    stripped = line.strip()
    # Check for triple quotes
    count = line.count('\"\"\"')
    
    if count == 1:
        if not in_docstring:
            in_docstring = True
            docstring_start_line = i + 1
            # print(f"DEBUG: Opening docstring at line {docstring_start_line}")
        else:
            in_docstring = False
            # print(f"DEBUG: Closing docstring at line {i+1} (Opened at {docstring_start_line})")
            docstring_start_line = -1
    elif count == 2:
        # Same line or multiple in one line (less common for docstrings)
        if not in_docstring:
            # print(f"DEBUG: Inline docstring at line {i+1}")
            pass
    elif count > 2:
        # Weird case, but possible. Toggling for each triple quote.
        for _ in range(count):
            in_docstring = not in_docstring
            if in_docstring: docstring_start_line = i+1
            else: docstring_start_line = -1

if in_docstring:
    print(f"CRITICAL ERROR: Docstring started at line {docstring_start_line} is NEVER CLOSED.")
    # Show context
    start_idx = max(0, docstring_start_line - 1)
    end_idx = min(len(lines), docstring_start_line + 5)
    print("\n--- CONTEXT AT RUPTURE POINT ---")
    for j in range(start_idx, end_idx):
        print(f"{j+1}: {lines[j].strip()}")
else:
    print("SUCCESS: All triple quotes are syntactically balanced.")
