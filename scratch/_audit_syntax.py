"""Deep Syntax Audit - Finds ALL syntax errors in the project."""
import py_compile
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = {'.venv', '__pycache__', '.git', 'node_modules', 'build', '.agents', '.agent', 'wandb', '.ruff_cache', '.pytest_cache', 'antigravity_export', 'archive'}

errors = []
total = 0

for root, dirs, files in os.walk(ROOT):
    dirs[:] = [d for d in dirs if d not in SKIP]
    for f in files:
        if not f.endswith('.py'):
            continue
        filepath = os.path.join(root, f)
        total += 1
        try:
            py_compile.compile(filepath, doraise=True)
        except py_compile.PyCompileError as e:
            errors.append((filepath, str(e)))

print(f"\n{'='*60}")
print(f"SYNTAX AUDIT: Scanned {total} Python files")
print(f"{'='*60}")
if errors:
    print(f"\n❌ FOUND {len(errors)} SYNTAX ERRORS:\n")
    for path, err in errors:
        rel = os.path.relpath(path, ROOT)
        print(f"  🐛 {rel}")
        # Extract just the error line
        for line in str(err).split('\n'):
            if 'SyntaxError' in line or 'line ' in line:
                print(f"     → {line.strip()}")
        print()
else:
    print("\n✅ ALL FILES PASS SYNTAX CHECK\n")
