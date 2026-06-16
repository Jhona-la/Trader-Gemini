"""
MEGA AUDIT PHASE 6: Find ALL commented-out function calls, 
disconnected features, and dead code that SHOULD be alive.
"""
import os
import re

SCAN_DIRS = ['core', 'execution', 'risk', 'strategies']
findings = []

for d in SCAN_DIRS:
    if not os.path.isdir(d):
        continue
    for root, dirs, files in os.walk(d):
        for f in files:
            if not f.endswith('.py'):
                continue
            fp = os.path.join(root, f)
            with open(fp, 'r', encoding='utf-8', errors='replace') as fh:
                lines = fh.readlines()
            
            for i, line in enumerate(lines):
                stripped = line.strip()
                
                # Pattern 1: Commented-out function call followed by live arguments
                if stripped.startswith('#') and '(' in stripped and not stripped.endswith(')'):
                    # Check if next line is an indented argument (live code after comment)
                    if i + 1 < len(lines):
                        next_stripped = lines[i+1].strip()
                        if next_stripped and not next_stripped.startswith('#') and '=' in next_stripped:
                            findings.append(('HALF_COMMENTED', fp, i+1, 
                                f'Comment + live args: {stripped[:60]}'))
                
                # Pattern 2: Commented-out method calls that exist in the codebase
                if stripped.startswith('# self.') and '(' in stripped:
                    method = re.match(r'#\s*self\.(\w+)\(', stripped)
                    if method:
                        findings.append(('COMMENTED_SELF', fp, i+1,
                            f'Commented self call: {stripped[:70]}'))
                
                # Pattern 3: pass in except blocks (swallowed errors)
                if stripped == 'pass' and i > 0:
                    prev = lines[i-1].strip()
                    if prev.startswith('except') and 'Exception' in prev:
                        # Check if there's a logger call nearby
                        has_log = False
                        for j in range(max(0, i-3), min(len(lines), i+3)):
                            if 'logger' in lines[j] or 'print' in lines[j]:
                                has_log = True
                        if not has_log:
                            findings.append(('SILENT_EXCEPT', fp, i+1,
                                f'Silent exception (no logging): {prev[:60]}'))

# Pattern 4: Check for features declared but never called
print("=" * 70)
print("PHASE 6: DISCONNECTED FEATURES & DEAD CODE")
print("=" * 70)

# Check which Portfolio methods are never called from anywhere
portfolio_content = open('core/portfolio.py', 'r', encoding='utf-8').read()
portfolio_methods = re.findall(r'def\s+(\w+)\s*\(self', portfolio_content)

# Search all files for each method
all_code = ''
for d in SCAN_DIRS:
    if not os.path.isdir(d):
        continue
    for root, dirs, files in os.walk(d):
        for f in files:
            if not f.endswith('.py') or f == 'portfolio.py':
                continue
            fp = os.path.join(root, f)
            with open(fp, 'r', encoding='utf-8', errors='replace') as fh:
                all_code += fh.read()

print("\n📊 PORTFOLIO METHODS NEVER CALLED EXTERNALLY:")
for method in sorted(set(portfolio_methods)):
    if method.startswith('_'):
        continue
    if method not in all_code:
        findings.append(('UNCALLED_METHOD', 'core/portfolio.py', 0,
            f'Portfolio.{method}() — never called from outside'))
        print(f"  Portfolio.{method}()")

# Check RiskManager methods never called externally
rm_content = open('risk/risk_manager.py', 'r', encoding='utf-8').read()
rm_methods = re.findall(r'def\s+(\w+)\s*\(self', rm_content)

all_code_no_rm = ''
for d in SCAN_DIRS:
    if not os.path.isdir(d):
        continue
    for root, dirs, files in os.walk(d):
        for f in files:
            if not f.endswith('.py') or f == 'risk_manager.py':
                continue
            fp = os.path.join(root, f)
            with open(fp, 'r', encoding='utf-8', errors='replace') as fh:
                all_code_no_rm += fh.read()

print("\n📊 RISK_MANAGER METHODS NEVER CALLED EXTERNALLY:")
for method in sorted(set(rm_methods)):
    if method.startswith('_'):
        continue
    if method not in all_code_no_rm:
        print(f"  RiskManager.{method}()")

# Print all findings
print(f"\n{'='*70}")
print(f"HALF-COMMENTED CODE (code after comment):")
for cat, fp, line, msg in findings:
    if cat == 'HALF_COMMENTED':
        print(f"  {fp}:{line} — {msg}")

print(f"\nCOMMENTED self.* CALLS:")
for cat, fp, line, msg in findings:
    if cat == 'COMMENTED_SELF':
        print(f"  {fp}:{line} — {msg}")

print(f"\nSILENT EXCEPTIONS (no logging):")
for cat, fp, line, msg in findings:
    if cat == 'SILENT_EXCEPT':
        print(f"  {fp}:{line} — {msg}")

print(f"\n{'='*70}")
print(f"TOTAL FINDINGS: {len(findings)}")
