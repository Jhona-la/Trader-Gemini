"""
🔬 MEGA FORENSIC AUDIT LEVEL 2
Scans EVERY .py file for:
1. Phantom computed values (assigned but never read elsewhere)
2. Dead imports (imported but never referenced)
3. Broken cross-references (self.X where X is never set)
4. Feature methods that are defined but never called
5. Metadata fields written to dicts but never consumed
6. Return values computed but caller ignores them
"""
import ast, os, re, sys
from collections import defaultdict

print("=" * 80)
print("🔬 MEGA FORENSIC AUDIT LEVEL 2 — EXHAUSTIVE PHANTOM SCAN")
print("=" * 80)

# Build file inventory
py_files = {}
skip_dirs = {'.venv', '__pycache__', 'node_modules', '.git', 'logs', 'backups'}
for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in skip_dirs]
    for f in files:
        if f.endswith('.py') and not f.startswith('scratch'):
            path = os.path.join(root, f)
            try:
                content = open(path, 'r', encoding='utf-8', errors='ignore').read()
                py_files[path] = content
            except:
                pass

total_content = '\n'.join(py_files.values())
print(f"\n📁 Scanned {len(py_files)} Python files\n")

# =============================================================
# SCAN 1: Methods defined in critical classes but NEVER called
# =============================================================
print("=" * 60)
print("📋 SCAN 1: PHANTOM METHODS (defined but never called)")
print("=" * 60)

critical_classes = {
    'core/engine.py': 'Engine',
    'core/portfolio.py': 'Portfolio', 
    'risk/risk_manager.py': 'RiskManager',
    'strategies/ml_strategy.py': 'MLHybridUltimateStrategyV3',
}

for fpath_suffix, class_name in critical_classes.items():
    fpath = f'./{fpath_suffix}'
    if fpath not in py_files:
        fpath = f'.\\{fpath_suffix}'
    if fpath not in py_files:
        continue
    
    try:
        tree = ast.parse(py_files[fpath])
    except:
        continue
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) 
                       and not n.name.startswith('__')]
            
            phantom_methods = []
            for m in methods:
                # Search for calls to this method across ALL files
                call_patterns = [f'.{m}(', f'.{m} (', f"'{m}'", f'"{m}"']
                found_elsewhere = False
                for p2, c2 in py_files.items():
                    if p2 == fpath:
                        # In same file, check if it's called by OTHER methods (not just defined)
                        # Count occurrences - if only 1 (the def), it's phantom
                        count = c2.count(f'.{m}(') + c2.count(f'.{m} (')
                        if count >= 1:
                            found_elsewhere = True
                            break
                    else:
                        for pat in call_patterns:
                            if pat in c2:
                                found_elsewhere = True
                                break
                    if found_elsewhere:
                        break
                
                if not found_elsewhere:
                    phantom_methods.append(m)
            
            if phantom_methods:
                print(f"\n  🔴 {fpath_suffix}::{class_name}:")
                for pm in phantom_methods[:10]:
                    print(f"    → {pm}()")

# =============================================================
# SCAN 2: Dict keys written in portfolio but never read by RM/Engine
# =============================================================
print("\n" + "=" * 60)
print("📋 SCAN 2: PHANTOM METADATA FIELDS (written → never read)")
print("=" * 60)

port_content = py_files.get('./core/portfolio.py', py_files.get('.\\core\\portfolio.py', ''))
rm_content = py_files.get('./risk/risk_manager.py', py_files.get('.\\risk\\risk_manager.py', ''))
eng_content = py_files.get('./core/engine.py', py_files.get('.\\core\\engine.py', ''))
ml_content = py_files.get('./strategies/ml_strategy.py', py_files.get('.\\strategies\\ml_strategy.py', ''))

# Extract all dict key assignments in portfolio (pos['key'] = or pos.get('key'))
key_writes = set(re.findall(r"pos\['(\w+)'\]\s*=", port_content))
key_writes |= set(re.findall(r"entry_pos\['(\w+)'\]\s*=", port_content))
key_writes |= set(re.findall(r"'(\w+)':\s*", port_content))  # dict literal keys

# Filter to metadata-like keys (not quantity, avg_price, etc.)
base_keys = {'quantity', 'avg_price', 'current_price', 'total_cost', 'leverage',
             'horizon', 'side', 'tp_pct', 'sl_pct', 'hwm', 'lwm', 'open_time'}
metadata_keys = key_writes - base_keys

# Check which are read by consumers
consumers = rm_content + eng_content + ml_content
for key in sorted(metadata_keys):
    read_count = consumers.count(f"'{key}'") + consumers.count(f'"{key}"')
    if read_count == 0 and len(key) > 3:  # Skip short generic keys
        # Verify it's really written as position metadata
        if (f"pos['{key}']" in port_content or f"entry_pos['{key}']" in port_content or
            f"'{key}':" in port_content):
            write_count = port_content.count(f"'{key}'")
            if write_count >= 1:
                print(f"  🔴 '{key}': written {write_count}x in portfolio, 0 reads by RM/Engine/ML")

# =============================================================
# SCAN 3: Computed return values IGNORED by callers
# =============================================================
print("\n" + "=" * 60)
print("📋 SCAN 3: IGNORED RETURN VALUES")
print("=" * 60)

# Check if key methods return values that are never captured
important_returns = {
    'check_stops': ('risk_manager', 'engine'),
    'calculate_realtime_edge': ('risk_manager', 'ml_strategy'),
    'get_market_quality_score': ('liquidity_guardian', 'risk_manager'),
    'analyze': ('sophia', 'ml_strategy'),
    '_check_var_limit': ('risk_manager', 'risk_manager'),
}

for method, (definer, caller) in important_returns.items():
    # Check if the return value is captured in caller
    caller_files = [c for p, c in py_files.items() if caller in p]
    for cf in caller_files:
        # Pattern: bare call without assignment
        bare_calls = re.findall(rf'\n\s+\w+\.{method}\(', cf)
        assigned_calls = re.findall(rf'\n\s+\w+\s*=\s*\w+\.{method}\(', cf)
        if bare_calls and not assigned_calls:
            print(f"  🟡 {method}(): called but return value IGNORED in {caller}")

# =============================================================
# SCAN 4: Self attributes set but NEVER read (per class)
# =============================================================
print("\n" + "=" * 60)
print("📋 SCAN 4: PHANTOM SELF ATTRIBUTES (set but never read)")
print("=" * 60)

for fpath_suffix in ['core/engine.py', 'risk/risk_manager.py', 'strategies/ml_strategy.py']:
    fpath = f'./{fpath_suffix}'
    content = py_files.get(fpath, py_files.get(f'.\\{fpath_suffix}', ''))
    if not content:
        continue
    
    # Find all self.X = assignments
    attr_writes = set(re.findall(r'self\.(\w+)\s*=', content))
    # Find all self.X reads (not followed by =)
    attr_reads = set(re.findall(r'self\.(\w+)(?!\s*=)', content))
    
    # Attributes set but never read in same file
    write_only = attr_writes - attr_reads
    # Filter out common patterns
    ignore = {'__', 'logger', '_lock', 'guard', '_cache_initialized'}
    write_only = {a for a in write_only if not any(a.startswith(i) for i in ignore) and len(a) > 3}
    
    if write_only:
        # Verify if read in OTHER files
        phantom_attrs = []
        for attr in sorted(write_only):
            found = False
            for p2, c2 in py_files.items():
                if p2 != fpath and f'.{attr}' in c2:
                    found = True
                    break
            if not found:
                phantom_attrs.append(attr)
        
        if phantom_attrs:
            print(f"\n  🔴 {fpath_suffix}:")
            for pa in phantom_attrs[:8]:
                print(f"    → self.{pa}")

# =============================================================
# SCAN 5: Cross-module method calls to non-existent methods
# =============================================================
print("\n" + "=" * 60)
print("📋 SCAN 5: BROKEN CROSS-REFERENCES")
print("=" * 60)

# Check key cross-module calls
cross_checks = [
    ('self.portfolio.get_horizon_position', 'core/portfolio.py', 'get_horizon_position'),
    ('self.portfolio.virtual_ledger', 'core/portfolio.py', 'virtual_ledger'),
    ('self.portfolio.update_math_stats', 'core/portfolio.py', 'update_math_stats'),
    ('self.risk_manager.exit_oracle', 'risk/risk_manager.py', 'exit_oracle'),
    ('self.risk_manager.prediction_tracker', 'risk/risk_manager.py', 'prediction_tracker'),
    ('self.portfolio.strategy_performance', 'core/portfolio.py', 'strategy_performance'),
    ('self.sophia.analyze', 'strategies/ml_strategy.py', 'analyze'),
]

for call_pattern, target_file, attr_name in cross_checks:
    target = py_files.get(f'./{target_file}', py_files.get(f'.\\{target_file}', ''))
    if target:
        exists = attr_name in target
        callers = sum(1 for c in py_files.values() if call_pattern in c)
        print(f"  {'✅' if exists else '❌'} {call_pattern}: defined={'YES' if exists else 'NO'}, callers={callers}")

# =============================================================
# SCAN 6: Imports that exist but module never uses the import
# =============================================================
print("\n" + "=" * 60)
print("📋 SCAN 6: DEAD IMPORTS IN CRITICAL FILES")
print("=" * 60)

for fpath_suffix in ['core/engine.py', 'risk/risk_manager.py', 'strategies/ml_strategy.py']:
    fpath = f'./{fpath_suffix}'
    content = py_files.get(fpath, py_files.get(f'.\\{fpath_suffix}', ''))
    if not content:
        continue
    
    try:
        tree = ast.parse(content)
    except:
        continue
    
    dead = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname or alias.name
                # Check if name is used beyond the import
                uses = len(re.findall(rf'\b{name}\b', content))
                if uses <= 1:  # Only the import itself
                    dead.append(f"{name} from {node.module}")
    
    if dead:
        print(f"\n  ⚠️ {fpath_suffix}:")
        for d in dead[:5]:
            print(f"    → {d}")

print("\n" + "=" * 80)
print("🏁 MEGA AUDIT LEVEL 2 COMPLETE")
print("=" * 80)
