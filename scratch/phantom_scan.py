"""DEEP PHANTOM SCANNER v3 — Full Project Sweep"""
import re, os, ast

print("=" * 70)
print("DEEP PHANTOM SCAN v3: FULL PROJECT")
print("=" * 70)

# ===========================================================
# 1. PORTFOLIO.PY: Phantom methods, dead checks
# ===========================================================
print("\n=== 1. PORTFOLIO.PY: Phantom References ===")
with open('core/portfolio.py', 'r', encoding='utf-8') as f:
    port_content = f.read()
    port_lines = port_content.split('\n')

# Find hasattr phantoms
for i, line in enumerate(port_lines):
    m = re.search(r"hasattr\(self,\s*['\"](\w+)['\"]\)", line)
    if m:
        attr = m.group(1)
        assigned = any(f"self.{attr} =" in l or f"self.{attr}=" in l for l in port_lines)
        if not assigned:
            print(f"  L{i+1}: self.{attr} — 🔇 NEVER ASSIGNED")

# Find methods defined but never called internally
port_methods = {}
for i, line in enumerate(port_lines):
    m = re.match(r'\s+def (\w+)\(self', line)
    if m and not m.group(1).startswith('_'):
        name = m.group(1)
        uses = port_content.count(f'self.{name}(') + port_content.count(f'.{name}(')
        if uses <= 1:  # only the definition
            port_methods[name] = i + 1

if port_methods:
    print(f"\n  Methods possibly never called internally:")
    for name, ln in list(port_methods.items())[:10]:
        print(f"    L{ln}: {name}()")

# ===========================================================
# 2. FEATURE ENGINEERING: Features computed but never used
# ===========================================================
print("\n=== 2. FEATURE ENGINEERING: Computed but unused ===")
fe_files = []
for root, dirs, files in os.walk('.'):
    if '.venv' in root or '__pycache__' in root or '.git' in root:
        continue
    for f in files:
        if 'feature' in f.lower() and f.endswith('.py'):
            fe_files.append(os.path.join(root, f))

for fp in fe_files:
    print(f"\n  File: {fp}")
    with open(fp, 'r', encoding='utf-8') as f:
        fe_content = f.read()
    
    # Find all df['xxx'] = assignments (feature creation)
    created_features = set()
    for m in re.finditer(r"df\[(?:'|\")(\w+)(?:'|\")\]\s*=", fe_content):
        created_features.add(m.group(1))
    
    # Check which are in the top_20 training list
    top20 = ['returns_5','returns_10','roc_10','rsi_14','atr_pct',
             'macd_hist','bb_position','bb_width','stoch_k','adx',
             'volume_ratio','gk_vol','hurst_memory','volatility_ransac',
             'micro_imbalance','spread_squeeze','scalp_velocity_1',
             'scalp_rsi_divergence','micro_label','market_cluster']
    
    used = [f for f in created_features if f in top20]
    unused = [f for f in created_features if f not in top20 
              and f not in ['label','open','high','low','close','volume','timestamp','datetime']]
    
    print(f"    Created: {len(created_features)} | Used in training: {len(used)} | NOT used: {len(unused)}")
    if unused:
        for f in sorted(unused)[:15]:
            print(f"      🔇 {f}")
        if len(unused) > 15:
            print(f"      ... and {len(unused)-15} more")

# ===========================================================
# 3. SOPHIA MODULES: Dead methods
# ===========================================================
print("\n=== 3. SOPHIA MODULES: Connection Audit ===")
sophia_files = [f for f in os.listdir('sophia') if f.endswith('.py') and f != '__init__.py']
for sf in sophia_files:
    fp = os.path.join('sophia', sf)
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Count references from OTHER files
    ref_count = 0
    for root, dirs, files in os.walk('.'):
        if '.venv' in root or '__pycache__' in root or '.git' in root or 'sophia' in root:
            continue
        for f in files:
            if f.endswith('.py'):
                try:
                    with open(os.path.join(root, f), 'r', encoding='utf-8') as fh:
                        other = fh.read()
                    module_name = sf.replace('.py', '')
                    if module_name in other:
                        ref_count += 1
                except:
                    pass
    
    status = "✅ CONNECTED" if ref_count > 0 else "🔇 ORPHAN"
    print(f"  {sf}: {ref_count} external refs ({status})")

# ===========================================================
# 4. CONFIG: Referenced but missing attributes
# ===========================================================
print("\n=== 4. CONFIG: Phantom Config References ===")
with open('config.py', 'r', encoding='utf-8') as f:
    config_content = f.read()

# Find all Config.xxx references across project
config_refs = {}
for root, dirs, files in os.walk('.'):
    if '.venv' in root or '__pycache__' in root or '.git' in root:
        continue
    for f in files:
        if f.endswith('.py') and f != 'config.py':
            try:
                with open(os.path.join(root, f), 'r', encoding='utf-8') as fh:
                    content = fh.read()
                for m in re.finditer(r'Config\.(\w+)\.(\w+)', content):
                    key = f"Config.{m.group(1)}.{m.group(2)}"
                    if key not in config_refs:
                        config_refs[key] = []
                    config_refs[key].append(os.path.join(root, f))
            except:
                pass

# Check if they exist in config.py
for key, files in sorted(config_refs.items()):
    parts = key.split('.')
    attr = parts[-1]
    if attr not in config_content:
        print(f"  🔇 {key} — referenced in {len(files)} files but NOT FOUND in config.py")
        for fp in files[:2]:
            print(f"      used in: {fp}")

# ===========================================================
# 5. DATA PROVIDER / HANDLER: Phantom method calls
# ===========================================================
print("\n=== 5. DATA PROVIDER: Phantom Methods ===")
dp_files = ['core/data_provider.py', 'core/data_handler.py']
for dpf in dp_files:
    if not os.path.exists(dpf):
        continue
    with open(dpf, 'r', encoding='utf-8') as f:
        dp_content = f.read()
        dp_lines = dp_content.split('\n')
    
    # Find defined methods
    defined = set()
    for i, line in enumerate(dp_lines):
        m = re.match(r'\s+def (\w+)\(self', line)
        if m:
            defined.add(m.group(1))
    
    # Find all calls to data_provider.xxx across project
    called = set()
    for root, dirs, files in os.walk('.'):
        if '.venv' in root or '__pycache__' in root or '.git' in root:
            continue
        for f in files:
            if f.endswith('.py'):
                try:
                    with open(os.path.join(root, f), 'r', encoding='utf-8') as fh:
                        content = fh.read()
                    for m in re.finditer(r'data_(?:provider|handler)\.(\w+)\(', content):
                        called.add(m.group(1))
                except:
                    pass
    
    phantom_calls = called - defined
    if phantom_calls:
        print(f"  {dpf}: Methods CALLED but NOT DEFINED:")
        for pc in sorted(phantom_calls):
            if pc not in ['get', 'items', 'values']:
                print(f"    🔇 {pc}()")

print("\n=== SCAN COMPLETE ===")
