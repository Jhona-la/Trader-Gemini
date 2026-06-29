"""
AUDITORÍA HOLOGRÁFICA SISTÉMICA - Trader Gemini
Phase I: Censo Featureal + Phase II: Topología + Phase III: Paridad
"""
import os
import re
import json
import sys

PROJECT_ROOT = r"C:\Users\jhona\Documents\Proyectos\Trader Gemini"

# ═══════════════════════════════════════════════════════════════
# PHASE I: CENSO FEATUREAL — Extraer TODAS las features del sistema
# ═══════════════════════════════════════════════════════════════

def scan_feature_engineering():
    """Scan feature_engineering.py for all generated features"""
    fe_path = os.path.join(PROJECT_ROOT, "strategies", "components", "feature_engineering.py")
    features = {}
    with open(fe_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all .alias('xxx') patterns → these are feature names
    aliases = re.findall(r"\.alias\(['\"](\w+)['\"]\)", content)
    for a in aliases:
        features[a] = {
            'origin': 'feature_engineering.py',
            'state': 'VIVA' if 'pl.lit(0' not in content.split(f".alias('{a}')")[0].split('\n')[-1] else 'MUERTA_HARDCODED_ZERO',
        }
    return features, aliases

def scan_indicator_modules():
    """Scan indicators/ for all calculated features"""
    ind_dir = os.path.join(PROJECT_ROOT, "strategies", "indicators")
    features = {}
    if not os.path.isdir(ind_dir):
        return features
    for fname in os.listdir(ind_dir):
        if fname.endswith('.py') and fname != '__init__.py':
            fpath = os.path.join(ind_dir, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
            # Find returned dict keys like 'ema_5', 'rsi_14', etc.
            keys = re.findall(r"['\"](\w+)['\"]:\s*(?:np\.|pl\.|result|values|arr)", content)
            for k in keys:
                features[k] = {'origin': f'indicators/{fname}', 'state': 'VIVA'}
    return features

def scan_hardcoded_zero_features(all_features):
    """Identify features hardcoded to pl.lit(0.0)"""
    fe_path = os.path.join(PROJECT_ROOT, "strategies", "components", "feature_engineering.py")
    with open(fe_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    zero_features = []
    for line in lines:
        m = re.search(r"pl\.lit\(0(?:\.0)?\)\.alias\(['\"](\w+)['\"]\)", line)
        if m:
            zero_features.append(m.group(1))
    return zero_features

# ═══════════════════════════════════════════════════════════════
# PHASE II: TOPOLOGÍA — Mapeo de imports y dependencias
# ═══════════════════════════════════════════════════════════════

def scan_silent_fallbacks():
    """Find try/except ImportError that silently fall back"""
    results = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Skip non-source directories
        if any(skip in root for skip in ['.venv', '__pycache__', '.git', 'graveyard', 'node_modules', '.models']):
            continue
        for f in files:
            if not f.endswith('.py'):
                continue
            fpath = os.path.join(root, f)
            try:
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as fp:
                    content = fp.read()
                # Find try/except ImportError blocks
                imports = re.findall(r'except\s+(?:ImportError|ModuleNotFoundError|Exception)\s*(?:as\s+\w+)?:\s*\n\s*(\w+)\s*=\s*None', content)
                for imp in imports:
                    results.append({
                        'file': os.path.relpath(fpath, PROJECT_ROOT),
                        'variable': imp,
                        'type': 'SILENT_FALLBACK_TO_NONE'
                    })
            except:
                pass
    return results

def scan_silent_gets():
    """Find dict.get() in hot-path files"""
    hot_files = [
        'core/engine.py', 'strategies/ml_strategy.py', 'strategies/technical.py',
        'core/signal_scorer.py', 'risk/risk_manager.py', 'core/portfolio.py',
        'strategies/components/feature_engineering.py'
    ]
    results = []
    for rel_path in hot_files:
        fpath = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.exists(fpath):
            continue
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f, 1):
                if '.get(' in line and 'self' not in line[:20]:
                    # Skip comments
                    stripped = line.strip()
                    if stripped.startswith('#'):
                        continue
                    results.append({
                        'file': rel_path,
                        'line': i,
                        'code': stripped[:120]
                    })
    return results

def scan_return_zero():
    """Find return 0.0 in except blocks"""
    results = []
    hot_dirs = ['core', 'strategies', 'risk', 'execution', 'sophia']
    for d in hot_dirs:
        dirpath = os.path.join(PROJECT_ROOT, d)
        if not os.path.isdir(dirpath):
            continue
        for root, _, files in os.walk(dirpath):
            if '__pycache__' in root:
                continue
            for f in files:
                if not f.endswith('.py'):
                    continue
                fpath = os.path.join(root, f)
                try:
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as fp:
                        lines = fp.readlines()
                    in_except = False
                    for i, line in enumerate(lines, 1):
                        if 'except' in line and ':' in line:
                            in_except = True
                        elif in_except and ('return 0.0' in line or 'return 0' in line or 'return None' in line):
                            results.append({
                                'file': os.path.relpath(fpath, PROJECT_ROOT),
                                'line': i,
                                'code': line.strip()[:100]
                            })
                            in_except = False
                        elif line.strip() and not line.strip().startswith('#') and in_except:
                            if not line.strip().startswith('pass') and not line.strip().startswith('logger'):
                                in_except = False
                except:
                    pass
    return results

def scan_time_sleep_in_async():
    """Find time.sleep() in async functions"""
    results = []
    for root, _, files in os.walk(PROJECT_ROOT):
        if any(skip in root for skip in ['.venv', '__pycache__', '.git', 'graveyard']):
            continue
        for f in files:
            if not f.endswith('.py'):
                continue
            fpath = os.path.join(root, f)
            try:
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as fp:
                    lines = fp.readlines()
                in_async = False
                for i, line in enumerate(lines, 1):
                    if 'async def' in line:
                        in_async = True
                    elif line.strip().startswith('def ') and 'async' not in line:
                        in_async = False
                    if in_async and 'time.sleep(' in line:
                        results.append({
                            'file': os.path.relpath(fpath, PROJECT_ROOT),
                            'line': i,
                            'code': line.strip()[:100]
                        })
            except:
                pass
    return results

def scan_float_cast():
    """Find silent float64->float32 conversions"""
    results = []
    hot_dirs = ['core', 'strategies', 'risk']
    for d in hot_dirs:
        dirpath = os.path.join(PROJECT_ROOT, d)
        if not os.path.isdir(dirpath):
            continue
        for root, _, files in os.walk(dirpath):
            if '__pycache__' in root:
                continue
            for f in files:
                if not f.endswith('.py'):
                    continue
                fpath = os.path.join(root, f)
                try:
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as fp:
                        for i, line in enumerate(fp, 1):
                            if '.astype(np.float32)' in line or 'cast(pl.Float32)' in line or 'dtype=np.float32' in line:
                                results.append({
                                    'file': os.path.relpath(fpath, PROJECT_ROOT),
                                    'line': i,
                                    'code': line.strip()[:120]
                                })
                except:
                    pass
    return results

# ═══════════════════════════════════════════════════════════════
# EXECUTION
# ═══════════════════════════════════════════════════════════════

print("=" * 80)
print("🔬 AUDITORÍA HOLOGRÁFICA DE INTEGRIDAD ARQUITECTÓNICA")
print("=" * 80)

# Phase I
print("\n📊 PHASE I: CENSO FEATUREAL")
print("-" * 40)
fe_features, aliases = scan_feature_engineering()
ind_features = scan_indicator_modules()
zero_features = scan_hardcoded_zero_features(fe_features)

all_features = {**fe_features, **ind_features}
print(f"  Total features detectadas en feature_engineering.py: {len(aliases)}")
print(f"  Total features en indicator modules: {len(ind_features)}")
print(f"  Features HARDCODED a 0.0 (MUERTAS): {len(zero_features)}")
print(f"\n  🔴 FEATURES MUERTAS (pl.lit(0.0) = SIEMPRE CERO):")
for zf in sorted(zero_features):
    print(f"    ❌ {zf}")

print(f"\n  🟢 FEATURES VIVAS (calculadas realmente):")
live_features = [a for a in aliases if a not in zero_features]
for lf in sorted(live_features):
    print(f"    ✅ {lf}")

# Phase II
print("\n\n🕸️ PHASE II: TOPOLOGÍA DE GRAFOS")
print("-" * 40)

silent_fallbacks = scan_silent_fallbacks()
print(f"\n  ⚠️ SILENT FALLBACKS (try/except → None): {len(silent_fallbacks)}")
for sf in silent_fallbacks[:30]:
    print(f"    📂 {sf['file']}: {sf['variable']} = None")

silent_gets = scan_silent_gets()
print(f"\n  ⚠️ SILENT .get() en hot-path: {len(silent_gets)}")
for sg in silent_gets[:20]:
    print(f"    📂 {sg['file']}:{sg['line']}: {sg['code']}")

return_zeros = scan_return_zero()
print(f"\n  ⚠️ return 0.0/None en except: {len(return_zeros)}")
for rz in return_zeros[:20]:
    print(f"    📂 {rz['file']}:{rz['line']}: {rz['code']}")

time_sleeps = scan_time_sleep_in_async()
print(f"\n  ⚠️ time.sleep() en async def: {len(time_sleeps)}")
for ts in time_sleeps[:15]:
    print(f"    📂 {ts['file']}:{ts['line']}: {ts['code']}")

float_casts = scan_float_cast()
print(f"\n  ⚠️ SILENT float64→float32 conversions: {len(float_casts)}")
for fc in float_casts[:15]:
    print(f"    📂 {fc['file']}:{fc['line']}: {fc['code']}")

# Summary
print("\n\n" + "=" * 80)
print("📋 VEREDICTO DE INTEGRIDAD ARQUITECTÓNICA")
print("=" * 80)
print(f"  Features TOTAL detectadas:         {len(all_features)}")
print(f"  Features VIVAS (calculadas):       {len(live_features)}")
print(f"  Features MUERTAS (hardcoded 0.0):  {len(zero_features)}")
print(f"  Silent Fallbacks (→ None):         {len(silent_fallbacks)}")
print(f"  Silent .get() en hot-path:         {len(silent_gets)}")
print(f"  return 0.0/None en except:         {len(return_zeros)}")
print(f"  time.sleep() en async:             {len(time_sleeps)}")
print(f"  float64→float32 silenciosa:        {len(float_casts)}")

# Export
report = {
    'total_features': len(all_features),
    'live_features': len(live_features),
    'dead_features_zero': len(zero_features),
    'dead_feature_names': sorted(zero_features),
    'live_feature_names': sorted(live_features),
    'silent_fallbacks_count': len(silent_fallbacks),
    'silent_gets_count': len(silent_gets),
    'return_zeros_count': len(return_zeros),
    'time_sleep_async_count': len(time_sleeps),
    'float_cast_count': len(float_casts),
    'silent_fallbacks': silent_fallbacks[:50],
    'silent_gets': silent_gets[:50],
    'return_zeros': return_zeros[:50],
    'time_sleeps': time_sleeps[:30],
    'float_casts': float_casts[:30],
}

out_path = os.path.join(PROJECT_ROOT, "logs", "audits", "holographic_audit_results.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(report, f, indent=2)
print(f"\n  💾 Resultados exportados a: {out_path}")
