"""Feature Waste Scanner"""
import re

with open('strategies/ml_strategy.py', 'r', encoding='utf-8') as f:
    content = f.read()

start = content.find('def _prepare_features(')
next_def = content.find('\n    def ', start + 10)
section = content[start:next_def] if next_def > 0 else content[start:start+5000]

created = set()
for m in re.finditer(r"df\[(?:'|\")([\w_]+)(?:'|\")\]\s*=", section):
    created.add(m.group(1))

top20 = ['returns_5','returns_10','roc_10','rsi_14','atr_pct',
         'macd_hist','bb_position','bb_width','stoch_k','adx',
         'volume_ratio','gk_vol','hurst_memory','volatility_ransac',
         'micro_imbalance','spread_squeeze','scalp_velocity_1',
         'scalp_rsi_divergence','micro_label','market_cluster']

ghost_ws = ['vbi','vbi_avg','liq_intensity','funding_rate','oi','oi_delta',
            'funding_distortion','l2_ofi','l2_spread','l2_microprice_dist']

used = sorted([f for f in created if f in top20])
unused = sorted([f for f in created if f not in top20 
                 and f not in ['label','open','high','low','close','volume','timestamp']])

print(f"Features CREATED in _prepare_features: {len(created)}")
print(f"Features USED in training (top20): {len(used)}")
print(f"Features COMPUTED but NOT used in training: {len(unused)}")
print()
for f in unused:
    if f in ghost_ws:
        tag = "GHOST (websocket-only)"
    else:
        tag = "WASTED COMPUTATION"
    print(f"  {f}: {tag}")

# Also check which top20 features are NOT created
missing = [f for f in top20 if f not in created]
if missing:
    print(f"\nFeatures in top20 NOT created by _prepare_features:")
    for f in missing:
        print(f"  {f}")

# Check other prepare methods
print(f"\nAll _prepare methods:")
for m in re.finditer(r"def (_prepare_\w+)\(", content):
    line_num = content[:m.start()].count('\n') + 1
    print(f"  L{line_num}: {m.group(1)}")

# Check engine.py for self.engine injection into strategy
print("\n=== ENGINE → STRATEGY injection audit ===")
with open('core/engine.py', 'r', encoding='utf-8') as f:
    eng = f.read()

for m in re.finditer(r"strategy\.(\w+)\s*=\s*self", eng):
    line_num = eng[:m.start()].count('\n') + 1
    line = eng.split('\n')[line_num - 1].strip()
    print(f"  L{line_num}: {line}")

# Check if strategies receive engine reference
print("\n=== ML STRATEGY: self.engine references ===")
engine_refs = [(i+1, line.strip()) for i, line in enumerate(content.split('\n'))
               if 'self.engine' in line and not line.strip().startswith('#')]
print(f"  Total refs: {len(engine_refs)}")
for ln, line in engine_refs[:5]:
    print(f"  L{ln}: {line[:100]}")

# Check data_provider phantom
print("\n=== DATA_PROVIDER: get_latest_bars definition ===")
with open('core/data_provider.py', 'r', encoding='utf-8') as f:
    dp = f.read()

for m in re.finditer(r"def (get_latest\w+|get_hft\w+|get_derivatives\w+)\(", dp):
    line_num = dp[:m.start()].count('\n') + 1
    print(f"  L{line_num}: {m.group(1)}")
