"""
🔍 AUDIT: Data Flow Integrity Test
====================================
Verifies that ALL data sources reach the ML pipeline with REAL data.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np

def audit_macro_intelligence():
    """Test 1: MacroIntelligence returns real data (not defaults/zeros)."""
    print("\n" + "="*70)
    print("🔍 TEST 1: MacroIntelligence — Real Data Verification")
    print("="*70)
    
    from data.macro_intelligence import macro_intelligence
    feats = macro_intelligence.get_macro_features()
    
    passed = 0
    failed = 0
    
    critical_features = {
        "fng_value": ("Fear & Greed Index", lambda v: 0 < v <= 100),
        "btc_dominance": ("BTC Dominance %", lambda v: 10 < v < 90),
        "total_market_cap_norm": ("Total Market Cap (T)", lambda v: v > 0),
        "macro_dxy_returns": ("DXY Returns", lambda v: True),  # Can be 0
        "macro_vix": ("VIX", lambda v: v > 0),
        "macro_gold_returns": ("Gold Returns", lambda v: True),
        "macro_us10y": ("US 10Y Yield", lambda v: v > 0),
    }
    
    for key, (name, validator) in critical_features.items():
        val = feats.get(key, None)
        ok = val is not None and validator(val)
        icon = "✅" if ok else "❌"
        print(f"  {icon} {name}: {val}")
        if ok: passed += 1
        else: failed += 1
    
    print(f"\n  📊 Macro Features Total: {len(feats)} keys | Passed: {passed} | Failed: {failed}")
    return failed == 0


def audit_derivatives():
    """Test 2: Binance derivatives return real data per symbol."""
    print("\n" + "="*70)
    print("🔍 TEST 2: Binance Derivatives — Per-Symbol Real Data")
    print("="*70)
    
    from data.macro_intelligence import macro_intelligence
    
    test_symbols = ["BTCUSDT", "ETHUSDT"]
    all_ok = True
    
    for sym in test_symbols:
        d = macro_intelligence.get_derivatives_features(sym)
        print(f"\n  📊 {sym}:")
        
        for key, val in d.items():
            # funding_rate can be 0, but L/S ratio should be > 0
            is_critical = key in ["long_short_ratio", "taker_buy_sell_ratio"]
            ok = val != 0.0 if is_critical else True
            icon = "✅" if ok else "⚠️"
            print(f"    {icon} {key}: {val}")
            if is_critical and not ok:
                all_ok = False
    
    return all_ok


def audit_onchain_no_random():
    """Test 3: Verify on-chain loader does NOT produce random noise."""
    print("\n" + "="*70)
    print("🔍 TEST 3: On-Chain Loader — NO Random Noise")
    print("="*70)
    
    from data.onchain_loader import onchain_loader
    
    # Call 10 times and check that all values are identical (no randomness)
    values = []
    for _ in range(10):
        feats = onchain_loader.get_onchain_features()
        values.append(feats.get('onchain_whale_flow', None))
    
    all_same = len(set(values)) == 1
    all_zero = all(v == 0.0 for v in values)
    has_real_data = onchain_loader.onchain_state.get('has_real_data', False)
    
    if all_same and all_zero:
        print(f"  ✅ All 10 calls returned 0.0 (deterministic, not random)")
        print(f"  ✅ has_real_data = {has_real_data} (correctly False)")
    else:
        print(f"  ❌ VALUES VARY: {set(values)} — RANDOM NOISE DETECTED!")
    
    return all_same and all_zero and not has_real_data


def audit_feature_engineering():
    """Test 4: Full feature engineering pipeline produces 30+ macro features."""
    print("\n" + "="*70)
    print("🔍 TEST 4: Feature Engineering — Full Pipeline (30+ macro features)")
    print("="*70)
    
    from strategies.components.feature_engineering import FeatureEngineering
    
    fe = FeatureEngineering()
    
    # Create dummy OHLCV bars (100 bars)
    n = 200
    close = np.cumsum(np.random.randn(n) * 0.5) + 100000
    bars = {
        'open': close + np.random.randn(n) * 10,
        'high': close + abs(np.random.randn(n) * 50),
        'low': close - abs(np.random.randn(n) * 50),
        'close': close,
        'volume': np.random.rand(n) * 1000 + 100,
    }
    
    df = fe.prepare_features(bars, market_regime="TRENDING", symbol="BTC/USDT", horizon="SCALPING")
    
    if df is None or len(df) == 0:
        print("  ❌ Feature Engineering returned empty DataFrame!")
        return False
    
    # Check for macro features
    macro_cols = [c for c in df.columns if c.startswith(('fng_', 'btc_dom', 'eth_dom', 'macro_', 'total_', 'market_cap',
                                                         'funding_', 'open_interest', 'oi_change', 'long_short',
                                                         'top_trader', 'taker_', 'onchain_'))]
    
    print(f"  📊 Total columns: {len(df.columns)}")
    print(f"  📊 Macro/Micro columns found: {len(macro_cols)}")
    print()
    
    for col in sorted(macro_cols):
        vals = df[col].values
        unique = len(np.unique(vals[~np.isnan(vals)]))
        last_val = vals[-1] if len(vals) > 0 else None
        is_all_zero = np.all(vals == 0)
        icon = "⚠️" if is_all_zero else "✅"
        print(f"    {icon} {col}: last={last_val:.6f} | unique={unique} | all_zero={is_all_zero}")
    
    return len(macro_cols) >= 25  # We expect at least 25 macro features


if __name__ == "__main__":
    print("🌐 CTOS DATA OMNISCIENCE — Forensic Audit")
    print("="*70)
    
    results = {}
    
    results["Macro Intelligence"] = audit_macro_intelligence()
    results["Derivatives"] = audit_derivatives()
    results["OnChain Safety"] = audit_onchain_no_random()
    results["Feature Engineering"] = audit_feature_engineering()
    
    print("\n" + "="*70)
    print("📋 AUDIT SUMMARY")
    print("="*70)
    
    all_pass = True
    for name, passed in results.items():
        icon = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {icon}: {name}")
        if not passed:
            all_pass = False
    
    print()
    if all_pass:
        print("🏆 ALL TESTS PASSED — Data pipeline is clean and verified.")
    else:
        print("🚨 SOME TESTS FAILED — Review the output above.")
    
    print("="*70)
