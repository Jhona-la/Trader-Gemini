"""
DEEP IMPORT AUDIT — Tests every critical module can be imported.
This catches circular imports, missing dependencies, and broken references.
"""
import sys
import os
import traceback

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CRITICAL_MODULES = [
    # Layer 0: Foundation
    ("config", "Config"),
    ("core.enums", "TradeDirection, TimeFrame"),
    ("core.events", "MarketEvent, SignalEvent, OrderEvent, FillEvent"),
    ("core.secure_store", "SecureString"),
    ("utils.logger", "logger"),
    ("utils.fast_json", "FastJson"),
    ("utils.atomic_guard", "AtomicGuard"),
    ("utils.axioma_math", "PrecisionAuditor"),
    
    # Layer 1: Data
    ("data.database", "DatabaseHandler"),
    ("data.binance_loader", "BinanceDataLoader"),
    
    # Layer 2: Core Infrastructure
    ("core.state_manager", "AtomicStateManager"),
    ("core.portfolio", "Portfolio"),
    ("core.clock", "global_clock"),
    ("core.global_state", "global_state"),
    ("core.event_bus", "event_bus"),
    ("core.market_regime", "MarketRegimeDetector"),
    ("core.prediction_tracker", "PredictionTracker"),
    ("core.compounding_engine", "get_compounding_engine"),
    ("core.transparent_logger", "TransparentLogger"),
    ("core.feedback_processor", "feedback_processor"),
    
    # Layer 3: Strategies
    ("strategies.technical", "TechnicalStrategy"),
    ("strategies.ml_strategy", "MLStrategyHybridUltimateV3"),
    ("strategies.components.adaptive_engine", "AdaptiveMLParameterEngine"),
    
    # Layer 4: Risk
    ("risk.risk_manager", "RiskManager"),
    ("risk.kill_switch", "KillSwitch"),
    
    # Layer 5: Execution
    ("execution.binance_executor", "BinanceExecutor"),
    
    # Layer 6: Models
    ("models.deep_predictor", "DeepPredictor"),
    ("models.omniscient_predictor", "OmniscientPredictor"),
    
    # Layer 7: Sophia
    ("sophia.intelligence", "SophiaIntelligence"),
    ("sophia.post_mortem", "PostMortemComparator"),
    ("sophia.nemesis", "NemesisEngine"),
    
    # Layer 8: Engine
    ("core.engine", "Engine"),
]

print(f"\n{'='*70}")
print(f"  DEEP IMPORT AUDIT — {len(CRITICAL_MODULES)} critical modules")
print(f"{'='*70}\n")

passed = 0
failed = 0
warnings = []

for module_path, expected_names in CRITICAL_MODULES:
    try:
        mod = __import__(module_path, fromlist=expected_names.split(', '))
        # Verify each expected name exists
        missing = []
        for name in expected_names.split(', '):
            name = name.strip()
            if not hasattr(mod, name):
                missing.append(name)
        
        if missing:
            print(f"  ⚠️  {module_path}: IMPORTED but missing: {', '.join(missing)}")
            warnings.append((module_path, missing))
        else:
            print(f"  ✅ {module_path}")
        passed += 1
    except Exception as e:
        failed += 1
        err_type = type(e).__name__
        err_msg = str(e).split('\n')[0][:80]
        print(f"  ❌ {module_path}: {err_type}: {err_msg}")

print(f"\n{'='*70}")
print(f"  RESULTS: {passed} passed, {failed} FAILED, {len(warnings)} warnings")
print(f"{'='*70}")

if warnings:
    print("\n⚠️  WARNINGS (partial imports):")
    for mod, missing in warnings:
        print(f"   {mod} → missing: {', '.join(missing)}")

if failed:
    print(f"\n🚨 {failed} CRITICAL IMPORT FAILURES — these modules will crash at runtime!")
    sys.exit(1)
else:
    print("\n✅ All critical modules importable. No circular dependency crashes.\n")
