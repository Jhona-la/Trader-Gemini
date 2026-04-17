"""Quick test of hot reload system."""
import sys
sys.path.insert(0, '.')

from utils.reloader import SyntaxValidator, ModuleReloader

print("="*60)
print("HOT RELOAD SYSTEM - Test")
print("="*60)

# Test syntax validator
v = SyntaxValidator()
ok, err = v.validate_file('strategies/ml_strategy.py')
print(f"Syntax Check: {'OK' if ok else err}")
print(f"File Hash: {v.get_file_hash('strategies/ml_strategy.py')}")

# Test module reloader
import strategies.ml_strategy
r = ModuleReloader()
result = r.reload_module('strategies.ml_strategy')
print(f"\nReload Test:")
print(f"  Success: {result.success}")
print(f"  Latency: {result.latency_ms:.2f}ms")
print(f"  Target (<100ms): {'MET' if result.latency_ms < 100 else 'NOT MET'}")

print("="*60)
print("HOT RELOAD READY")
