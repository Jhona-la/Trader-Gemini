"""Analyze what limits trade frequency."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

print("=== FREQUENCY LIMITERS ===")
s = Config.Horizons.Scalping
w = Config.Horizons.Swing
print(f"Scalping cooldown: {s['cooldown_seconds']}s")
print(f"Swing cooldown: {w.get('cooldown_seconds', 'N/A')}s")
print(f"Scalping max_hold_time: {s['max_hold_time']}s ({s['max_hold_time']/3600:.1f}h)")
print(f"Scalping timeframes: {s['timeframes']}")
print(f"Primary TF: {s['primary_tf']}")
print(f"strength_threshold: {s['strength_threshold']}")
print(f"min_volume_ratio: {s['min_volume_ratio']}")
print(f"sophia_win_prob_min: {Config.Horizons.GlobalThresholds['sophia_win_prob_min']}")

print(f"\n=== SYMBOLS ===")
print(f"TRADING_PAIRS: {Config.TRADING_PAIRS}")
print(f"CRYPTO_FUTURES_PAIRS: {Config.CRYPTO_FUTURES_PAIRS}")
print(f"Pairs count: {len(Config.CRYPTO_FUTURES_PAIRS)}")

print(f"\n=== LEVERAGE ===")
print(f"MAX_LEVERAGE: {getattr(Config, 'MAX_LEVERAGE', 'N/A')}")
print(f"DEFAULT_LEVERAGE: {getattr(Config, 'DEFAULT_LEVERAGE', 'N/A')}")

print(f"\n=== POSITION SIZING ===")
risk = getattr(Config, 'Risk', None)
if risk:
    print(f"MAX_POSITION_PCT: {getattr(risk, 'MAX_POSITION_PCT', 'N/A')}")
    print(f"MAX_DRAWDOWN: {getattr(risk, 'MAX_DRAWDOWN', 'N/A')}")
    print(f"MAX_DAILY_LOSS: {getattr(risk, 'MAX_DAILY_LOSS', 'N/A')}")
    print(f"MAX_POSITIONS: {getattr(risk, 'MAX_POSITIONS', 'N/A')}")
    print(f"MIN_ORDER_VALUE: {getattr(risk, 'MIN_ORDER_VALUE', 'N/A')}")
