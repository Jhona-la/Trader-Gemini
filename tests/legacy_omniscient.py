import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from models.omniscient_predictor import omniscient_engine

# Simular 60 barras con 122 features
dummy = np.random.randn(60, 122).astype(np.float32) * 0.01
price = 104000.0

# SCALPING test
route = omniscient_engine.predict_trajectory(dummy, price, horizon='SCALPING')
print("=== SCALPING (1000 velas de 1 minuto) ===")
print(f"Bar duration: {route['bar_duration']}")
print(f"Total candles: {route['total_candles']}")
print(f"Inference: {route['inference_ms']:.1f}ms")
print(f"Macro Peak: +{route['macro_peak_pct']:.4f}% = ${route['macro_peak_usd']:,.2f} en {route['macro_peak_time']}")
print(f"Macro Dump: {route['macro_dump_pct']:.4f}% = ${route['macro_dump_usd']:,.2f} en {route['macro_dump_time']}")
print()
print("--- Primeras 5 velas individuales ---")
for c in route['candles'][:5]:
    icon = 'BULL' if c['bullish'] else 'BEAR'
    print(f"  T+{c['bar']} ({c['time_label']}): O=${c['open_usd']:,.2f} H=${c['high_usd']:,.2f} L=${c['low_usd']:,.2f} C=${c['close_usd']:,.2f} | Size: ${c['candle_size_usd']:.2f} ({c['candle_size_pct']:.4f}%) [{icon}]")
print()
print("--- Waypoints ---")
for wp in route['waypoints']:
    print(f"  T+{wp['bar']} ({wp['time']}): close={wp['close_pct']:+.4f}% = ${wp['close_usd']:,.2f} | candle_size=${wp['candle_size_usd']:.2f} ({wp['candle_size_pct']:.4f}%)")

# SWING test
route2 = omniscient_engine.predict_trajectory(dummy, price, horizon='SWING')
print()
print("=== SWING (1000 velas de 1 hora) ===")
print(f"Bar duration: {route2['bar_duration']}")
print(f"Macro Peak: +{route2['macro_peak_pct']:.4f}% en {route2['macro_peak_time']}")
print(f"Macro Dump: {route2['macro_dump_pct']:.4f}% en {route2['macro_dump_time']}")
for c in route2['candles'][:3]:
    print(f"  T+{c['bar']} ({c['time_label']}): Size=${c['candle_size_usd']:.2f} ({c['candle_size_pct']:.4f}%)")

# Save trajectory
path = omniscient_engine.save_trajectory_to_file(route, "BTCUSDT")
print(f"\nTrajectory saved to: {path}")
