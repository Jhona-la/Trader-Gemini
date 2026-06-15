import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config

print("=== TEST ADAPTIVE PROFILE ENGINE ===")
print("Testing BTC/USDT (Major) - SCALPING")
btc_scalp = Config.AdaptiveProfileEngine.get("BTC/USDT", "SCALPING")
print(btc_scalp)

print("\nTesting ETH/USDT (Major) - SWING")
eth_swing = Config.AdaptiveProfileEngine.get("ETH/USDT", "SWING")
print(eth_swing)

print("\nTesting DOGE/USDT (Meme) - SCALPING")
doge_scalp = Config.AdaptiveProfileEngine.get("DOGE/USDT", "SCALPING")
print(doge_scalp)

print("\nTesting RENDER/USDT (Alt) - SWING")
render_swing = Config.AdaptiveProfileEngine.get("RENDER/USDT", "SWING")
print(render_swing)

print("\n=== TEST BACKWARDS COMPATIBILITY ===")
print("Trailing Asset Profile for DOT/USDT:")
print(Config.Trailing.get_asset_profile("DOT/USDT"))
