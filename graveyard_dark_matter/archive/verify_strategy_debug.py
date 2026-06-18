from strategies.technical import HybridScalpingStrategy
from core.genotype import Genotype
import os

print("--- STRATEGY PROFILE VERIFICATION ---")
# Mock providers
class Mock: pass
dp = Mock()
eq = Mock()
s = HybridScalpingStrategy(dp, eq)

print(f"Conservative Profile: {s.PROFILES['CONSERVATIVE']}")
print(f"ADX Threshold (Conservative): {s.PROFILES['CONSERVATIVE']['adx_threshold']}")

# Test genotype loading
symbol = "BTC/USDT"
filename = f"data/genotypes/{symbol.replace('/','')}_gene.json"
print(f"Checking for {filename}...")
if os.path.exists(filename):
    g = Genotype.load(filename)
    print(f"Loaded Genotype Genes: {g.genes}")
else:
    print("No genotype file found.")

params = s.get_symbol_params(symbol)
print(f"Final Merged Params for {symbol}: {params}")
