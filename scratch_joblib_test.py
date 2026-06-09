import os
import sys

from config import Config
from core.simulation import SimulationEngine, SimDataProvider
from optimization.objective_function import WalkForwardValidator
from core.backtest_infra import fetch_binance_data
from core.genotype import Genotype
from joblib import Parallel, delayed

def eval_config(cfg, provider):
    try:
        print("Worker received dtype names:", provider.arrays["BTC/USDT"].dtype.names)
        engine = SimulationEngine(provider)
        genotype = Genotype("BTC/USDT", cfg)
        trades = engine.run(genotype, "BTC/USDT", start_idx=0, end_idx=100)
        return 1.0
    except Exception as e:
        import traceback
        traceback.print_exc()
        return -999.0

if __name__ == '__main__':
    df = fetch_binance_data("BTC/USDT", days=1)
    df.index.name = 'timestamp'
    provider = SimDataProvider({"BTC/USDT": df})
    print("Testing Joblib Parallel with SimDataProvider...")
    results = Parallel(n_jobs=2, max_nbytes=None)(delayed(eval_config)({"tp_pct": 0.05}, provider) for _ in range(2))
    print(results)
