import sys
import traceback
sys.path.append('.')
from optimization.hyper_optimizer import HyperOptimizer
from core.simulation import SimDataProvider
from scripts.run_hyper_optimization import load_real_data, create_simulation_runner

def test():
    df = load_real_data("BTC/USDT", days=1)
    provider = SimDataProvider({"BTC/USDT": df})
    runner = create_simulation_runner("BTC/USDT", provider)
    opt = HyperOptimizer(simulation_runner=runner)
    
    cfg = {'bollinger_period': 20, 'bollinger_std': 2.0, 'rsi_period': 14, 'macd_fast': 8, 'macd_slow': 21, 'rsi_window': 14, 'tp_pct': 0.02, 'sl_pct': 0.02, 'atr_sl_multiplier': 2.0}
    
    try:
        fold_results = opt.simulation_runner("BTC/USDT", cfg)
        score1, _ = opt.objective_func.evaluate_configuration(fold_results)
        print("Score 1:", score1)
    except Exception as e:
        traceback.print_exc()

if __name__ == '__main__':
    test()
