import optuna
import os

try:
    study = optuna.load_study(study_name='evo_BTCUSDT_30D', storage='sqlite:///data/mass_evolver.db')
    trials = study.trials
    completed = [t for t in trials if t.state.name == 'COMPLETE']
    best = study.best_trial if len(completed) > 0 else None
    
    print("=== BTC EVOLUTION STATUS ===")
    print(f"Trials Completed: {len(completed)}")
    if best:
        print(f"Best PnL: ${best.user_attrs['pnl_usd']:.2f}")
        print(f"Best Win Rate: {best.user_attrs['win_rate']:.1f}%")
        print(f"Best Score: {best.value:.2f}")
except Exception as e:
    print(f"Could not load BTC study: {e}")

try:
    study_eth = optuna.load_study(study_name='evo_ETHUSDT_30D', storage='sqlite:///data/mass_evolver.db')
    trials_eth = study_eth.trials
    completed_eth = [t for t in trials_eth if t.state.name == 'COMPLETE']
    print("\n=== ETH EVOLUTION STATUS ===")
    print(f"Trials Completed: {len(completed_eth)}")
except Exception as e:
    from utils.error_handler import SystemIntegrityError
    raise SystemIntegrityError('Silent fallback blocked by Holographic Audit')
