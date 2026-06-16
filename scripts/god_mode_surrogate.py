import os
import sys
import time
import numpy as np
import optuna
import lightgbm as lgb
import pandas as pd
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.genotype import Genotype
from core.simulation import SimDataProvider, SimulationEngine

class ParetoSurrogateEnsemble:
    def __init__(self):
        self.lgb_performance = None
        self.lgb_risk = None
        self.lgb_robustness = None
        
        self.max_score = 0
        
    def train(self, X, y_perf, y_risk, y_rob):
        params = {'objective': 'regression', 'metric': 'rmse', 'verbose': -1, 'n_jobs': -1}
        self.lgb_performance = lgb.train(params, lgb.Dataset(X, label=y_perf), num_boost_round=100)
        self.lgb_risk = lgb.train(params, lgb.Dataset(X, label=y_risk), num_boost_round=100)
        self.lgb_robustness = lgb.train(params, lgb.Dataset(X, label=y_rob), num_boost_round=100)
        self.max_score = max(y_perf)

    def predict(self, X):
        return (
            self.lgb_performance.predict(X),
            self.lgb_risk.predict(X),
            self.lgb_robustness.predict(X)
        )

class GodModeSurrogate:
    def __init__(self):
        self.surrogate = ParetoSurrogateEnsemble()
        self.base_genes = Genotype(symbol='MOCK').genes
        
    def generate_dummy_data(self):
        # 10,000 velas = F4 (Full)
        n_candles = 10000
        returns = np.random.normal(0.0001, 0.002, n_candles)
        closes = 100.0 * np.exp(np.cumsum(returns))
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2026-01-01', periods=n_candles, freq='1min'),
            'open': closes,
            'high': closes * 1.001,
            'low': closes * 0.999,
            'close': closes,
            'volume': np.random.uniform(10, 1000, n_candles)
        })
        return SimDataProvider({'BTCUSDT': df})

    def dict_to_vector(self, params):
        # Flatten all scalar params to a vector for LGBM
        vec = []
        for k in sorted(self.base_genes.keys()):
            val = params.get(k, self.base_genes[k])
            if isinstance(val, (int, float)):
                vec.append(val)
        return vec

    def run_bare_metal(self, engine, params, max_candles=None):
        geno = Genotype(symbol='BTCUSDT', genes=params)
        trades = engine.run(geno, 'BTCUSDT', max_candles=max_candles)
        
        if not trades:
            return 0.0, -1.0, 0.0 # Score, Risk, Rob
            
        pnl = [t.pnl_pct for t in trades]
        total_pnl = sum(pnl)
        win_rate = sum(1 for t in trades if t.is_win) / len(trades)
        
        # Drawdown max (simplified)
        cum_pnl = np.cumsum(pnl)
        peak = np.maximum.accumulate(cum_pnl)
        drawdown = peak - cum_pnl
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0
        
        score = total_pnl * win_rate
        return float(score), float(-max_dd), float(win_rate) # Risk is inverse DD (higher is better)

    def phase_explore(self, engine, n_trials=50):
        print(f"\n[FASE F4 Exploratoria] Muestreo Bare Metal {n_trials} trials...")
        X, y_perf, y_risk, y_rob = [], [], [], []
        
        for _ in range(n_trials):
            params = self.base_genes.copy()
            # Mutate slightly
            params['tp_pct'] = np.random.uniform(0.01, 0.05)
            params['sl_pct'] = np.random.uniform(0.01, 0.05)
            # Fill brain weights properly for execution
            params['brain_weights'] = np.random.uniform(-1, 1, 100).tolist()
            
            score, risk, rob = self.run_bare_metal(engine, params, max_candles=10000)
            X.append(self.dict_to_vector(params))
            y_perf.append(score)
            y_risk.append(risk)
            y_rob.append(rob)
            
        return np.array(X), np.array(y_perf), np.array(y_risk), np.array(y_rob)

    def phase_exploit(self, engine, n_virtual_trials=100000):
        print(f"\n[FASE F0-F3 Multi-Fidelidad] Optuna TPE Hyperband con {n_virtual_trials} trials...")
        start_t = time.time()
        
        # The fidelity steps: 1: F1 (100 candles), 2: F2 (1000 candles), 3: F3 (5000 candles)
        steps_candles = {1: 100, 2: 1000, 3: 5000}
        
        def objective(trial):
            params = self.base_genes.copy()
            for k, val in self.base_genes.items():
                if isinstance(val, int):
                    params[k] = trial.suggest_int(k, max(1, val//2), int(val*1.5) + 1)
                elif isinstance(val, float):
                    if val < 1.0:
                        params[k] = trial.suggest_float(k, 0.001, 0.1)
                    else:
                        params[k] = trial.suggest_float(k, val/2.0, val*1.5)
                        
            # Set dummy brain weights for trial run
            params['brain_weights'] = np.random.uniform(-1, 1, 100).tolist()
            
            vec = self.dict_to_vector(params)
            
            # FASE F0 (Surrogate Pareto Filter)
            p_score, p_risk, p_rob = self.surrogate.predict([vec])
            p_score, p_risk, p_rob = p_score[0], p_risk[0], p_rob[0]
            
            # Anti-Hallucination + Risk Check
            # We relax conditions if the base performance is zero
            if self.surrogate.max_score > 0.001:
                if p_score < self.surrogate.max_score * 0.1 or p_risk < -0.20 or p_rob < 0.30:
                    raise optuna.TrialPruned() # Cortado instantáneo en F0 (0 ms de coste CPU real)
            
            # Si sobrevive F0, entramos a la escalera de fidelidad
            last_score = p_score
            for step in range(1, 4):
                mc = steps_candles[step]
                score, risk, rob = self.run_bare_metal(engine, params, max_candles=mc)
                
                trial.report(score, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                    
                last_score = score
                
            return last_score

        pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=3, reduction_factor=3)
        study = optuna.create_study(direction='maximize', pruner=pruner)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study.optimize(objective, n_trials=n_virtual_trials, n_jobs=1)
        
        print(f"✅ Optuna completó la búsqueda Multi-Fidelidad en {time.time()-start_t:.2f}s.")
        
        trials = sorted([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE], key=lambda x: x.value, reverse=True)
        return trials[:10]

    def phase_validate(self, engine, top_trials):
        print("\n[FASE F4] Validando el Top 10 en Dataset Completo (10,000 Velas)...")
        best_score = -999
        best_params = None
        
        for idx, t in enumerate(top_trials):
            params = self.base_genes.copy()
            for k, v in t.params.items():
                params[k] = v
            params['brain_weights'] = np.random.uniform(-1, 1, 100).tolist()
            
            score, risk, rob = self.run_bare_metal(engine, params, max_candles=None)
            print(f"  Validando Trial #{t.number} -> F4 Score: {score:.4f} (Risk: {risk:.4f}, Rob: {rob:.4f})")
            
            if score > best_score:
                best_score = score
                best_params = params
                
        print(f"\n👑 SANTO GRIAL ENCONTRADO. Score Final: {best_score:.4f}")
        
        if best_params:
            out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config', 'genotypes'))
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, 'god_mode_best.json'), 'w') as f:
                json.dump(best_params, f, indent=4)
            print(f"✅ Guardado en {out_dir}/god_mode_best.json")


def main():
    print("==================================================")
    print("⚛️ GOD MODE SURROGATE: MULTI-FIDELITY SUPREME")
    print("==================================================")
    
    god = GodModeSurrogate()
    engine = SimulationEngine(god.generate_dummy_data())
    
    # FASE 0: Recolectar datos y Entrenar Pareto Ensemble
    X, y_perf, y_risk, y_rob = god.phase_explore(engine, n_trials=50)
    print("\n[FASE Entrenamiento] Entrenando Pareto Ensemble Multiobjetivo...")
    god.surrogate.train(X, y_perf, y_risk, y_rob)
    
    # FASE MULTI-FIDELITY
    top_trials = god.phase_exploit(engine, n_virtual_trials=2000) # 2k trials para test
    
    # FASE VALIDACION
    if top_trials:
        god.phase_validate(engine, top_trials)
    else:
        print("❌ Ningún genoma sobrevivió la poda Hyperband.")

if __name__ == '__main__':
    main()
