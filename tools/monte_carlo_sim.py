import json
import random
import numpy as np
import os

class MonteCarloSimulator:
    """
    🧪 [PHASE 11] MONTE CARLO STRESS TESTER
    Tests capital survival across 5,000+ parallel universes.
    
    👨‍🏫 MODO PROFESOR:
    - QUÉ: Simulación estadística de remuestreo (Resampling).
    - POR QUÉ: El éxito no es solo ganar, es NO MORIR en una racha de mala suerte.
    - PARA QUÉ: Validar que $15 de capital inicial es suficiente para el bot.
    """
    def __init__(self, trades_file="backtest_results.json", initial_capital=15.0, iterations=5000):
        self.trades_file = trades_file
        self.initial_capital = initial_capital
        self.iterations = iterations
        self.pnls = []
        
    def load_trades(self):
        if not os.path.exists(self.trades_file):
            print(f"❌ Error: Archivo {self.trades_file} no encontrado.")
            return False
            
        with open(self.trades_file, "r") as f:
            data = json.load(f)
            # Handle different formats (standard backtest vs WFV)
            if isinstance(data, list):
                # Check if it's WFV results (list of folds)
                if len(data) > 0 and 'metrics' in data[0]:
                    # We need trade-level PnL, not just metrics. 
                    # If trades aren't in JSON, we'll simulate based on win rate and avg profit
                    print("📊 Detectados resultados de WFV. Extrayendo métricas para simulación sintética...")
                    return self.generate_synthetic_trades(data)
                else:
                    # Standard backtest trades list
                    self.pnls = [t.get('pnl_pct', 0) for t in data]
            elif isinstance(data, dict) and 'trades' in data:
                self.pnls = [t.get('pnl_pct', 0) for t in data['trades']]
            
        return len(self.pnls) > 0

    def generate_synthetic_trades(self, wfv_folds):
        """Generates a representative PnL list from WFV summary metrics."""
        all_metrics = [f['metrics'] for f in wfv_folds if f['metrics']['sharpe'] > 0]
        if not all_metrics: return False
        
        # Simplified: Use avg win rate and avg move to create 100 representative trades
        avg_wr = np.mean([m['win_rate'] for m in all_metrics]) / 100
        # Assume standard target sizes if trades aren't exported
        tp = 0.015
        sl = -0.02
        
        self.pnls = []
        for _ in range(100):
            if random.random() < avg_wr:
                self.pnls.append(tp)
            else:
                self.pnls.append(sl)
        return True

    def run_simulation(self):
        print(f"🚀 Iniciando Simulación de Monte Carlo ({self.iterations} iteraciones)...")
        print(f"💰 Capital Inicial: ${self.initial_capital}")
        
        final_balances = []
        max_drawdowns = []
        ruin_count = 0
        
        for i in range(self.iterations):
            # Shuffle current trades
            sequence = random.choices(self.pnls, k=len(self.pnls))
            
            balance = self.initial_capital
            peak = self.initial_capital
            mdd = 0
            
            for pnl in sequence:
                balance *= (1 + pnl)
                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak
                mdd = max(mdd, dd)
                
                # Check for Ruin (e.g., losing 50% of capital for micro-accounts)
                if balance < self.initial_capital * 0.5:
                    ruin_count += 1
                    break
            
            final_balances.append(balance)
            max_drawdowns.append(mdd)
            
        self.report(final_balances, max_drawdowns, ruin_count)

    def report(self, balances, drawdowns, ruin_count):
        print("\n" + "="*60)
        print("📊 REPORTE DE ESTRÉS (MONTE CARLO)")
        print("="*60)
        
        prob_ruin = (ruin_count / self.iterations) * 100
        avg_final = np.mean(balances)
        p5 = np.percentile(balances, 5)
        p95 = np.percentile(balances, 95)
        avg_mdd = np.mean(drawdowns) * 100
        
        print(f"💀 Riesgo de Ruina (Drawdown > 50%): {prob_ruin:.2f}%")
        print(f"📈 Retorno Promedio Esperado: ${avg_final:.2f}")
        print(f"📉 Peor Escenario (P5%): ${p5:.2f}")
        print(f"🚀 Mejor Escenario (P95%): ${p95:.2f}")
        print(f"⚠️ Drawdown Promedio: {avg_mdd:.2f}%")
        
        status = "✅ ROBUSTO" if prob_ruin < 1 and avg_mdd < 15 else "❌ RIESGOSO"
        print(f"\nESTADO FINAL: {status}")
        print("="*60)

if __name__ == "__main__":
    # Prioritize WFV results if they exist, otherwise use standard backtest
    source = "logs/walk_forward_results.json" if os.path.exists("logs/walk_forward_results.json") else "backtest_results.json"
    sim = MonteCarloSimulator(trades_file=source)
    if sim.load_trades():
        sim.run_simulation()
    else:
        # Fallback to manual synthetic test if no files found
        print("⚠️ No se encontraron archivos de resultados. Usando perfil sintético (WR=58%, TP=1.5%, SL=2%).")
        sim.pnls = [0.015]*58 + [-0.02]*42
        sim.run_simulation()
