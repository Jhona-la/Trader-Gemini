"""
📉 OMEGA-VOID §3.3: Real Mathematical Expectancy Analysis

QUÉ: Análisis estadístico completo de la esperanza matemática del bot.
POR QUÉ: Un Sharpe Ratio calculado sin slippage realista es mentira.
     La esperanza matemática REAL incluye:
     - Slippage de 3 ticks (modelo realista)
     - Comisiones Binance (0.02% con BNB discount)
     - Funding rate impact en posiciones overnight
PARA QUÉ: Certificar que el bot tiene edge REAL (E[R] > 0 después de costos)
     con intervalos de confianza del 95%.
CÓMO: Monte Carlo simulation con 1000 equity curves generadas
     a partir de la distribución empírica de trades.
CUÁNDO: Después de cada backtest y antes de producción.
DÓNDE: tests/expectancy_analysis.py
QUIÉN: ExpectancyAnalyzer → genera certificado estadístico.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ============================================================
# COST MODEL (Binance Futures)
# ============================================================

@dataclass
class CostModel:
    """
    Modelo de costos realista para Binance Futures.
    
    QUÉ: Todos los costos friccionales que afectan el PnL real.
    POR QUÉ: La mayoría de backtests ignoran funding, slippage dinámico
         y spread, dando resultados irrealistamente optimistas.
    """
    taker_fee_pct: float = 0.0004     # 0.04% (con BNB discount = 0.02%)
    maker_fee_pct: float = 0.0002     # 0.02%
    slippage_ticks: int = 3           # 3 price ticks of slippage
    tick_size_pct: float = 0.0001     # 0.01% per tick (BTC/USDT)
    funding_rate_8h: float = 0.0001   # 0.01% per 8h (neutral market)
    spread_pct: float = 0.0001        # 0.01% half-spread
    
    def round_trip_cost(self, notional: float) -> float:
        """Total cost of opening + closing a position."""
        fee_cost = notional * (self.taker_fee_pct * 2)  # Entry + exit
        slippage_cost = notional * (self.slippage_ticks * self.tick_size_pct * 2)
        spread_cost = notional * (self.spread_pct * 2)
        return fee_cost + slippage_cost + spread_cost
    
    def funding_cost(self, notional: float, hours_held: float) -> float:
        """Funding cost for holding a futures position."""
        funding_periods = hours_held / 8
        return notional * self.funding_rate_8h * funding_periods


class ExpectancyAnalyzer:
    """
    ⚡ OMEGA-VOID: Real Mathematical Expectancy Calculator.
    
    QUÉ: Calcula la esperanza matemática REAL incorporando todos los costos.
    POR QUÉ: E[R_gross] - Costs = E[R_real]. Si E[R_real] <= 0, NO hay edge.
         Muchos bots parecen rentables en backtest pero son negativos 
         después de costos reales.
    PARA QUÉ: Certificar que E[R_real] > 0 con 95% de confianza estadística.
    CÓMO:
         1. Calcula distribución de retornos con costos reales
         2. Ejecuta 1000 Monte Carlo simulations
         3. Genera intervalos de confianza (95% CI)
         4. Calcula Sharpe/Sortino/Calmar con costos
    CUÁNDO: Después de cada backtest significativo.
    DÓNDE: tests/expectancy_analysis.py → ExpectancyAnalyzer
    QUIÉN: Certification pipeline, QA.
    """
    
    def __init__(
        self,
        cost_model: Optional[CostModel] = None,
        n_simulations: int = 1000,
        confidence_level: float = 0.95,
    ):
        self.costs = cost_model or CostModel()
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
    
    def analyze(
        self,
        trade_returns_pct: np.ndarray,
        trade_durations_hours: Optional[np.ndarray] = None,
        position_sizes_usd: Optional[np.ndarray] = None,
        initial_capital: float = 13.0,
    ) -> Dict:
        """
        Full expectancy analysis with Monte Carlo.
        
        Args:
            trade_returns_pct: Array of gross return percentages per trade
            trade_durations_hours: How long each trade was held
            position_sizes_usd: Notional size of each trade
            initial_capital: Starting capital
            
        Returns:
            Complete analysis report.
        """
        n_trades = len(trade_returns_pct)
        if n_trades < 10:
            return {'error': 'Need at least 10 trades for analysis', 'n_trades': n_trades}
        
        # Default values
        if trade_durations_hours is None:
            trade_durations_hours = np.full(n_trades, 0.5)  # 30min average
        if position_sizes_usd is None:
            position_sizes_usd = np.full(n_trades, initial_capital * 0.1)  # 10% allocation
        
        # ============================================================
        # 1. REAL COST ADJUSTMENT
        # ============================================================
        net_returns_pct = np.zeros(n_trades)
        total_costs = 0.0
        
        for i in range(n_trades):
            notional = position_sizes_usd[i]
            
            # Trading costs
            round_trip = self.costs.round_trip_cost(notional)
            funding = self.costs.funding_cost(notional, trade_durations_hours[i])
            total_cost = round_trip + funding
            total_costs += total_cost
            
            # Net return after costs
            gross_pnl = trade_returns_pct[i] / 100 * notional
            net_pnl = gross_pnl - total_cost
            net_returns_pct[i] = (net_pnl / notional) * 100
        
        # ============================================================
        # 2. EXPECTANCY METRICS
        # ============================================================
        wins = net_returns_pct[net_returns_pct > 0]
        losses = net_returns_pct[net_returns_pct <= 0]
        
        win_rate = len(wins) / n_trades
        loss_rate = len(losses) / n_trades
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        # E[R] = Win_Rate * Avg_Win - Loss_Rate * |Avg_Loss|
        expectancy = win_rate * avg_win - loss_rate * abs(avg_loss)
        
        # Profit Factor = Gross_Wins / |Gross_Losses|
        total_wins = np.sum(wins) if len(wins) > 0 else 0
        total_losses = abs(np.sum(losses)) if len(losses) > 0 else 1e-10
        profit_factor = total_wins / total_losses
        
        # Sharpe Ratio (annualized from per-trade)
        # Assuming ~250 trading days, ~50 trades per day for scalping
        trades_per_year = 250 * 50  # Approximate
        mean_ret = np.mean(net_returns_pct)
        std_ret = np.std(net_returns_pct)
        
        sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 1e-10 else 0
        
        # Sortino Ratio (downside deviation only)
        downside_returns = net_returns_pct[net_returns_pct < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-10
        sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 1e-10 else 0
        
        # ============================================================
        # 3. MONTE CARLO SIMULATION
        # ============================================================
        mc_results = self._monte_carlo(net_returns_pct, initial_capital)
        
        # ============================================================
        # 4. CONFIDENCE INTERVALS
        # ============================================================
        ci_level = self.confidence_level
        ci_lower = np.percentile(mc_results['final_capitals'], (1 - ci_level) / 2 * 100)
        ci_upper = np.percentile(mc_results['final_capitals'], (1 + ci_level) / 2 * 100)
        
        # Probability of profit
        prob_profit = np.mean(mc_results['final_capitals'] > initial_capital)
        
        # Probability of ruin (losing > 50%)
        prob_ruin = np.mean(mc_results['final_capitals'] < initial_capital * 0.5)
        
        report = {
            'n_trades': n_trades,
            'initial_capital': initial_capital,
            'total_costs_usd': round(total_costs, 4),
            'costs_per_trade_usd': round(total_costs / n_trades, 4),
            
            # Expectancy
            'expectancy_pct': round(expectancy, 4),
            'has_edge': expectancy > 0,
            'win_rate_pct': round(win_rate * 100, 1),
            'avg_win_pct': round(avg_win, 4),
            'avg_loss_pct': round(avg_loss, 4),
            'profit_factor': round(profit_factor, 3),
            
            # Risk-Adjusted
            'sharpe_annual': round(sharpe, 3),
            'sortino_annual': round(sortino, 3),
            'mean_return_pct': round(mean_ret, 4),
            'std_return_pct': round(std_ret, 4),
            
            # Monte Carlo
            'mc_simulations': self.n_simulations,
            'mc_median_capital': round(np.median(mc_results['final_capitals']), 2),
            'mc_mean_capital': round(np.mean(mc_results['final_capitals']), 2),
            f'mc_ci_{int(ci_level*100)}_lower': round(ci_lower, 2),
            f'mc_ci_{int(ci_level*100)}_upper': round(ci_upper, 2),
            'mc_worst_case_p5': round(np.percentile(mc_results['final_capitals'], 5), 2),
            'mc_best_case_p95': round(np.percentile(mc_results['final_capitals'], 95), 2),
            'mc_max_drawdowns_avg': round(np.mean(mc_results['max_drawdowns']) * 100, 2),
            
            # Probabilities
            'prob_profit': round(prob_profit * 100, 1),
            'prob_ruin': round(prob_ruin * 100, 1),
            
            # Verdict
            'verdict': self._verdict(expectancy, sharpe, prob_profit, prob_ruin),
        }
        
        return report
    
    def _monte_carlo(
        self,
        returns: np.ndarray,
        initial_capital: float,
    ) -> Dict:
        """
        Runs N Monte Carlo equity curve simulations.
        
        QUÉ: Resamplea los retornos con reemplazo para generar
             1000 posibles trayectorias de equity.
        POR QUÉ: Una sola equity curve no es estadísticamente significativa.
             Monte Carlo muestra la DISTRIBUCIÓN de resultados posibles.
        CÓMO: Bootstrap resampling de la distribución empírica de trades.
        """
        n = len(returns)
        final_capitals = np.zeros(self.n_simulations)
        max_drawdowns = np.zeros(self.n_simulations)
        
        for sim in range(self.n_simulations):
            # Resample with replacement
            sampled = np.random.choice(returns, size=n, replace=True)
            
            # Build equity curve
            equity = initial_capital
            peak = equity
            max_dd = 0.0
            
            for ret in sampled:
                equity *= (1 + ret / 100)
                peak = max(peak, equity)
                dd = (peak - equity) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
                
                if equity <= 0:
                    break
            
            final_capitals[sim] = max(0, equity)
            max_drawdowns[sim] = max_dd
        
        return {
            'final_capitals': final_capitals,
            'max_drawdowns': max_drawdowns,
        }
    
    def _verdict(
        self,
        expectancy: float,
        sharpe: float,
        prob_profit: float,
        prob_ruin: float,
    ) -> str:
        """Generates human-readable verdict."""
        if expectancy <= 0:
            return "❌ NEGATIVE EDGE — NO real edge after costs"
        if sharpe < 1.0:
            return "⚠️ WEAK EDGE — Positive but Sharpe < 1.0"
        if prob_ruin > 10:
            return "⚠️ HIGH RUIN RISK — Edge exists but ruin probability > 10%"
        if prob_profit < 60:
            return "⚠️ LOW CONFIDENCE — Edge exists but < 60% profitable sims"
        if sharpe >= 2.0 and prob_profit >= 80:
            return "✅ INSTITUTIONAL GRADE — Strong edge with high confidence"
        return "✅ VIABLE — Positive edge with acceptable risk"


def generate_sample_trades(n: int = 200, win_rate: float = 0.55) -> np.ndarray:
    """
    Generates realistic sample trade returns for testing.
    
    Simulates a scalping strategy with:
    - Win rate ~55%
    - Avg win ~0.3%
    - Avg loss ~-0.2%
    - Some tail events
    """
    returns = np.zeros(n)
    
    for i in range(n):
        if np.random.random() < win_rate:
            # Win: log-normal distribution (right-skewed)
            returns[i] = np.random.lognormal(-1.2, 0.5)  # ~0.3% avg
        else:
            # Loss: normal distribution
            returns[i] = -abs(np.random.normal(0.2, 0.1))
    
    # Add a few tail events (-2% to -5%)
    n_tails = max(1, n // 50)
    tail_indices = np.random.choice(n, n_tails, replace=False)
    for idx in tail_indices:
        returns[idx] = -np.random.uniform(2, 5)
    
    return returns


def run_expectancy_analysis() -> Dict:
    """Full standalone expectancy analysis."""
    print("=" * 60)
    print("📉 OMEGA-VOID §3.3: Mathematical Expectancy Analysis")
    print("=" * 60)
    
    # Generate sample trades
    trade_returns = generate_sample_trades(n=200, win_rate=0.55)
    
    print(f"\n   Sample Trades: {len(trade_returns)}")
    print(f"   Gross Win Rate: {np.mean(trade_returns > 0)*100:.1f}%")
    print(f"   Gross Avg Return: {np.mean(trade_returns):.4f}%")
    
    # Run analysis
    analyzer = ExpectancyAnalyzer(n_simulations=1000)
    report = analyzer.analyze(
        trade_returns_pct=trade_returns,
        initial_capital=13.0,
    )
    
    if 'error' in report:
        print(f"   ❌ Error: {report['error']}")
        return report
    
    # Display
    print(f"\n   ═══ After Real Costs ═══")
    print(f"   Costs/Trade: ${report['costs_per_trade_usd']:.4f}")
    print(f"   Total Costs: ${report['total_costs_usd']:.4f}")
    print(f"   Net Win Rate: {report['win_rate_pct']:.1f}%")
    print(f"   Expectancy: {report['expectancy_pct']:.4f}%")
    print(f"   Has Edge: {'✅ YES' if report['has_edge'] else '❌ NO'}")
    print(f"   Profit Factor: {report['profit_factor']:.3f}")
    
    print(f"\n   ═══ Risk-Adjusted ═══")
    print(f"   Sharpe (annual): {report['sharpe_annual']:.3f}")
    print(f"   Sortino (annual): {report['sortino_annual']:.3f}")
    
    print(f"\n   ═══ Monte Carlo ({report['mc_simulations']} sims) ═══")
    print(f"   Median Capital: ${report['mc_median_capital']}")
    print(f"   95% CI: [${report.get('mc_ci_95_lower', '?')}, ${report.get('mc_ci_95_upper', '?')}]")
    print(f"   Worst 5%: ${report['mc_worst_case_p5']}")
    print(f"   Best 95%: ${report['mc_best_case_p95']}")
    print(f"   Avg Max Drawdown: {report['mc_max_drawdowns_avg']:.1f}%")
    
    print(f"\n   ═══ Probabilities ═══")
    print(f"   P(Profit): {report['prob_profit']:.1f}%")
    print(f"   P(Ruin >50%): {report['prob_ruin']:.1f}%")
    
    print(f"\n   🏆 {report['verdict']}")
    
    # Save
    report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'expectancy_report.json'
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n   📄 Report saved: {report_path}")
    
    return report


if __name__ == '__main__':
    run_expectancy_analysis()
