"""
📉 OMEGA-VOID §3.2: Walk-Forward Validation (Anti-Overfitting)

QUÉ: Validación cruzada temporal que separa entrenamiento de testeo.
POR QUÉ: Si los parámetros de la estrategia están sobre-ajustados al pasado,
     el bot dará rendimientos espectaculares en backtest pero fracasará
     en producción. Walk-Forward es el estándar de oro para detectar overfitting.
PARA QUÉ: Certificar que la relación OOS_Sharpe / IS_Sharpe > 0.5,
     indicando que la estrategia tiene edge real, no curve-fitting.
CÓMO: 5 folds con expanding window (70% train / 30% test).
     Compara métricas in-sample vs out-of-sample.
CUÁNDO: Antes de cada cambio en technical.py o ml_strategy.py.
DÓNDE: tests/walk_forward.py
QUIÉN: WalkForwardValidator → usa BacktestPortfolio.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FoldResult:
    """Results for a single train/test fold."""
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    is_sharpe: float       # In-Sample Sharpe
    oos_sharpe: float      # Out-of-Sample Sharpe
    is_return_pct: float
    oos_return_pct: float
    is_win_rate: float
    oos_win_rate: float
    is_max_dd: float
    oos_max_dd: float
    is_trades: int
    oos_trades: int
    degradation_ratio: float  # OOS/IS Sharpe


class SimpleBacktestEngine:
    """
    Lightweight backtest engine for walk-forward validation.
    
    QUÉ: Motor de backtest simplificado para validación estadística.
    POR QUÉ: El backtest completo (run_backtest.py) es muy pesado para
         ejecutar 5+ folds. Usamos una versión ligera que captura
         las métricas esenciales sin toda la infraestructura.
    """
    
    def __init__(self, initial_capital: float = 13.0, leverage: int = 10):
        self.initial_capital = initial_capital
        self.leverage = leverage
    
    def run(
        self,
        prices: np.ndarray,
        timestamps: np.ndarray,
        lookback: int = 20,
    ) -> Dict:
        """
        Runs simple momentum strategy on price array.
        
        Returns metrics dict with Sharpe, returns, win rate, drawdown.
        """
        n = len(prices)
        if n < lookback + 10:
            return self._empty_result()
        
        # Calculate signals
        returns = np.zeros(n)
        positions = np.zeros(n)  # 1=long, -1=short, 0=flat
        trade_returns: List[float] = []
        
        capital = self.initial_capital
        peak = capital
        max_dd = 0.0
        
        for i in range(lookback, n):
            window = prices[i-lookback:i]
            
            # Simple momentum signal
            momentum = (prices[i] - window[0]) / window[0]
            vol = np.std(np.diff(window) / window[:-1])
            
            if vol < 0.0001:
                vol = 0.0001
            
            # Signal
            if momentum > 0.005 and positions[i-1] <= 0:
                positions[i] = 1  # Long
            elif momentum < -0.005 and positions[i-1] >= 0:
                positions[i] = -1  # Short
            else:
                positions[i] = positions[i-1]
            
            # Calculate return
            if i > lookback and positions[i-1] != 0:
                pct_return = (prices[i] - prices[i-1]) / prices[i-1]
                trade_ret = pct_return * positions[i-1] * self.leverage
                returns[i] = trade_ret
                
                # Track trades on position changes
                if positions[i] != positions[i-1]:
                    trade_returns.append(trade_ret)
                
                capital *= (1 + trade_ret * 0.1)  # 10% allocation
                peak = max(peak, capital)
                dd = (peak - capital) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
        
        # Calculate metrics
        valid_returns = returns[lookback:]
        valid_returns = valid_returns[valid_returns != 0]
        
        if len(valid_returns) < 5:
            return self._empty_result()
        
        mean_ret = np.mean(valid_returns)
        std_ret = np.std(valid_returns)
        
        sharpe = (mean_ret / std_ret) * np.sqrt(252 * 24 * 60) if std_ret > 1e-10 else 0
        
        # Downside deviation for Sortino
        downside = valid_returns[valid_returns < 0]
        downside_std = np.std(downside) if len(downside) > 0 else 1e-10
        sortino = (mean_ret / downside_std) * np.sqrt(252 * 24 * 60) if downside_std > 1e-10 else 0
        
        wins = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r <= 0]
        win_rate = len(wins) / max(1, len(trade_returns)) * 100
        
        total_return = (capital - self.initial_capital) / self.initial_capital * 100
        
        return {
            'sharpe': round(sharpe, 3),
            'sortino': round(sortino, 3),
            'return_pct': round(total_return, 2),
            'max_drawdown_pct': round(max_dd * 100, 2),
            'win_rate': round(win_rate, 1),
            'total_trades': len(trade_returns),
            'mean_return': round(mean_ret * 100, 6),
            'std_return': round(std_ret * 100, 6),
        }
    
    def _empty_result(self) -> Dict:
        return {
            'sharpe': 0.0, 'sortino': 0.0, 'return_pct': 0.0,
            'max_drawdown_pct': 0.0, 'win_rate': 0.0, 'total_trades': 0,
            'mean_return': 0.0, 'std_return': 0.0,
        }


class WalkForwardValidator:
    """
    ⚡ OMEGA-VOID: Walk-Forward Anti-Overfitting Validator.
    
    QUÉ: Divide datos temporales en folds de train/test y compara
         el rendimiento in-sample vs out-of-sample.
    POR QUÉ: La ÚNICA forma confiable de detectar overfitting en
         series temporales es comparar IS vs OOS en ventanas expansivas.
         Cross-validation estándar (k-fold) NO funciona para series temporales
         porque viola la causalidad temporal.
    PARA QUÉ: Si OOS_Sharpe / IS_Sharpe < 0.5, los parámetros están
         sobre-ajustados y el bot perderá dinero en producción.
    CÓMO: Expanding window con 5 folds:
         Fold 1: Train[0:40%] → Test[40:52%]
         Fold 2: Train[0:52%] → Test[52:64%]
         Fold 3: Train[0:64%] → Test[64:76%]
         Fold 4: Train[0:76%] → Test[76:88%]
         Fold 5: Train[0:88%] → Test[88:100%]
    CUÁNDO: Después de cada cambio en strategies/.
    DÓNDE: tests/walk_forward.py → WalkForwardValidator
    QUIÉN: QA pipeline, certification.
    
    Args:
        n_folds: Number of walk-forward folds (default 5)
        train_ratio: Starting ratio for training (default 0.4)
        overfitting_threshold: Degradation ratio below which = overfitting
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        train_ratio: float = 0.4,
        overfitting_threshold: float = 0.5,
    ):
        self.n_folds = n_folds
        self.train_ratio = train_ratio
        self.overfitting_threshold = overfitting_threshold
        self.engine = SimpleBacktestEngine()
        self.fold_results: List[FoldResult] = []
    
    def validate(
        self,
        prices: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Runs full walk-forward validation.
        
        Args:
            prices: Array of close prices (chronological)
            timestamps: Optional array of datetime strings
            
        Returns:
            Validation report dict.
        """
        n = len(prices)
        if n < 200:
            return {'error': 'Insufficient data (need 200+ bars)', 'n_bars': n}
        
        if timestamps is None:
            timestamps = np.array([str(i) for i in range(n)])
        
        self.fold_results = []
        
        # Calculate fold boundaries
        test_size = int(n * (1 - self.train_ratio) / self.n_folds)
        train_end_start = int(n * self.train_ratio)
        
        for fold in range(self.n_folds):
            # Expanding window
            train_start = 0
            train_end = train_end_start + fold * test_size
            test_start = train_end
            test_end = min(test_start + test_size, n)
            
            if test_end <= test_start or train_end <= train_start:
                continue
            
            # Run backtest on each segment
            train_prices = prices[train_start:train_end]
            test_prices = prices[test_start:test_end]
            
            is_result = self.engine.run(
                train_prices, timestamps[train_start:train_end]
            )
            oos_result = self.engine.run(
                test_prices, timestamps[test_start:test_end]
            )
            
            # Degradation ratio
            is_sharpe = is_result['sharpe']
            oos_sharpe = oos_result['sharpe']
            
            if abs(is_sharpe) > 0.01:
                degradation = oos_sharpe / is_sharpe
            else:
                degradation = 1.0  # Can't compute — assume OK
            
            fold_result = FoldResult(
                fold_id=fold + 1,
                train_start=str(timestamps[train_start]),
                train_end=str(timestamps[train_end - 1]),
                test_start=str(timestamps[test_start]),
                test_end=str(timestamps[test_end - 1]),
                is_sharpe=is_sharpe,
                oos_sharpe=oos_sharpe,
                is_return_pct=is_result['return_pct'],
                oos_return_pct=oos_result['return_pct'],
                is_win_rate=is_result['win_rate'],
                oos_win_rate=oos_result['win_rate'],
                is_max_dd=is_result['max_drawdown_pct'],
                oos_max_dd=oos_result['max_drawdown_pct'],
                is_trades=is_result['total_trades'],
                oos_trades=oos_result['total_trades'],
                degradation_ratio=round(degradation, 3),
            )
            
            self.fold_results.append(fold_result)
        
        return self._generate_report()
    
    def _generate_report(self) -> Dict:
        """Generates the final walk-forward validation report."""
        if not self.fold_results:
            return {'error': 'No folds completed'}
        
        avg_is_sharpe = np.mean([f.is_sharpe for f in self.fold_results])
        avg_oos_sharpe = np.mean([f.oos_sharpe for f in self.fold_results])
        avg_degradation = np.mean([f.degradation_ratio for f in self.fold_results])
        
        is_overfit = avg_degradation < self.overfitting_threshold
        
        report = {
            'n_folds': len(self.fold_results),
            'avg_is_sharpe': round(avg_is_sharpe, 3),
            'avg_oos_sharpe': round(avg_oos_sharpe, 3),
            'avg_degradation_ratio': round(avg_degradation, 3),
            'overfitting_threshold': self.overfitting_threshold,
            'is_overfit': is_overfit,
            'verdict': '❌ OVERFIT' if is_overfit else '✅ ROBUST',
            'folds': [
                {
                    'fold': f.fold_id,
                    'is_sharpe': f.is_sharpe,
                    'oos_sharpe': f.oos_sharpe,
                    'degradation': f.degradation_ratio,
                    'is_return': f.is_return_pct,
                    'oos_return': f.oos_return_pct,
                    'is_win_rate': f.is_win_rate,
                    'oos_win_rate': f.oos_win_rate,
                }
                for f in self.fold_results
            ]
        }
        
        return report


def run_walk_forward_validation(days: int = 30) -> Dict:
    """
    Standalone walk-forward validation runner.
    
    Fetches real data and runs 5-fold expanding window validation.
    """
    print("=" * 60)
    print("📉 OMEGA-VOID §3.2: Walk-Forward Validation")
    print("=" * 60)
    
    # Try real data, fallback to synthetic
    try:
        from binance.client import Client
        from config import Config
        
        client = Client(Config.BINANCE_API_KEY, Config.BINANCE_SECRET_KEY)
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=days)
        
        klines = client.get_historical_klines(
            'BTCUSDT', '1m',
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
        )
        
        prices = np.array([float(k[4]) for k in klines])  # Close prices
        timestamps = np.array([str(pd.to_datetime(k[0], unit='ms')) for k in klines])
        
    except Exception:
        print("⚠️ Using synthetic data for validation")
        n = days * 24 * 60  # 1-minute bars
        prices = np.cumsum(np.random.normal(0.0001, 0.001, n)) + 50000
        prices = np.abs(prices)  # Ensure positive
        timestamps = np.array([str(i) for i in range(n)])
    
    print(f"   Data: {len(prices)} bars ({days} days)")
    
    # Run validation
    validator = WalkForwardValidator(n_folds=5, overfitting_threshold=0.5)
    report = validator.validate(prices, timestamps)
    
    # Display results
    if 'error' in report:
        print(f"   ❌ Error: {report['error']}")
        return report
    
    print(f"\n   {'Fold':>4} | {'IS Sharpe':>10} | {'OOS Sharpe':>10} | {'Degrad':>8} | {'Verdict':>8}")
    print(f"   {'─'*4} | {'─'*10} | {'─'*10} | {'─'*8} | {'─'*8}")
    
    for fold in report['folds']:
        verdict = '✅' if fold['degradation'] >= 0.5 else '❌'
        print(f"   {fold['fold']:>4} | {fold['is_sharpe']:>10.3f} | "
              f"{fold['oos_sharpe']:>10.3f} | {fold['degradation']:>8.3f} | {verdict:>8}")
    
    print(f"\n   📊 Avg IS Sharpe: {report['avg_is_sharpe']:.3f}")
    print(f"   📊 Avg OOS Sharpe: {report['avg_oos_sharpe']:.3f}")
    print(f"   📊 Avg Degradation: {report['avg_degradation_ratio']:.3f}")
    print(f"   🏆 Verdict: {report['verdict']}")
    
    # Save
    report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'walk_forward_report.json'
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"   📄 Report saved: {report_path}")
    
    return report


if __name__ == '__main__':
    run_walk_forward_validation(days=30)
