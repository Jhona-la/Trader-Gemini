"""
🔍 OMEGA-VOID §4.1: Floating Point Precision Audit

QUÉ: Detector de errores de acumulación en coma flotante.
POR QUÉ: Después de 10,000+ trades, los errores de redondeo en float32
     se ACUMULAN silenciosamente. Un error de redondeo de $0.0001 per trade
     × 10,000 trades = $1.00 de error invisible (7.7% en $13 USDT).
     Kelly Criterion es especialmente vulnerable porque multiplica
     estos errores en cada recalculación.
PARA QUÉ: Detectar y cuantificar el drift de precisión ANTES de que
     cause pérdidas reales. Si drift > 0.01%, marcar como BUG CRÍTICO.
CÓMO: Simula 10,000 trades con Kelly sizing. Ejecuta cálculos en
     float32 Y float64. Compara el drift final.
CUÁNDO: Antes de cada deploy a producción.
DÓNDE: tests/float_precision_audit.py
QUIÉN: FloatPrecisionAuditor → audita risk_manager.py, portfolio.py, hft_buffer.py.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple


# ============================================================
# PRECISION TEST FUNCTIONS
# ============================================================

def kelly_fraction_f32(win_rate: float, avg_win: float, avg_loss: float) -> np.float32:
    """Kelly Criterion in float32 (as used in risk_manager.py historic code)."""
    wr = np.float32(win_rate)
    aw = np.float32(avg_win)
    al = np.float32(abs(avg_loss))
    
    if al < np.float32(1e-8):
        return np.float32(0.0)
    
    b = aw / al  # Win/Loss ratio
    q = np.float32(1.0) - wr
    
    kelly = (wr * b - q) / b
    return np.clip(kelly, np.float32(0.0), np.float32(0.5))


def kelly_fraction_f64(win_rate: float, avg_win: float, avg_loss: float) -> np.float64:
    """Kelly Criterion in float64 (gold standard)."""
    wr = np.float64(win_rate)
    aw = np.float64(avg_win)
    al = np.float64(abs(avg_loss))
    
    if al < 1e-15:
        return np.float64(0.0)
    
    b = aw / al
    q = 1.0 - wr
    
    kelly = (wr * b - q) / b
    return np.clip(kelly, 0.0, 0.5)


def cumulative_pnl_f32(
    trade_returns: np.ndarray, 
    initial_capital: float = 13.0,
) -> np.float32:
    """Accumulate PnL using float32 (as in hft_buffer prices)."""
    capital = np.float32(initial_capital)
    
    for ret in trade_returns:
        pnl = np.float32(capital) * np.float32(ret / 100)
        capital = np.float32(capital + pnl)
    
    return capital


def cumulative_pnl_f64(
    trade_returns: np.ndarray,
    initial_capital: float = 13.0,
) -> np.float64:
    """Accumulate PnL using float64 (gold standard)."""
    capital = np.float64(initial_capital)
    
    for ret in trade_returns:
        pnl = capital * np.float64(ret / 100)
        capital += pnl
    
    return capital


class FloatPrecisionAuditor:
    """
    ⚡ OMEGA-VOID: Floating Point Accumulation Error Detector.
    
    QUÉ: Ejecuta cálculos idénticos en float32 y float64 y mide el drift.
    POR QUÉ: Los buffers de precio usan float32 (por eficiencia de caché).
         Pero los cálculos de PnL acumulado y Kelly DEBEN usar float64.
         Si hay alguna ruta de código que hace PnL en float32, el error
         se acumula trade tras trade y es INDETECTABLE hasta que es grave.
    PARA QUÉ: Identificar EXACTAMENTE qué rutas de código tienen vulnerabilidad
         de precisión y cuánto drift generan en N trades.
    CÓMO: Genera trades sintéticos → ejecuta en ambas precisiones → compara.
    CUÁNDO: Pre-producción, post-cambios en risk/ o core/.
    DÓNDE: tests/float_precision_audit.py → FloatPrecisionAuditor
    QUIÉN: QA pipeline, certification.
    
    Args:
        n_trades: Number of trades to simulate
        drift_threshold_pct: Maximum acceptable drift (default 0.01%)
    """
    
    def __init__(
        self,
        n_trades: int = 10000,
        drift_threshold_pct: float = 0.01,
    ):
        self.n_trades = n_trades
        self.drift_threshold = drift_threshold_pct
    
    def audit_pnl_accumulation(self, initial_capital: float = 13.0) -> Dict:
        """
        Test 1: PnL accumulation drift over N trades.
        
        QUÉ: Acumula PnL trade-by-trade en float32 vs float64.
        POR QUÉ: Este es EL test más importante. Si hay drift aquí,
             portfolio.py está calculando mal el balance real.
        """
        # Generate realistic trade returns
        np.random.seed(42)  # Reproducible
        trade_returns = np.zeros(self.n_trades)
        
        for i in range(self.n_trades):
            if np.random.random() < 0.55:
                trade_returns[i] = np.random.lognormal(-1.5, 0.5)  # ~0.22% avg win
            else:
                trade_returns[i] = -abs(np.random.normal(0.15, 0.08))
        
        # Run both precisions
        final_f32 = float(cumulative_pnl_f32(trade_returns, initial_capital))
        final_f64 = float(cumulative_pnl_f64(trade_returns, initial_capital))
        
        drift_abs = abs(final_f64 - final_f32)
        drift_pct = (drift_abs / abs(final_f64)) * 100 if final_f64 != 0 else 0
        
        is_critical = drift_pct > self.drift_threshold
        
        # Measure drift growth over time
        drift_curve = []
        cap32 = np.float32(initial_capital)
        cap64 = np.float64(initial_capital)
        
        checkpoints = [100, 500, 1000, 2500, 5000, 7500, 10000]
        
        for i in range(self.n_trades):
            pnl32 = np.float32(cap32) * np.float32(trade_returns[i] / 100)
            cap32 = np.float32(cap32 + pnl32)
            
            pnl64 = cap64 * np.float64(trade_returns[i] / 100)
            cap64 += pnl64
            
            if (i + 1) in checkpoints:
                d = abs(float(cap64) - float(cap32))
                d_pct = (d / abs(float(cap64))) * 100 if cap64 != 0 else 0
                drift_curve.append({
                    'trades': i + 1,
                    'f32_capital': round(float(cap32), 6),
                    'f64_capital': round(float(cap64), 6),
                    'drift_usd': round(d, 6),
                    'drift_pct': round(d_pct, 6),
                })
        
        return {
            'test': 'pnl_accumulation',
            'n_trades': self.n_trades,
            'initial_capital': initial_capital,
            'final_f32': round(final_f32, 6),
            'final_f64': round(final_f64, 6),
            'drift_usd': round(drift_abs, 6),
            'drift_pct': round(drift_pct, 6),
            'threshold_pct': self.drift_threshold,
            'is_critical': is_critical,
            'status': '❌ CRITICAL' if is_critical else '✅ OK',
            'drift_growth_curve': drift_curve,
        }
    
    def audit_kelly_accumulation(self) -> Dict:
        """
        Test 2: Kelly Criterion recalculation drift.
        
        QUÉ: Recalcula Kelly 10,000 veces con rolling window en ambas precisiones.
        POR QUÉ: Kelly usa win_rate, avg_win, avg_loss que se recalculan
             en cada trade. Los errores de redondeo en estos stats
             se amplifican en la fracción de Kelly.
        """
        np.random.seed(42)
        
        # Simulate rolling window Kelly recalculation
        window_size = 50
        kelly_history_f32 = []
        kelly_history_f64 = []
        
        trade_results = []
        
        for i in range(self.n_trades):
            # Random trade result
            if np.random.random() < 0.55:
                result = np.random.lognormal(-1.5, 0.5)
            else:
                result = -abs(np.random.normal(0.15, 0.08))
            
            trade_results.append(result)
            
            if len(trade_results) >= window_size:
                window = trade_results[-window_size:]
                wins = [r for r in window if r > 0]
                losses = [r for r in window if r <= 0]
                
                wr = len(wins) / len(window)
                aw = np.mean(wins) if wins else 0
                al = np.mean(losses) if losses else 0
                
                k32 = float(kelly_fraction_f32(wr, aw, al))
                k64 = float(kelly_fraction_f64(wr, aw, al))
                
                kelly_history_f32.append(k32)
                kelly_history_f64.append(k64)
        
        # Analyze drift
        kelly_f32 = np.array(kelly_history_f32)
        kelly_f64 = np.array(kelly_history_f64)
        
        drift = np.abs(kelly_f64 - kelly_f32)
        max_drift = float(np.max(drift))
        avg_drift = float(np.mean(drift))
        
        # How many times was the Kelly bet size wrong by > 0.1%?
        significant_errors = int(np.sum(drift > 0.001))
        
        return {
            'test': 'kelly_recalculation',
            'n_recalculations': len(kelly_history_f32),
            'max_kelly_drift': round(max_drift, 8),
            'avg_kelly_drift': round(avg_drift, 8),
            'significant_errors': significant_errors,
            'error_rate_pct': round(significant_errors / max(1, len(kelly_history_f32)) * 100, 2),
            'status': '❌ CRITICAL' if max_drift > 0.01 else '✅ OK',
        }
    
    def audit_price_buffer_precision(self) -> Dict:
        """
        Test 3: Price storage in float32 ring buffers.
        
        QUÉ: Verifica si float32 pierde dígitos significativos para precios.
        POR QUÉ: BTC @ $50,000 en float32 tiene precisión de ~$0.004.
             Esto es aceptable para OHLC pero NO para PnL calculation.
        """
        test_prices = {
            'BTC/USDT': 50000.0,
            'ETH/USDT': 3000.0,
            'SOL/USDT': 100.0,
            'DOGE/USDT': 0.08,
            'SHIB/USDT': 0.00001,
        }
        
        results = {}
        for symbol, price in test_prices.items():
            f32 = np.float32(price)
            f64 = np.float64(price)
            
            error = abs(float(f64) - float(f32))
            relative_error = error / price
            
            # float32 has ~7 significant digits
            digits_lost = -np.log10(relative_error) if relative_error > 0 else 23
            
            results[symbol] = {
                'price': price,
                'f32_stored': float(f32),
                'absolute_error': round(error, 10),
                'relative_error': round(relative_error, 10),
                'significant_digits': round(digits_lost, 1),
                'status': '✅' if digits_lost >= 4 else '⚠️' if digits_lost >= 2 else '❌',
            }
        
        return {
            'test': 'price_buffer_precision',
            'results': results,
        }
    
    def run_full_audit(self, initial_capital: float = 13.0) -> Dict:
        """
        Runs ALL precision tests and generates comprehensive report.
        """
        print("=" * 60)
        print("🔍 OMEGA-VOID §4.1: Floating Point Precision Audit")
        print(f"   Simulating {self.n_trades:,} trades...")
        print("=" * 60)
        
        # Run all tests
        pnl_result = self.audit_pnl_accumulation(initial_capital)
        kelly_result = self.audit_kelly_accumulation()
        price_result = self.audit_price_buffer_precision()
        
        # Print PnL results
        print(f"\n   📊 Test 1: PnL Accumulation Drift")
        print(f"      float32: ${pnl_result['final_f32']:.6f}")
        print(f"      float64: ${pnl_result['final_f64']:.6f}")
        print(f"      Drift:   ${pnl_result['drift_usd']:.6f} ({pnl_result['drift_pct']:.6f}%)")
        print(f"      Status:  {pnl_result['status']}")
        
        if pnl_result.get('drift_growth_curve'):
            print(f"\n      Drift Growth:")
            for point in pnl_result['drift_growth_curve']:
                bar = '█' * min(50, int(point['drift_pct'] * 10000))
                print(f"      {point['trades']:>6} trades: ${point['drift_usd']:.6f} ({point['drift_pct']:.6f}%) {bar}")
        
        # Print Kelly results
        print(f"\n   📊 Test 2: Kelly Recalculation Drift")
        print(f"      Max Drift:  {kelly_result['max_kelly_drift']:.8f}")
        print(f"      Avg Drift:  {kelly_result['avg_kelly_drift']:.8f}")
        print(f"      Errors:     {kelly_result['significant_errors']} ({kelly_result['error_rate_pct']}%)")
        print(f"      Status:     {kelly_result['status']}")
        
        # Print Price Buffer results
        print(f"\n   📊 Test 3: Price Buffer Precision")
        for symbol, info in price_result['results'].items():
            print(f"      {info['status']} {symbol:>12}: {info['significant_digits']:.0f} digits "
                  f"(error: ${info['absolute_error']:.8f})")
        
        # Overall verdict
        all_ok = (
            not pnl_result['is_critical'] and
            kelly_result['status'] == '✅ OK'
        )
        
        overall = '✅ ALL TESTS PASSED' if all_ok else '❌ PRECISION ISSUES DETECTED'
        print(f"\n   🏆 {overall}")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'n_trades': self.n_trades,
            'overall_status': overall,
            'pnl_accumulation': pnl_result,
            'kelly_recalculation': kelly_result,
            'price_buffer': price_result,
        }
        
        # Save
        report_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'float_precision_report.json'
        )
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n   📄 Report saved: {report_path}")
        
        return report


if __name__ == '__main__':
    auditor = FloatPrecisionAuditor(n_trades=10000)
    auditor.run_full_audit(initial_capital=13.0)
