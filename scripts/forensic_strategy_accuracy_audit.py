"""
🔬 AUDITORÍA FORENSE DE PRECISIÓN DE ESTRATEGIAS v1.0
=====================================================
Trader Gemini - Análisis Post-Mortem Profundo

QUÉ: Script de auditoría que analiza con máxima granularidad:
  1. ¿Qué tan precisas son nuestras predicciones de dirección?
  2. ¿Durante cuánto tiempo tenemos razón antes de que el precio se voltee?
  3. ¿Por qué perdemos dinero INCLUSO cuando acertamos la dirección?
  4. ¿Cuál es el "prediction decay" — cuánto dura nuestro edge?
  5. ¿Qué causa las pérdidas: comisiones, SL prematuro, o falta de alcance del TP?

POR QUÉ: Sabemos que el sistema tiene poder predictivo pero las operaciones
  pierden dinero. Hay una desconexión entre "acertar la dirección" y
  "ganar dinero". Este audit revela EXACTAMENTE dónde está la fuga.

PARA QUÉ: Con los resultados podremos:
  - Ajustar SL/TP para que se alineen con la duración real del edge
  - Detectar si las comisiones "comen" el edge predictivo
  - Saber cuánto tiempo mantener una posición antes de que el edge se agote
  - Identificar si el trailing stop nos saca prematuramente

CÓMO: Para cada trade:
  1. Simula la entrada con la señal de producción
  2. Rastrea tick-by-tick el MFE (Maximum Favorable Excursion) y MAE
  3. Calcula el "Direction Accuracy Window" — cuántas barras tuvimos razón
  4. Compara el PnL si hubiéramos salido en el punto óptimo vs donde salimos
  5. Desglosa la pérdida en: comisiones, SL prematuro, TP inalcanzable

CUÁNDO: Ejecutar antes de ir a producción como validación definitiva.
DÓNDE: scripts/forensic_strategy_accuracy_audit.py
QUIÉN: Usa las mismas funciones de producción que run_multi_horizon_backtest.py
"""

import sys
import os

# Thread safety FIRST
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import time
import json
import math
import warnings
warnings.filterwarnings('ignore')

from config import Config

# Import backtest functions (SAME as production — no divergence)
from scripts.run_god_mode_backtest import (
    fetch_data, compute_indicators, calibrate_sl_tp,
    signal_technical, SophiaClusterEngine, WalkForwardXGBoost,
    detect_regime, HORIZON_PROFILES, COMMISSION_PCT, STRATEGY_SPECIALIZATION_MAP
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE LA AUDITORÍA
# ══════════════════════════════════════════════════════════════════════════════

AUDIT_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']  # Top 3 activos
AUDIT_HORIZON = 1  # 1 día (scalping)
AUDIT_LEVERAGE = Config.BINANCE_LEVERAGE
AUDIT_INITIAL_CAPITAL = Config.INITIAL_CAPITAL
AUDIT_DAYS = 9  # 9 días de historia

# Forward windows para medir "durante cuánto tiempo acertamos"
FORWARD_WINDOWS = [1, 3, 5, 10, 15, 30, 60, 120, 300, 600]  # barras


# ══════════════════════════════════════════════════════════════════════════════
# CLASE PRINCIPAL: ForensicAccuracyAuditor
# ══════════════════════════════════════════════════════════════════════════════

class ForensicAccuracyAuditor:
    """
    🔬 Auditor Forense de Precisión de Estrategias
    
    Analiza cada señal generada y rastrea:
    - Direction Accuracy: ¿acertamos la dirección?
    - Temporal Decay: ¿durante cuánto tiempo estuvimos en lo correcto?
    - MFE/MAE Analysis: ¿cuál fue el mejor/peor momento del trade?
    - Commission Bleed: ¿las comisiones eliminan el edge?
    - Exit Cause Analysis: ¿por qué se cerró el trade?
    """
    
    def __init__(self):
        self.signals = []  # Todas las señales generadas
        self.trades = []   # Trades completados con análisis detallado
        self.prediction_accuracy = {
            'total_signals': 0,
            'direction_correct': 0,
            'direction_incorrect': 0,
            # Por ventana temporal
            'accuracy_by_window': {w: {'correct': 0, 'total': 0} for w in FORWARD_WINDOWS}
        }
        
    def analyze_signal_accuracy(self, df, signal_idx, direction, sl_pct, tp_pct, 
                                  entry_price, size_usd, strategy_name):
        """
        📊 Análisis forense completo de una señal individual.
        
        Para cada señal:
        1. ¿La dirección fue correcta? (forward return positivo en la dirección)
        2. ¿Durante cuántas barras fue correcta? (pre-decay window)
        3. ¿Cuál fue el MFE y MAE real?
        4. ¿Qué habría pasado con diferentes SL/TP?
        5. ¿Las comisiones destruyeron una operación ganadora?
        """
        
        rows = df.reset_index()
        total_bars = len(rows)
        remaining = total_bars - signal_idx
        
        if remaining < 10:
            return None
            
        close_at_entry = entry_price
        
        # === 1. DIRECTION ACCURACY POR VENTANA TEMPORAL ===
        direction_results = {}
        for window in FORWARD_WINDOWS:
            target_idx = signal_idx + window
            if target_idx >= total_bars:
                continue
            
            future_close = rows.iloc[target_idx]['close']
            if direction == 'long':
                is_correct = future_close > close_at_entry
                raw_return = (future_close - close_at_entry) / close_at_entry
            else:
                is_correct = future_close < close_at_entry
                raw_return = (close_at_entry - future_close) / close_at_entry
            
            direction_results[window] = {
                'correct': is_correct,
                'raw_return_pct': raw_return * 100,
                'net_return_pct': (raw_return - COMMISSION_PCT * 2) * 100,
            }
            
            self.prediction_accuracy['accuracy_by_window'][window]['total'] += 1
            if is_correct:
                self.prediction_accuracy['accuracy_by_window'][window]['correct'] += 1
        
        # === 2. MFE/MAE TICK-BY-TICK (hasta 600 barras forward) ===
        max_forward = min(600, remaining - 1)
        
        mfe = 0.0  # Maximum Favorable Excursion (best unrealized profit)
        mae = 0.0  # Maximum Adverse Excursion (worst unrealized loss)
        mfe_bar = 0
        mae_bar = 0
        
        # Tracking: ¿cuántas barras consecutivas estuvimos "en profit"?
        consecutive_profit_bars = 0
        max_consecutive_profit = 0
        first_profit_bar = None
        first_adverse_bar = None
        
        # Tracking: ¿cuántas barras hasta que el precio toca nuestros SL/TP?
        sl_hit_bar = None
        tp_hit_bar = None
        
        sl_price = close_at_entry * (1 - sl_pct) if direction == 'long' else close_at_entry * (1 + sl_pct)
        tp_price = close_at_entry * (1 + tp_pct) if direction == 'long' else close_at_entry * (1 - tp_pct)
        
        # "Optimal exit" tracking
        optimal_exit_pnl = 0.0
        optimal_exit_bar = 0
        
        bar_by_bar_pnl = []
        
        for j in range(1, max_forward + 1):
            bar = rows.iloc[signal_idx + j]
            high = bar['high']
            low = bar['low']
            close = bar['close']
            
            if direction == 'long':
                unrealized_pct = (close - close_at_entry) / close_at_entry
                bar_mfe = (high - close_at_entry) / close_at_entry
                bar_mae = (low - close_at_entry) / close_at_entry
                
                if sl_hit_bar is None and low <= sl_price:
                    sl_hit_bar = j
                if tp_hit_bar is None and high >= tp_price:
                    tp_hit_bar = j
            else:
                unrealized_pct = (close_at_entry - close) / close_at_entry
                bar_mfe = (close_at_entry - low) / close_at_entry
                bar_mae = (close_at_entry - high) / close_at_entry
                
                if sl_hit_bar is None and high >= sl_price:
                    sl_hit_bar = j
                if tp_hit_bar is None and low <= tp_price:
                    tp_hit_bar = j
            
            # Track MFE/MAE
            if bar_mfe > mfe:
                mfe = bar_mfe
                mfe_bar = j
            if bar_mae < mae:
                mae = bar_mae
                mae_bar = j
            
            # Track consecutive profit
            if unrealized_pct > 0:
                consecutive_profit_bars += 1
                max_consecutive_profit = max(max_consecutive_profit, consecutive_profit_bars)
                if first_profit_bar is None:
                    first_profit_bar = j
            else:
                consecutive_profit_bars = 0
                if first_adverse_bar is None and unrealized_pct < -COMMISSION_PCT * 2:
                    first_adverse_bar = j
            
            # Track optimal exit (best net PnL moment)
            net_pnl = unrealized_pct - COMMISSION_PCT * 2  # minus round-trip fees
            if net_pnl > optimal_exit_pnl:
                optimal_exit_pnl = net_pnl
                optimal_exit_bar = j
            
            bar_by_bar_pnl.append({
                'bar': j,
                'unrealized_pct': unrealized_pct * 100,
                'net_pct': net_pnl * 100,
                'cumulative_mfe': mfe * 100,
                'cumulative_mae': mae * 100,
            })
        
        # === 3. ANÁLISIS DE PÉRDIDA CAUSAL ===
        round_trip_fee = COMMISSION_PCT * 2
        round_trip_fee_usd = size_usd * round_trip_fee
        
        # ¿Qué tipo de pérdida fue?
        loss_cause = 'NONE'
        if sl_hit_bar is not None and (tp_hit_bar is None or sl_hit_bar < tp_hit_bar):
            # SL se tocó primero
            if mfe > round_trip_fee:
                loss_cause = 'SL_PREMATURO'  # Tuvimos profit pero el SL nos sacó
            elif abs(mae) < sl_pct * 0.5:
                loss_cause = 'COMISION_BLEED'  # Nunca cayó mucho, pero fees >edge
            else:
                loss_cause = 'DIRECCION_INCORRECTA'  # Simplemente nos equivocamos
        elif tp_hit_bar is not None:
            loss_cause = 'TP_ALCANZADO'  # Trade exitoso
        else:
            # Ni SL ni TP se tocaron en 600 barras
            if optimal_exit_pnl > 0:
                loss_cause = 'TP_INALCANZABLE'  # Hubo profit pero TP era demasiado lejano
            else:
                loss_cause = 'ESTANCAMIENTO'  # El precio no se movió suficiente
        
        # === 4. PREDICTION DECAY CURVE ===
        # ¿A partir de qué barra empezamos a estar en contra?
        decay_bar = None
        for j in range(len(bar_by_bar_pnl)):
            if j > 5 and bar_by_bar_pnl[j]['unrealized_pct'] < 0:
                # Si después de 5 barras de profit estamos en negativo
                if all(b['unrealized_pct'] < 0 for b in bar_by_bar_pnl[j:min(j+5, len(bar_by_bar_pnl))]):
                    decay_bar = j
                    break
        
        # === 5. WHAT-IF ANALYSIS ===
        # ¿Qué habría pasado con diferentes SL/TP?
        what_if_results = {}
        for sl_mult in [0.5, 0.75, 1.0, 1.5, 2.0]:
            for tp_mult in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
                test_sl = sl_pct * sl_mult
                test_tp = tp_pct * tp_mult
                
                # Simular trade con estos SL/TP
                sim_result = self._simulate_exit(
                    rows, signal_idx, direction, close_at_entry,
                    test_sl, test_tp, max_bars=min(300, max_forward)
                )
                key = f"SL{sl_mult}x_TP{tp_mult}x"
                what_if_results[key] = sim_result
        
        # === COMPILAR RESULTADO ===
        result = {
            'timestamp': str(rows.iloc[signal_idx].get('datetime', signal_idx)),
            'strategy': strategy_name,
            'direction': direction,
            'entry_price': close_at_entry,
            'sl_pct': sl_pct * 100,
            'tp_pct': tp_pct * 100,
            'sl_price': sl_price,
            'tp_price': tp_price,
            
            # Direction Accuracy
            'direction_accuracy': direction_results,
            
            # MFE/MAE
            'mfe_pct': mfe * 100,
            'mfe_bar': mfe_bar,
            'mae_pct': mae * 100,
            'mae_bar': mae_bar,
            
            # Temporal Analysis
            'first_profit_bar': first_profit_bar,
            'first_adverse_bar': first_adverse_bar,
            'max_consecutive_profit_bars': max_consecutive_profit,
            'sl_hit_bar': sl_hit_bar,
            'tp_hit_bar': tp_hit_bar,
            'optimal_exit_bar': optimal_exit_bar,
            'optimal_exit_pnl_pct': optimal_exit_pnl * 100,
            'prediction_decay_bar': decay_bar,
            
            # Loss Cause
            'loss_cause': loss_cause,
            
            # Commission Analysis
            'round_trip_fee_pct': round_trip_fee * 100,
            'fee_vs_mfe_ratio': (round_trip_fee / mfe * 100) if mfe > 0 else float('inf'),
            'fee_destroys_edge': mfe > 0 and mfe < round_trip_fee,
            
            # Optimal vs Actual
            'missed_profit_pct': optimal_exit_pnl * 100 if loss_cause != 'TP_ALCANZADO' else 0,
            
            # What-If Matrix (resumen)
            'what_if_best': max(what_if_results.items(), key=lambda x: x[1].get('pnl_pct', -999)) if what_if_results else None,
        }
        
        self.signals.append(result)
        self.prediction_accuracy['total_signals'] += 1
        
        # Quick direction check (60-bar forward)
        if 60 in direction_results:
            if direction_results[60]['correct']:
                self.prediction_accuracy['direction_correct'] += 1
            else:
                self.prediction_accuracy['direction_incorrect'] += 1
        
        return result
    
    def _simulate_exit(self, rows, entry_idx, direction, entry_price, sl_pct, tp_pct, max_bars=300):
        """Simula un trade con SL/TP específicos y retorna resultado."""
        total = len(rows)
        sl_price = entry_price * (1 - sl_pct) if direction == 'long' else entry_price * (1 + sl_pct)
        tp_price = entry_price * (1 + tp_pct) if direction == 'long' else entry_price * (1 - tp_pct)
        
        for j in range(1, min(max_bars, total - entry_idx)):
            bar = rows.iloc[entry_idx + j]
            high = bar['high']
            low = bar['low']
            
            if direction == 'long':
                if low <= sl_price:
                    pnl = (sl_price - entry_price) / entry_price - COMMISSION_PCT * 2
                    return {'exit': 'SL', 'bars': j, 'pnl_pct': pnl * 100}
                if high >= tp_price:
                    pnl = (tp_price - entry_price) / entry_price - COMMISSION_PCT * 2
                    return {'exit': 'TP', 'bars': j, 'pnl_pct': pnl * 100}
            else:
                if high >= sl_price:
                    pnl = (entry_price - sl_price) / entry_price - COMMISSION_PCT * 2
                    return {'exit': 'SL', 'bars': j, 'pnl_pct': pnl * 100}
                if low <= tp_price:
                    pnl = (entry_price - tp_price) / entry_price - COMMISSION_PCT * 2
                    return {'exit': 'TP', 'bars': j, 'pnl_pct': pnl * 100}
        
        # Time exit
        last_close = rows.iloc[min(entry_idx + max_bars, total - 1)]['close']
        if direction == 'long':
            pnl = (last_close - entry_price) / entry_price - COMMISSION_PCT * 2
        else:
            pnl = (entry_price - last_close) / entry_price - COMMISSION_PCT * 2
        return {'exit': 'TIME', 'bars': max_bars, 'pnl_pct': pnl * 100}

    def compile_report(self):
        """
        📊 Compila el reporte forense final con todas las métricas agregadas.
        """
        if not self.signals:
            return "❌ No signals found to analyze."
        
        n = len(self.signals)
        
        # === 1. DIRECTION ACCURACY BY WINDOW ===
        print("\n" + "═" * 80)
        print("🔬 AUDITORÍA FORENSE DE PRECISIÓN — RESULTADOS")
        print("═" * 80)
        
        print(f"\n📊 Total señales analizadas: {n}")
        
        print("\n┌─────────────────────────────────────────────────────────────┐")
        print("│ 📈 DIRECTION ACCURACY POR VENTANA TEMPORAL                 │")
        print("├──────────┬──────────┬──────────┬──────────┬────────────────┤")
        print("│ Ventana  │ Correcta │  Total   │ Accuracy │  Net Return %  │")
        print("├──────────┼──────────┼──────────┼──────────┼────────────────┤")
        
        for w in FORWARD_WINDOWS:
            data = self.prediction_accuracy['accuracy_by_window'][w]
            if data['total'] > 0:
                acc = data['correct'] / data['total'] * 100
                # Average net return
                returns = [s['direction_accuracy'].get(w, {}).get('net_return_pct', 0) 
                          for s in self.signals if w in s.get('direction_accuracy', {})]
                avg_ret = np.mean(returns) if returns else 0
                emoji = "✅" if acc > 55 else "⚠️" if acc > 50 else "❌"
                print(f"│ {w:>5} bar │ {data['correct']:>6}   │ {data['total']:>6}   │ {acc:>5.1f}% {emoji}│ {avg_ret:>+10.4f}%    │")
        
        print("└──────────┴──────────┴──────────┴──────────┴────────────────┘")
        
        # === 2. LOSS CAUSE BREAKDOWN ===
        loss_causes = {}
        for s in self.signals:
            cause = s['loss_cause']
            if cause not in loss_causes:
                loss_causes[cause] = 0
            loss_causes[cause] += 1
        
        print("\n┌─────────────────────────────────────────────────────────────┐")
        print("│ 🔍 CAUSA DE PÉRDIDA / RESULTADO POR SEÑAL                  │")
        print("├────────────────────────┬──────────┬─────────────────────────┤")
        print("│ Causa                  │  Count   │  % del Total            │")
        print("├────────────────────────┼──────────┼─────────────────────────┤")
        
        cause_emojis = {
            'TP_ALCANZADO': '🎯',
            'SL_PREMATURO': '💀',
            'COMISION_BLEED': '💸',
            'DIRECCION_INCORRECTA': '❌',
            'TP_INALCANZABLE': '🏔️',
            'ESTANCAMIENTO': '😴',
            'NONE': '⬜',
        }
        
        for cause, count in sorted(loss_causes.items(), key=lambda x: -x[1]):
            pct = count / n * 100
            emoji = cause_emojis.get(cause, '❓')
            print(f"│ {emoji} {cause:<20} │ {count:>6}   │ {pct:>8.1f}%                │")
        
        print("└────────────────────────┴──────────┴─────────────────────────┘")
        
        # === 3. MFE vs FEES ANALYSIS ===
        mfes = [s['mfe_pct'] for s in self.signals]
        maes = [s['mae_pct'] for s in self.signals]
        fee_pct = COMMISSION_PCT * 2 * 100
        
        fee_destroys = sum(1 for s in self.signals if s['fee_destroys_edge'])
        
        print("\n┌─────────────────────────────────────────────────────────────┐")
        print("│ 💰 MFE vs COMISIONES — ¿Las fees destruyen el edge?        │")
        print("├──────────────────────────────────────────────────────────────┤")
        print(f"│ Round-trip fee:         {fee_pct:.4f}%                       │")
        print(f"│ MFE promedio:           {np.mean(mfes):>+.4f}%                       │")
        print(f"│ MFE mediana:            {np.median(mfes):>+.4f}%                       │")
        print(f"│ MAE promedio:           {np.mean(maes):>+.4f}%                       │")
        print(f"│ Señales donde fee>MFE:  {fee_destroys}/{n} ({fee_destroys/n*100:.1f}%)        │")
        print(f"│ MFE/Fee ratio medio:    {np.mean(mfes)/fee_pct:.2f}x                       │")
        print("└──────────────────────────────────────────────────────────────┘")
        
        # === 4. TEMPORAL DECAY ANALYSIS ===
        mfe_bars = [s['mfe_bar'] for s in self.signals]
        decay_bars = [s['prediction_decay_bar'] for s in self.signals if s['prediction_decay_bar'] is not None]
        optimal_bars = [s['optimal_exit_bar'] for s in self.signals]
        sl_bars = [s['sl_hit_bar'] for s in self.signals if s['sl_hit_bar'] is not None]
        tp_bars = [s['tp_hit_bar'] for s in self.signals if s['tp_hit_bar'] is not None]
        consec_profit = [s['max_consecutive_profit_bars'] for s in self.signals]
        
        print("\n┌─────────────────────────────────────────────────────────────┐")
        print("│ ⏱️ ANÁLISIS TEMPORAL — ¿Cuánto dura nuestro edge?          │")
        print("├──────────────────────────────────────────────────────────────┤")
        print(f"│ MFE alcanzado en barra: mediana={np.median(mfe_bars):.0f}, media={np.mean(mfe_bars):.0f}        │")
        if decay_bars:
            print(f"│ Prediction decay en:    mediana={np.median(decay_bars):.0f}, media={np.mean(decay_bars):.0f}        │")
        print(f"│ Salida óptima en barra: mediana={np.median(optimal_bars):.0f}, media={np.mean(optimal_bars):.0f}        │")
        if sl_bars:
            print(f"│ SL tocado en barra:     mediana={np.median(sl_bars):.0f}, media={np.mean(sl_bars):.0f}        │")
        if tp_bars:
            print(f"│ TP tocado en barra:     mediana={np.median(tp_bars):.0f}, media={np.mean(tp_bars):.0f}        │")
        print(f"│ Max barras en profit:   mediana={np.median(consec_profit):.0f}, media={np.mean(consec_profit):.0f}        │")
        print("└──────────────────────────────────────────────────────────────┘")
        
        # === 5. SL PREMATURO ANALYSIS ===
        sl_prematuro_trades = [s for s in self.signals if s['loss_cause'] == 'SL_PREMATURO']
        if sl_prematuro_trades:
            print("\n┌─────────────────────────────────────────────────────────────┐")
            print("│ 💀 SL PREMATURO — Trades que tuvieron profit pero SL ganó   │")
            print("├──────────────────────────────────────────────────────────────┤")
            avg_mfe_sl = np.mean([t['mfe_pct'] for t in sl_prematuro_trades])
            avg_optimal_pnl = np.mean([t['optimal_exit_pnl_pct'] for t in sl_prematuro_trades])
            avg_sl_bar = np.mean([t['sl_hit_bar'] for t in sl_prematuro_trades if t['sl_hit_bar']])
            avg_mfe_bar_sl = np.mean([t['mfe_bar'] for t in sl_prematuro_trades])
            
            print(f"│ Cantidad:               {len(sl_prematuro_trades)} trades ({len(sl_prematuro_trades)/n*100:.1f}% del total)       │")
            print(f"│ MFE promedio alcanzado:  {avg_mfe_sl:.4f}%                       │")
            print(f"│ PnL óptima perdida:      {avg_optimal_pnl:.4f}%                       │")
            print(f"│ SL tocado en barra avg:  {avg_sl_bar:.0f}                              │")
            print(f"│ MFE alcanzado en barra:  {avg_mfe_bar_sl:.0f} (MFE < SL bar = SL prematuro)│")
            print("│                                                              │")
            print("│ 💡 DIAGNÓSTICO: El SL se toca DESPUÉS del MFE. El trade     │")
            print("│    alcanzó profit pero luego revirtió y el SL lo liquidó.     │")
            print("│    SOLUCIÓN: Implementar trailing stop más agresivo que       │")
            print("│    asegure breakeven cuando MFE > 1x fee.                    │")
            print("└──────────────────────────────────────────────────────────────┘")
        
        # === 6. COMMISSION BLEED ANALYSIS ===
        fee_bleed_trades = [s for s in self.signals if s['loss_cause'] == 'COMISION_BLEED']
        if fee_bleed_trades:
            print("\n┌─────────────────────────────────────────────────────────────┐")
            print("│ 💸 COMMISSION BLEED — Comisiones > Edge predictivo          │")
            print("├──────────────────────────────────────────────────────────────┤")
            avg_mfe_cb = np.mean([t['mfe_pct'] for t in fee_bleed_trades])
            avg_mae_cb = np.mean([t['mae_pct'] for t in fee_bleed_trades])
            
            print(f"│ Cantidad:               {len(fee_bleed_trades)} trades ({len(fee_bleed_trades)/n*100:.1f}% del total)       │")
            print(f"│ MFE promedio:            {avg_mfe_cb:.4f}% (< fee {fee_pct:.4f}%)      │")
            print(f"│ MAE promedio:            {avg_mae_cb:.4f}%                       │")
            print("│                                                              │")
            print("│ 💡 DIAGNÓSTICO: El precio se mueve tan poco que las         │")
            print("│    comisiones devoran cualquier micro-ganancia.               │")
            print("│    SOLUCIÓN: Filtrar señales donde ATR < 3x fees, o          │")
            print("│    usar órdenes LIMIT maker (0.02% fee vs 0.0375%).          │")
            print("└──────────────────────────────────────────────────────────────┘")
        
        # === 7. WHAT-IF BEST SL/TP CONFIGURATION ===
        print("\n┌─────────────────────────────────────────────────────────────┐")
        print("│ 🎯 WHAT-IF: ¿Cuál sería el SL/TP óptimo?                  │")
        print("├──────────────────────────────────────────────────────────────┤")
        
        # Aggregate what-if results
        what_if_agg = {}
        for s in self.signals:
            if s.get('what_if_best') and s['what_if_best'][1]:
                key = s['what_if_best'][0]
                pnl = s['what_if_best'][1].get('pnl_pct', 0)
                if key not in what_if_agg:
                    what_if_agg[key] = []
                what_if_agg[key].append(pnl)
        
        if what_if_agg:
            best_configs = sorted(what_if_agg.items(), key=lambda x: np.mean(x[1]), reverse=True)[:5]
            
            for config, pnls in best_configs:
                avg_pnl = np.mean(pnls)
                win_rate = sum(1 for p in pnls if p > 0) / len(pnls) * 100
                print(f"│ {config:<20}: avg_pnl={avg_pnl:>+.3f}% WR={win_rate:.0f}% (n={len(pnls)})  │")
        
        print("└──────────────────────────────────────────────────────────────┘")
        
        # === 8. OVERALL DIAGNOSIS ===
        print("\n" + "═" * 80)
        print("🏥 DIAGNÓSTICO GLOBAL DE PRECISIÓN")
        print("═" * 80)
        
        if self.prediction_accuracy['direction_correct'] + self.prediction_accuracy['direction_incorrect'] > 0:
            overall_acc = self.prediction_accuracy['direction_correct'] / (
                self.prediction_accuracy['direction_correct'] + self.prediction_accuracy['direction_incorrect']
            ) * 100
            print(f"\n  📊 Direction Accuracy (60-bar): {overall_acc:.1f}%")
        
        # Determine primary loss cause
        primary_cause = max(loss_causes.items(), key=lambda x: x[1]) if loss_causes else ('UNKNOWN', 0)
        print(f"  🔍 Primary Loss Cause: {primary_cause[0]} ({primary_cause[1]}/{n} = {primary_cause[1]/n*100:.0f}%)")
        
        if np.mean(mfes) < fee_pct:
            print(f"  ⚠️ CRITICAL: Average MFE ({np.mean(mfes):.4f}%) < Round-trip Fee ({fee_pct:.4f}%)")
            print(f"     → El edge predictivo es INSUFICIENTE para cubrir las comisiones")
        
        if decay_bars:
            avg_decay = np.mean(decay_bars)
            print(f"  ⏱️ Prediction Edge Duration: ~{avg_decay:.0f} barras ({avg_decay:.0f} minutos en M1)")
            if avg_decay < 30:
                print(f"     → Edge CORTO: considerar scalping ultra-rápido con salida en <{avg_decay:.0f} barras")
        
        avg_mfe_bar = np.mean(mfe_bars)
        if sl_bars:
            avg_sl_bar_v = np.mean(sl_bars)
            if avg_sl_bar_v < avg_mfe_bar:
                print(f"  💀 SL se toca (barra {avg_sl_bar_v:.0f}) ANTES del MFE (barra {avg_mfe_bar:.0f})")
                print(f"     → El SL es DEMASIADO AJUSTADO. Debería ser >{avg_mfe_bar:.0f} barras de distancia")
        
        return self.signals


# ══════════════════════════════════════════════════════════════════════════════
# MAIN: Ejecutar la auditoría
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("🔬 FORENSIC STRATEGY ACCURACY AUDIT v1.0")
    print("=" * 60)
    print(f"  Capital: ${AUDIT_INITIAL_CAPITAL}")
    print(f"  Leverage: {AUDIT_LEVERAGE}x")
    print(f"  Symbols: {AUDIT_SYMBOLS}")
    print(f"  Horizon: {AUDIT_HORIZON}D ({AUDIT_DAYS} días de datos)")
    print(f"  Fee: {COMMISSION_PCT*100:.4f}% per side")
    print("=" * 60)
    
    auditor = ForensicAccuracyAuditor()
    profile = HORIZON_PROFILES.get(AUDIT_HORIZON, HORIZON_PROFILES[1])
    
    for symbol_raw in AUDIT_SYMBOLS:
        symbol = symbol_raw.replace('USDT', '/USDT')
        print(f"\n{'─'*60}")
        print(f"📊 Procesando {symbol}...")
        
        df = fetch_data(symbol, AUDIT_DAYS)
        if df is None or len(df) < 500:
            print(f"  ⚠️ Datos insuficientes para {symbol}, skipping.")
            continue
        
        # Compute indicators (same as production)
        df = compute_indicators(df, horizon_days=AUDIT_HORIZON)
        df = df.dropna()
        
        # Calibrate SL/TP (same as production)
        cal_lookahead = profile.get('ml_lookahead', 60)
        calibrated_sl, calibrated_tp = calibrate_sl_tp(
            df['close'].values, cal_lookahead,
            sl_cap=profile['sl_cap'], tp_cap=profile['tp_cap']
        )
        print(f"  📐 Calibrated SL={calibrated_sl*100:.3f}% TP={calibrated_tp*100:.3f}%")
        
        # Init Sophia cluster engine (same as production)
        sophia = SophiaClusterEngine(n_clusters=4, refit_interval=profile['sophia_refit'])
        
        # Init XGBoost (same as production)
        xgb_engine = WalkForwardXGBoost(
            min_train_size=5000,
            horizon_days=AUDIT_HORIZON,
            horizon_profile=profile
        )
        
        # RSI window
        rsi_window = []
        
        rows = df.reset_index()
        total = len(rows)
        warmup = 200
        signal_count = 0
        
        strategies_to_audit = STRATEGY_SPECIALIZATION_MAP.get(AUDIT_HORIZON, ['Technical', 'ML_XGBoost'])
        
        print(f"  📊 {total} barras disponibles, warmup={warmup}")
        print(f"  🧠 Estrategias a auditar: {strategies_to_audit}")
        
        for strategy_name in strategies_to_audit:
            print(f"\n  🔍 Auditando {strategy_name} en {symbol}...")
            
            # Reset XGBoost for each strategy
            if strategy_name == 'ML_XGBoost':
                xgb_engine = WalkForwardXGBoost(
                    min_train_size=2000,
                    horizon_days=AUDIT_HORIZON,
                    horizon_profile=profile
                )
            
            sophia_local = SophiaClusterEngine(n_clusters=4, refit_interval=profile['sophia_refit'])
            rsi_window_local = []
            
            for i in range(warmup, total - 60):  # Leave 60 bars for forward analysis
                row = rows.iloc[i]
                prev_row = rows.iloc[i-1]
                
                # Update RSI window
                rsi_window_local.append(row['rsi'])
                if len(rsi_window_local) > 200:
                    rsi_window_local.pop(0)
                
                # Sophia update
                sophia_local.update(row)
                
                # Dynamic RSI
                if len(rsi_window_local) >= 50:
                    rsi_buy = max(20, min(np.percentile(rsi_window_local, 15), 40))
                    rsi_sell = min(80, max(np.percentile(rsi_window_local, 85), 60))
                else:
                    rsi_buy, rsi_sell = 30, 70
                
                params = {
                    'rsi_buy': rsi_buy, 'rsi_sell': rsi_sell,
                    'calibrated_sl': calibrated_sl, 'calibrated_tp': calibrated_tp
                }
                
                # Sophia safety check
                is_safe, regime, conf = sophia_local.is_safe_to_trade(row)
                
                # Get signal based on strategy
                direction = None
                sl_pct = calibrated_sl
                tp_pct = calibrated_tp
                
                if strategy_name == 'Technical':
                    direction, sl_pct, tp_pct = signal_technical(
                        row, prev_row, params, regime=regime, horizon_profile=profile
                    )
                
                elif strategy_name == 'ML_XGBoost':
                    # Base signal from Technical
                    base_dir, base_sl, base_tp = signal_technical(
                        row, prev_row, params, regime=regime, horizon_profile=profile
                    )
                    
                    if base_dir:
                        # Train if needed
                        if not xgb_engine.is_trained or xgb_engine.should_retrain():
                            train_start = max(0, i - xgb_engine.min_train_size - 100)
                            train_df = df.iloc[train_start:i]
                            xgb_engine.train(train_df, horizon_profile=profile)
                        
                        if xgb_engine.is_trained:
                            pred_start = max(0, i - 200)
                            bars_window = df.iloc[pred_start:i+1]
                            ml_dir, _, _, ml_info = xgb_engine.predict(
                                bars_window, base_dir=base_dir,
                                base_sl=base_sl, base_tp=base_tp,
                                horizon_profile=profile
                            )
                            if ml_dir == base_dir:
                                direction = base_dir
                                sl_pct = base_sl
                                tp_pct = base_tp
                        else:
                            direction = base_dir
                            sl_pct = base_sl
                            tp_pct = base_tp
                
                # If signal generated, analyze it
                if direction is not None:
                    entry_price = row['close']
                    size_pct = 0.30  # Same as production
                    size_usd = AUDIT_INITIAL_CAPITAL * AUDIT_LEVERAGE * size_pct
                    
                    result = auditor.analyze_signal_accuracy(
                        df, i, direction, sl_pct, tp_pct,
                        entry_price, size_usd, strategy_name
                    )
                    
                    if result:
                        signal_count += 1
                        
                        # Print progress every 50 signals
                        if signal_count % 50 == 0:
                            print(f"     📊 {signal_count} señales analizadas...")
            
            print(f"  ✅ {signal_count} señales totales encontradas para {strategy_name}")
    
    # === COMPILE GLOBAL REPORT ===
    print("\n\n" + "🔬" * 40)
    auditor.compile_report()
    
    # Save raw results
    output_path = os.path.join('logs', 'forensic_accuracy_audit.json')
    os.makedirs('logs', exist_ok=True)
    
    # Save summary (not full bar-by-bar data)
    summary = []
    for s in auditor.signals:
        entry = {k: v for k, v in s.items() if k not in ['bar_by_bar_pnl']}
        # Convert numpy types
        for k, v in entry.items():
            if isinstance(v, (np.integer, np.int64)):
                entry[k] = int(v)
            elif isinstance(v, (np.floating, np.float64)):
                entry[k] = float(v)
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, (np.integer, np.int64)):
                        v[k2] = int(v2)
                    elif isinstance(v2, (np.floating, np.float64)):
                        v[k2] = float(v2)
        summary.append(entry)
    
    try:
        with open(output_path, 'w') as f:
            json.dump({
                'audit_date': datetime.now().isoformat(),
                'symbols': AUDIT_SYMBOLS,
                'horizon': AUDIT_HORIZON,
                'total_signals': len(summary),
                'prediction_accuracy': {
                    k: v for k, v in auditor.prediction_accuracy.items()
                    if k != 'accuracy_by_window'
                },
                # Simplified accuracy by window
                'accuracy_by_window': {
                    str(w): {
                        'accuracy_pct': round(d['correct'] / d['total'] * 100, 1) if d['total'] > 0 else 0,
                        'n_samples': d['total']
                    }
                    for w, d in auditor.prediction_accuracy['accuracy_by_window'].items()
                    if d['total'] > 0
                }
            }, f, indent=2, default=str)
        print(f"\n💾 Resultados guardados en: {output_path}")
    except Exception as e:
        print(f"⚠️ Error guardando resultados brutos: {e}")
        
    # === GENERAR PREDICTION_METRICS.JSON PARA SISTEMA LIVE ===
    try:
        prediction_metrics = {}
        for s in summary:
            strat = s['strategy']
            if strat not in prediction_metrics:
                prediction_metrics[strat] = {'hits': 0, 'total': 0, 'decays': []}
            
            # Un hit es cuando mfe > fee_buffer (trade viable)
            if s['mfe_pct'] > s['round_trip_fee_pct']:
                prediction_metrics[strat]['hits'] += 1
            prediction_metrics[strat]['total'] += 1
            if s.get('prediction_decay_bar'):
                prediction_metrics[strat]['decays'].append(s['prediction_decay_bar'])
                
        final_metrics = {}
        for strat, data in prediction_metrics.items():
            if data['total'] < 10: continue
            acc = data['hits'] / data['total']
            
            # Confidence factor: escala linear donde 50% = 0.5, 80% = 1.0, >90% = 1.2
            # Interpolamos para que al 60% la agresividad sea baja, 80% alta.
            c_factor = max(0.5, min(1.2, (acc - 0.5) * 2.5 + 0.5)) if acc > 0.5 else 0.5
            
            avg_decay = np.median(data['decays']) if data['decays'] else 60
            
            final_metrics[strat] = {
                "accuracy_pct": round(acc * 100, 2),
                "confidence_factor": round(c_factor, 2),
                "optimal_ttl_bars": int(avg_decay)
            }
        
        metrics_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'prediction_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        print(f"✅ Prediction Metrics guardadas en: {metrics_path}")
        
        # === GENERAR PREDICTION_ACCURACY_LOG.md ===
        md_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'PREDICTION_ACCURACY_LOG.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# 🎯 PREDICTION ACCURACY LOG\n\n")
            f.write(f"**Generado:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## 1. Precisión por Estrategia\n\n")
            f.write("| Estrategia | Precisión (%) | Confidence Factor | Tiempo de Vida Óptimo (barras) |\n")
            f.write("|------------|---------------|-------------------|--------------------------------|\n")
            for strat, mets in final_metrics.items():
                f.write(f"| {strat} | {mets['accuracy_pct']}% | {mets['confidence_factor']}x | {mets['optimal_ttl_bars']} barras |\n")
            
            f.write("\n## 2. Recomendación de Ejecución LIMIT Dinámica\n\n")
            f.write("El factor de confianza (Confidence Factor) modula la distancia a la que se colocan los Take Profits LIMIT.\n")
            for strat, mets in final_metrics.items():
                if mets['accuracy_pct'] > 75:
                    f.write(f"- **{strat}:** 🟢 **Alta Precisión**. LIMIT agresivo activado. Exposición aumentada.\n")
                elif mets['accuracy_pct'] > 60:
                    f.write(f"- **{strat}:** 🟡 **Precisión Media**. LIMIT conservador. Ajustes finos requeridos.\n")
                else:
                    f.write(f"- **{strat}:** 🔴 **Precisión Baja**. Se recomienda pausar esta estrategia.\n")
                    
        print(f"📄 Reporte generado: {md_path}")
            
    except Exception as e:
        print(f"⚠️ Error generando métricas predictivas en vivo: {e}")


if __name__ == '__main__':
    main()
