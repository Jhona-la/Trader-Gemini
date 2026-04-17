"""
🔬 DIAGNÓSTICO DE RENTABILIDAD - Trader Gemini
================================================
QUÉ: Análisis profundo de por qué las estrategias pierden dinero en 7D-30D
POR QUÉ: El Sharpe solo es positivo en 1D (Orchestrator: 1.48)
PARA QUÉ: Identificar la causa raíz (stops, overtrading, régimen, modelo)
CÓMO: Backtest aislado por estrategia con métricas detalladas
CUÁNDO: Antes de optimizar parámetros
DÓNDE: scripts/profitability_diagnosis.py
QUIÉN: Quant Developer + Risk Manager
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import ccxt
import json
from datetime import datetime

# ── Import backtest components ──
from scripts.run_multi_horizon_backtest import (
    fetch_data, compute_indicators, 
    SophiaClusterEngine, WalkForwardXGBoost,
    INITIAL_CAPITAL
)

DIAGNOSIS_HORIZON = 7  # Focus on 7D as it's the first failing horizon
SYMBOL = 'BTC/USDT'


def analyze_technical_strategy(df, capital=INITIAL_CAPITAL):
    """Analizar la estrategia técnica trade por trade."""
    trades = []
    position = None
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        close = row['close']
        rsi = row['rsi']
        atr_pct = row['atr_pct']
        
        if position is None:
            # Entry logic (simplified from backtest)
            ema20 = row['ema20']
            ema50 = row['ema50']
            bb_upper = row['bb_upper']
            bb_lower = row['bb_lower']
            vol_ratio = row['vol_ratio']
            
            signal = None
            if rsi < 30 and close <= bb_lower and vol_ratio > 1.0:
                signal = 'long'
            elif rsi > 70 and close >= bb_upper and vol_ratio > 1.0:
                signal = 'short'
            
            if signal:
                sl_pct = min(max(atr_pct * 1.8, 0.008), 0.025)
                tp_pct = min(max(atr_pct * 3.5, 0.015), 0.06)
                position = {
                    'entry_price': close,
                    'entry_bar': i,
                    'side': signal,
                    'sl_pct': sl_pct,
                    'tp_pct': tp_pct,
                    'entry_rsi': rsi,
                    'entry_atr_pct': atr_pct,
                    'entry_vol_ratio': vol_ratio,
                }
        else:
            # Exit logic
            entry = position['entry_price']
            if position['side'] == 'long':
                pnl_pct = (close - entry) / entry
                hit_tp = pnl_pct >= position['tp_pct']
                hit_sl = pnl_pct <= -position['sl_pct']
            else:
                pnl_pct = (entry - close) / entry
                hit_tp = pnl_pct >= position['tp_pct']
                hit_sl = pnl_pct <= -position['sl_pct']
            
            if hit_tp or hit_sl:
                holding_bars = i - position['entry_bar']
                trades.append({
                    'entry_bar': position['entry_bar'],
                    'exit_bar': i,
                    'side': position['side'],
                    'pnl_pct': pnl_pct * 100,
                    'holding_bars': holding_bars,
                    'holding_minutes': holding_bars,  # 1-min bars
                    'exit_reason': 'TP' if hit_tp else 'SL',
                    'entry_rsi': position['entry_rsi'],
                    'entry_atr_pct': position['entry_atr_pct'],
                    'sl_pct': position['sl_pct'] * 100,
                    'tp_pct': position['tp_pct'] * 100,
                })
                position = None
    
    return trades


def print_diagnosis(trades, strategy_name):
    """Generar reporte de diagnóstico detallado."""
    if not trades:
        print(f"  ⚠️  {strategy_name}: Sin trades generados")
        return
    
    df_trades = pd.DataFrame(trades)
    
    total = len(df_trades)
    wins = len(df_trades[df_trades['pnl_pct'] > 0])
    losses = len(df_trades[df_trades['pnl_pct'] <= 0])
    win_rate = wins / total * 100
    
    avg_win = df_trades[df_trades['pnl_pct'] > 0]['pnl_pct'].mean() if wins > 0 else 0
    avg_loss = df_trades[df_trades['pnl_pct'] <= 0]['pnl_pct'].mean() if losses > 0 else 0
    
    avg_holding = df_trades['holding_minutes'].mean()
    median_holding = df_trades['holding_minutes'].median()
    
    total_pnl = df_trades['pnl_pct'].sum()
    
    # Distribución de exits
    tp_exits = len(df_trades[df_trades['exit_reason'] == 'TP'])
    sl_exits = len(df_trades[df_trades['exit_reason'] == 'SL'])
    
    # Distribución de trades por dirección
    longs = len(df_trades[df_trades['side'] == 'long'])
    shorts = len(df_trades[df_trades['side'] == 'short'])
    
    # SL/TP ratios
    avg_sl = df_trades['sl_pct'].mean()
    avg_tp = df_trades['tp_pct'].mean()
    rr_ratio = avg_tp / avg_sl if avg_sl > 0 else 0
    
    # Trades por hora
    if len(df_trades) > 1:
        total_bars = df_trades['exit_bar'].max() - df_trades['entry_bar'].min()
        trades_per_hour = total / (total_bars / 60) if total_bars > 0 else 0
    else:
        trades_per_hour = 0
    
    print(f"\n  {'='*55}")
    print(f"  📊 {strategy_name} — DIAGNÓSTICO DETALLADO")
    print(f"  {'='*55}")
    print(f"  📈 Total Trades:     {total}")
    print(f"  ✅ Wins / ❌ Losses: {wins} / {losses}")
    print(f"  🎯 Win Rate:         {win_rate:.1f}%")
    print(f"  💰 Total PnL:        {total_pnl:+.3f}%")
    print(f"  📊 Avg Win:          {avg_win:+.4f}%")
    print(f"  📉 Avg Loss:         {avg_loss:+.4f}%")
    print(f"  ⏱️  Avg Holding:      {avg_holding:.0f} min (median: {median_holding:.0f} min)")
    print(f"  📊 Trades/Hora:      {trades_per_hour:.2f}")
    print(f"  🟢 Longs / 🔴 Shorts: {longs} / {shorts}")
    print(f"  🎯 TP Exits / 🛑 SL: {tp_exits} / {sl_exits}")
    print(f"  📐 Avg SL / TP:      {avg_sl:.3f}% / {avg_tp:.3f}%")
    print(f"  📊 R:R Ratio:        {rr_ratio:.2f}")
    
    # DIAGNÓSTICO
    print(f"\n  🔬 CAUSA RAÍZ:")
    issues = []
    
    if win_rate < 40:
        issues.append(f"  ❌ Win Rate muy baja ({win_rate:.1f}%) → Señales entran en momentos incorrectos")
    
    if rr_ratio < 1.5:
        issues.append(f"  ❌ R:R ratio insuficiente ({rr_ratio:.2f}) → TP muy cerca o SL muy lejos")
    
    if trades_per_hour > 2:
        issues.append(f"  ❌ Overtrading ({trades_per_hour:.1f} trades/hora) → Filtros de entrada débiles")
    
    if sl_exits > tp_exits * 1.5:
        issues.append(f"  ❌ SL dominante ({sl_exits} SL vs {tp_exits} TP) → Stops demasiado ajustados")
    
    if avg_holding < 30:
        issues.append(f"  ❌ Holding demasiado corto ({avg_holding:.0f} min) → Scalping en timeframe inadecuado")
    
    if abs(avg_loss) > avg_win * 1.5:
        issues.append(f"  ❌ Pérdida promedio >> Ganancia promedio → Asimetría negativa")
    
    if not issues:
        issues.append("  ✅ No se detectaron problemas claros")
    
    for issue in issues:
        print(issue)


def main():
    print("=" * 60)
    print(f"🔬 DIAGNÓSTICO DE RENTABILIDAD - {SYMBOL} {DIAGNOSIS_HORIZON}D")
    print("=" * 60)
    
    # Fetch data
    print(f"\n📡 Descargando datos {DIAGNOSIS_HORIZON}D para {SYMBOL}...")
    df = fetch_data(SYMBOL, DIAGNOSIS_HORIZON + 2)
    if df is None or len(df) < 500:
        print("❌ Datos insuficientes")
        return
    print(f"  ✅ {len(df)} velas descargadas")
    
    # Compute indicators
    print("  📊 Calculando indicadores...")
    df = compute_indicators(df)
    print(f"  ✅ Indicadores calculados ({len(df.columns)} columnas)")
    
    # === DIAGNÓSTICO TÉCNICO ===
    print("\n" + "="*60)
    print("🔹 ESTRATEGIA: TECHNICAL (Mean Reversion + Trend)")
    print("="*60)
    tech_trades = analyze_technical_strategy(df)
    print_diagnosis(tech_trades, "Technical")
    
    # === DIAGNÓSTICO SOPHIA ===
    print("\n" + "="*60)
    print("🔹 ESTRATEGIA: SOPHIA (Clustering)")
    print("="*60)
    sophia = SophiaClusterEngine(n_clusters=4, window_size=500, refit_interval=100)
    sophia_trades = []
    position = None
    
    # Warmup
    warmup_end = min(1000, len(df) // 4)
    if warmup_end > sophia.window_size:
        # Pushed through the update function
        for j in range(warmup_end):
            sophia.update(df.iloc[j])
    
    for i in range(warmup_end, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Incremental update
        sophia.update(row)
        
        if position is None:
            signal, sl_pct, tp_pct = sophia.generate_signal(row, prev)
            if signal:
                position = {
                    'entry_price': row['close'],
                    'entry_bar': i,
                    'side': signal,
                    'sl_pct': sl_pct,
                    'tp_pct': tp_pct,
                    'entry_rsi': row['rsi'],
                    'entry_atr_pct': row['atr_pct'],
                    'entry_vol_ratio': row['vol_ratio'],
                }
        else:
            entry = position['entry_price']
            close = row['close']
            if position['side'] == 'long':
                pnl_pct = (close - entry) / entry
            else:
                pnl_pct = (entry - close) / entry
            
            hit_tp = pnl_pct >= position['tp_pct']
            hit_sl = pnl_pct <= -position['sl_pct']
            
            if hit_tp or hit_sl:
                sophia_trades.append({
                    'entry_bar': position['entry_bar'],
                    'exit_bar': i,
                    'side': position['side'],
                    'pnl_pct': pnl_pct * 100,
                    'holding_bars': i - position['entry_bar'],
                    'holding_minutes': i - position['entry_bar'],
                    'exit_reason': 'TP' if hit_tp else 'SL',
                    'entry_rsi': position['entry_rsi'],
                    'entry_atr_pct': position['entry_atr_pct'],
                    'sl_pct': position['sl_pct'] * 100,
                    'tp_pct': position['tp_pct'] * 100,
                })
                position = None
    
    print_diagnosis(sophia_trades, "Sophia Clustering")
    
    # === DIAGNÓSTICO ML XGBoost ===
    print("\n" + "="*60)
    print("🔹 ESTRATEGIA: ML_XGBoost (Walk-Forward)")
    print("="*60)
    xgb = WalkForwardXGBoost(retrain_interval=1440, min_train_size=500, lookahead=30, threshold=0.58)
    ml_trades = []
    position = None
    
    warmup = 2000
    if warmup < len(df):
        xgb.train(df.iloc[:warmup])
    
    for i in range(warmup, len(df)):
        row = df.iloc[i]
        close = row['close']
        atr_pct = row['atr_pct']
        
        # Retrain check
        xgb.bars_since_train += 1
        if xgb.bars_since_train >= xgb.retrain_interval and xgb.is_trained:
            start = max(0, i - 5000)
            xgb.train(df.iloc[start:i])
        
        if position is None and xgb.is_trained:
            prediction_result = xgb.predict(df.iloc[max(0,i-100):i+1])
            if len(prediction_result) == 3:
                signal, confidence, _ = prediction_result
            else:
                signal, confidence = prediction_result
            
            if signal in ['long', 'short']:
                sl_pct = min(max(atr_pct * 2.0, 0.008), 0.025)
                tp_pct = min(max(atr_pct * 4.0, 0.015), 0.06)
                position = {
                    'entry_price': close,
                    'entry_bar': i,
                    'side': signal,
                    'sl_pct': sl_pct,
                    'tp_pct': tp_pct,
                    'entry_rsi': row['rsi'],
                    'entry_atr_pct': atr_pct,
                    'entry_vol_ratio': row['vol_ratio'],
                    'ml_confidence': confidence,
                }
        elif position is not None:
            entry = position['entry_price']
            if position['side'] == 'long':
                pnl_pct = (close - entry) / entry
            else:
                pnl_pct = (entry - close) / entry
            
            hit_tp = pnl_pct >= position['tp_pct']
            hit_sl = pnl_pct <= -position['sl_pct']
            
            if hit_tp or hit_sl:
                ml_trades.append({
                    'entry_bar': position['entry_bar'],
                    'exit_bar': i,
                    'side': position['side'],
                    'pnl_pct': pnl_pct * 100,
                    'holding_bars': i - position['entry_bar'],
                    'holding_minutes': i - position['entry_bar'],
                    'exit_reason': 'TP' if hit_tp else 'SL',
                    'entry_rsi': position['entry_rsi'],
                    'entry_atr_pct': position['entry_atr_pct'],
                    'sl_pct': position['sl_pct'] * 100,
                    'tp_pct': position['tp_pct'] * 100,
                })
                position = None
    
    print_diagnosis(ml_trades, "ML_XGBoost")
    
    # === RESUMEN COMPARATIVO ===
    print("\n" + "="*60)
    print("📊 RESUMEN COMPARATIVO")
    print("="*60)
    for name, t in [("Technical", tech_trades), ("Sophia", sophia_trades), ("ML_XGBoost", ml_trades)]:
        if t:
            pnl = sum(x['pnl_pct'] for x in t)
            wr = len([x for x in t if x['pnl_pct'] > 0]) / len(t) * 100
            avg_h = np.mean([x['holding_minutes'] for x in t])
            print(f"  {name:15s} | Trades: {len(t):4d} | PnL: {pnl:+7.3f}% | WR: {wr:5.1f}% | Avg Hold: {avg_h:6.0f}min")
        else:
            print(f"  {name:15s} | Sin trades")
    
    print(f"\n{'='*60}")
    print("✅ Diagnóstico completo. Revisar causas raíz arriba.")


if __name__ == "__main__":
    main()
