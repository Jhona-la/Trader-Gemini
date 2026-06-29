import json
import pandas as pd
import numpy as np
from datetime import datetime
import sys

def analyze():
    try:
        with open('backtest_results.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Wait for backtest to finish (file not found).")
        return

    trades = data.get('detailed_trades', [])
    if not trades:
        print("No trades found.")
        return

    df = pd.DataFrame(trades)
    
    # Parse timestamps
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    
    # Expand metadata
    df['atr'] = df['metadata'].apply(lambda x: x.get('atr', 0) if x else 0)
    df['regime'] = df['metadata'].apply(lambda x: x.get('regime', 'UNKNOWN') if x else 'UNKNOWN')
    df['entry_price'] = df['entry']
    df['volatility_pct'] = (df['atr'] / df['entry_price']) * 100
    
    # Filter
    losses = df[df['pnl_usd'] < 0].copy()
    winners = df[df['pnl_usd'] > 0].copy()
    
    print("# 📉 ANÁLISIS DE 33 OPERACIONES PERDEDORAS")
    print(f"Total Trades: {len(df)} | Winners: {len(winners)} | Losers: {len(losses)}")
    
    if len(losses) == 0:
        print("No losses to analyze!")
        return

    # 1. Patrones Temporales (Horarios)
    losses['hour'] = losses['entry_time'].dt.hour
    hourly_losses = losses['hour'].value_counts().sort_index()
    
    print("\n## 1. Patrones Temporales")
    print("| Hora (UTC) | # Losses | Session |")
    print("|---|---|---|")
    for hour, count in hourly_losses.items():
        session = "Asian"
        if 7 <= hour < 15: session = "London"
        if 13 <= hour < 21: session = "NY"
        if 7 <= hour < 8: session = "Pre-London"
        if 13 <= hour < 15: session = "London/NY Overlap"
        
        print(f"| {hour:02d}:00 | {count} | {session} |")

    # 2. Volatilidad (ATR)
    avg_vol_loss = losses['volatility_pct'].mean()
    avg_vol_win = winners['volatility_pct'].mean()
    
    print("\n## 2. Análisis de Volatilidad (ATR)")
    print(f"- **Avg Volatility (Losers):** {avg_vol_loss:.4f}%")
    print(f"- **Avg Volatility (Winners):** {avg_vol_win:.4f}%")
    if avg_vol_loss > avg_vol_win:
        print("> ⚠️ Las pérdidas ocurren en mayor volatilidad relative.")
    else:
        print("> ℹ️ La volatilidad no parece ser el factor discriminante principal.")

    # 3. Regime Analysis
    print("\n## 3. Relación con Market Regime")
    regime_counts = losses['regime'].value_counts()
    print("| Regime | # Losses | % of Total Losses |")
    print("|---|---|---|")
    for regime, count in regime_counts.items():
        pct = (count / len(losses)) * 100
        print(f"| {regime} | {count} | {pct:.1f}% |")

    # 4. SL Effectiveness
    # Check if losses hit full SL (-1.5%) or trailing
    # Assuming SL is -1.5% fixed
    # We check how many are near -1.5%
    sl_hits = losses[losses['pnl_pct'] <= -1.45] # Approx
    trailing_stops = losses[losses['pnl_pct'] > -1.45]
    
    print("\n## 4. Efectividad de Stops")
    print(f"- **Full SL Hits (~1.5% loss):** {len(sl_hits)} ({len(sl_hits)/len(losses)*100:.1f}%)")
    print(f"- **Trailing Stops / Early Exit:** {len(trailing_stops)} ({len(trailing_stops)/len(losses)*100:.1f}%)")
    
    if len(sl_hits) > len(trailing_stops):
        print("> 🚨 La mayoría de pérdidas tocan el Stop Loss completo. El Trailing Stop no está salvando suficientes trades.")
    
    # 5. Recommendation
    print("\n## 5. Recomendaciones para Risk Manager")
    print("Based on the data:")
    if avg_vol_loss > 0.5:
         print("- [ ] **Reducir Leverage en alta volatilidad:** Detectamos que las pérdidas ocurren con ATR alto.")
    if 13 <= hourly_losses.idxmax() <= 16:
         print("- [ ] **Filtro de Horario:** Considerar pausa durante apertura NY (13:00-15:00 UTC) si hay picos de pérdidas.")
    if len(sl_hits) > 20:
         print("- [ ] **Ajustar SL Dinámico:** El SL fijo de 1.5% es golpeado frecuentemente. Considerar reducirlo si la volatilidad es baja, o usar ATR-based SL.")

if __name__ == "__main__":
    analyze()
