
import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Ensure root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

def audit_correlation():
    print("🔍 [CORRELATION AUDIT] Starting Multi-Symbol Analysis...")
    
    # 1. Load Data
    data_dir = Config.DATA_DIR # dashboard/data/futures
    pairs = Config.TRADING_PAIRS
    
    print(f"📂 Loading data from: {data_dir}")
    print(f"💎 Analyzing {len(pairs)} symbols...")
    
    close_prices = pd.DataFrame()
    
    for symbol in pairs:
        safe_sym = symbol.replace("/", "")
        file_path = os.path.join(data_dir, f"{safe_sym}_5m.parquet") # Use 5m for correlation
        
        if not os.path.exists(file_path):
            # Try 1m if 5m missing
            file_path = os.path.join(data_dir, f"{safe_sym}_1m.parquet")
            
        if os.path.exists(file_path):
            try:
                df = pd.read_parquet(file_path)
                print(f"DEBUG {symbol}: Cols={df.columns.tolist()} Index={type(df.index)}")
                
                # Ensure datetime index
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime']) # Force conversion
                    df.set_index('datetime', inplace=True)
                elif 'timestamp' in df.columns:
                     df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                     df.set_index('datetime', inplace=True)
                
                # Resample to 1h for robust correlation
                closes = df['close'].resample('1h').last()
                close_prices[symbol] = closes
            except Exception as e:
                print(f"⚠️ Error loading {symbol}: {e}")
        else:
            print(f"❌ Missing data for {symbol}")
            
    # 2. Compute Correlation
    if close_prices.empty:
        print("❌ No data available for audit.")
        return
        
    # Drop NaN
    close_prices.dropna(axis=0, how='any', inplace=True)
    
    print(f"📊 Data Points per Symbol: {len(close_prices)}")
    
    # Log Returns Correlation (More accurate than price correlation)
    log_rets = np.log(close_prices / close_prices.shift(1)).dropna()
    corr_matrix = log_rets.corr()
    
    # 3. Analyze Risks
    # Find pairs with > 0.90 correlation
    high_corr_pairs = []
    
    print("\n⚠️  HIGH CORRELATION ALERTS (> 0.90):")
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            sym_a = corr_matrix.columns[i]
            sym_b = corr_matrix.columns[j]
            corr = corr_matrix.iloc[i, j]
            
            if corr > 0.90:
                print(f"   🔴 {sym_a} <-> {sym_b}: {corr:.4f}")
                high_corr_pairs.append((sym_a, sym_b, corr))
                
    # Average Correlation
    avg_corr = corr_matrix.mean().mean()
    print(f"\n📈 GLOBAL MARKET COUPLING (Avg Corr): {avg_corr:.4f}")
    
    # Generate Report
    report_path = "analysis/correlation_report.md"
    with open(report_path, "w", encoding='utf-8') as f:
        f.write("# 🛡️ Auditoría de Correlación (Protocolo Supremo)\n\n")
        f.write(f"**Fecha:** {datetime.now()}\n")
        f.write(f"**Global Coupling:** {avg_corr:.4f}\n\n")
        f.write("## ⚠️ Pares de Alto Riesgo (>0.90)\n")
        if high_corr_pairs:
            for a, b, c in high_corr_pairs:
                f.write(f"- **{a}** vs **{b}**: `{c:.4f}`\n")
        else:
            f.write("✅ No critical correlations found.\n")
            
        f.write("\n## 📋 Recomendación\n")
        if avg_corr > 0.7:
            f.write("🔴 **ALTO RIESGO SISTÉMICO:** El mercado se mueve en bloque. Reducir exposición total (Max Concurrent: 3).\n")
        elif avg_corr > 0.5:
            f.write("🟠 **RIESGO MODERADO:** Diversificar entradas.\n")
        else:
            f.write("🟢 **BAJO RIESGO:** Mercado desacoplado. Safe to trade aggressively.\n")
            
    print(f"📝 Report generated: {report_path}")

if __name__ == "__main__":
    audit_correlation()
