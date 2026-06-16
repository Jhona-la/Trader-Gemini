import asyncio
import time
from datetime import datetime, timezone, timedelta
import pandas as pd
from collections import defaultdict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data.binance_loader import BinanceData
from core.events import MarketEvent, SignalType
from strategies.technical import HybridScalpingStrategy

class MockQueue:
    def __init__(self):
        self.events = []
        
    async def put(self, event):
        self.events.append(event)
    
    def put_nowait(self, event):
        self.events.append(event)

async def generate_report():
    print("🚀 Iniciando Reporte Nano-Core Express (26 Monedas)...")
    
    symbols = getattr(Config.Data, 'SYMBOLS', ["BTC/USDT", "ETH/USDT"])
    if not symbols:
        symbols = ["BTC/USDT"]
        
    print(f"📋 Symbols ({len(symbols)}): {', '.join(symbols[:5])}...")
    
    loader = BinanceData(events_queue=None, symbol_list=symbols)
    loader.fetch_initial_history()
    
    mock_q = MockQueue()
    
    results = []
    
    # Process symbols
    for sym in symbols:
        print(f"\nProcesando {sym}...")
        df = loader.get_latest_bars(sym, n=500)
        if df is None or len(df) == 0:
            print(f"⚠️ No data para {sym}")
            continue
            
        print(f"📊 {len(df)} barras de {sym} recuperadas.")
        
        # Initialize strategy
        tech_strat = HybridScalpingStrategy(data_provider=loader, events_queue=mock_q, genotype=None, horizon="SCALPING")
        
        latencies = []
        signals = 0
        buys = 0
        sells = 0
        
        # Simulate last 500 bars to see signals
        start_idx = max(0, len(df) - 500)
        
        for i in range(start_idx, len(df)):
            row = df[i]
            event = MarketEvent(
                symbol=sym,
                timestamp=pd.to_datetime(row['timestamp'], unit='ms', utc=True),
                open_price=float(row['open']),
                high_price=float(row['high']),
                low_price=float(row['low']),
                close_price=float(row['close']),
                volume=row['volume'],
                is_closed=True  # Important to avoid repainting block
            )
            
            t0 = time.perf_counter()
            tech_strat.calculate_signals(event)
            t1 = time.perf_counter()
            
            latencies.append((t1 - t0) * 1000) # in ms
            
        # Count signals for this symbol
        sym_events = [e for e in mock_q.events if e.symbol == sym]
        for e in sym_events:
            if e.signal_type == SignalType.BUY: buys += 1
            elif e.signal_type == SignalType.SELL: sells += 1
            
        avg_lat = sum(latencies)/len(latencies) if latencies else 0
        
        print(f"✅ {sym}: {buys} BUYs, {sells} SELLs | Avg Latency: {avg_lat:.3f} ms")
        
        results.append({
            'Symbol': sym,
            'BUYs': buys,
            'SELLs': sells,
            'Total Signals': buys + sells,
            'Avg Latency (ms)': avg_lat
        })
        
    print("\n" + "="*50)
    print("📈 REPORTE NANO-CORE (LATENCIA Y SEÑALES)")
    print("="*50)
    
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    
    print("\n" + "="*50)
    total_latency = df_res['Avg Latency (ms)'].mean()
    total_signals = df_res['Total Signals'].sum()
    print(f"🚀 Overall Avg Latency: {total_latency:.4f} ms")
    print(f"🎯 Total Signals (last 500 bars): {total_signals}")
    
    if total_latency < 1.0:
        print("✅ LATENCIA APROBADA (Sub-milisecond execution!)")
    else:
        print("⚠️ LATENCIA SUB-OPTIMA (Se requiere profiling adicional)")

if __name__ == "__main__":
    asyncio.run(generate_report())
