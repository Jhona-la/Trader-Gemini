import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import asyncio
import pandas as pd
from datetime import datetime, timezone
from config import Config
from data.binance_loader import BinanceData
from strategies.technical import HybridScalpingStrategy
from strategies.ml_strategy import MLStrategyHybridUltimate
from core.events import MarketEvent

class MockQueue:
    def __init__(self):
        self.last_event = None
    def put(self, event, block=False):
        print(f"MockQueue received event: {event.signal_type} for {event.symbol} / {event.horizon}")
        self.last_event = event
    def put_nowait(self, event):
        self.put(event)

async def audit_flow():
    print("🚀 Iniciando Auditoria Forense de Flujo de Datos...")
    
    loader = BinanceData(events_queue=None, symbol_list=["BTC/USDT"])
    loader.fetch_initial_history()
    df = loader.get_latest_bars("BTC/USDT", n=100)
    
    if df is None or len(df) == 0:
        print("❌ df vacio despues de get_latest_bars")
        return
        
    print(f"✅ Data recuperada exitosamente: {len(df)} barras")
    
    mock_q = MockQueue()
    tech_strat = HybridScalpingStrategy(data_provider=loader, events_queue=mock_q, genotype=None, horizon="SCALPING")
    ml_strat = MLStrategyHybridUltimate(data_provider=loader, events_queue=mock_q)
    
    last_row = df[-1]
    event = MarketEvent(
        symbol="BTC/USDT",
        timestamp=datetime.now(timezone.utc),
        close_price=last_row['close'],
        high_price=last_row['high'],
        low_price=last_row['low'],
        order_flow={},
        health_metrics={},
        is_closed=True
    )
    
    # 1. Technical Strategy
    try:
        mock_q.last_event = None
        tech_strat.calculate_signals(event)
        if mock_q.last_event:
             print(f"✅ Technical Strategy: Señal={mock_q.last_event.signal_type}, Meta={mock_q.last_event.metadata}")
        else:
             print("✅ Technical Strategy: No signal generated")
    except Exception as e:
        print(f"❌ Technical Strategy Error: {e}")
        
    # 2. ML Strategy
    try:
        mock_q.last_event = None
        ml_strat.calculate_signals(event)
        
        # In case the ML strategy puts the event async, let's yield briefly
        await asyncio.sleep(0.5)
        
        if mock_q.last_event:
            print(f"✅ ML Strategy: Señal={mock_q.last_event.signal_type}, Meta={mock_q.last_event.metadata}")
        else:
            print("✅ ML Strategy: No signal generated")
    except Exception as e:
        print(f"❌ ML Strategy Error: {e}")

if __name__ == "__main__":
    asyncio.run(audit_flow())
