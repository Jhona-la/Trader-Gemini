import os
import sys
import time
import asyncio
import numpy as np
from datetime import datetime, timezone
import queue

# Set working directory to project root
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.engine import Engine
from core.events import MarketEvent, SignalEvent
from core.enums import EventType, SignalType
from core.portfolio import Portfolio
from risk.risk_manager import RiskManager
from data.binance_loader import BinanceData

class MockExecutor:
    def __init__(self):
        self.latency_log = []
        
    def process_order_event(self, event):
        # Medimos cuánto tardó desde que se emitió la señal hasta que llegó al ejecutor
        latency_us = (time.time() - event.metadata.get('_signal_time', time.time())) * 1_000_000
        self.latency_log.append(latency_us)

class MockStrategy:
    def __init__(self, events_queue, symbol="BTC/USDT", horizon="SCALPING"):
        self.events_queue = events_queue
        self.symbol = symbol
        self.horizon = horizon
        self.strategy_id = f"MOCK_{horizon}"

    def calculate_signals(self, event):
        if event.type != EventType.MARKET:
            return
            
        signal = SignalEvent(
            strategy_id=self.strategy_id,
            symbol=self.symbol,
            datetime=datetime.now(timezone.utc),
            signal_type=SignalType.LONG,
            strength=0.8,
            atr=50.0,
            tp_pct=0.01,
            sl_pct=0.005,
            current_price=60000.0,
            leverage=20,
            horizon=self.horizon,
            metadata={'_signal_time': time.time()}
        )
        self.events_queue.put(signal)

async def run_latency_audit():
    print("\n" + "="*50)
    print("🚀 INICIANDO AUDITORÍA FORENSE DE LATENCIA 🚀")
    print("="*50)
    
    events_queue = queue.Queue()
    portfolio = Portfolio(initial_capital=13.0, auto_save=False)
    risk_manager = RiskManager(portfolio)
    data_handler = BinanceData(events_queue, symbol_list=["BTC/USDT"])
    executor = MockExecutor()
    
    engine = Engine(events_queue)
    engine.register_portfolio(portfolio)
    engine.register_risk_manager(risk_manager)
    engine.register_execution_handler(executor)
    
    strat_scalp = MockStrategy(events_queue, horizon="SCALPING")
    strat_swing = MockStrategy(events_queue, horizon="SWING")
    
    engine.register_strategy(strat_scalp)
    engine.register_strategy(strat_swing)
    
    print("✓ Arquitectura base cargada.")
    print("✓ Inyectando 100 señales simultáneas (Stress Test)...")
    
    start_test = time.perf_counter()
    
    # Inyectamos 100 señales
    for i in range(100):
        strat_scalp.calculate_signals(MarketEvent(symbol="BTC/USDT", timestamp=datetime.now(timezone.utc)))
        strat_swing.calculate_signals(MarketEvent(symbol="BTC/USDT", timestamp=datetime.now(timezone.utc)))
    
    # Procesar la cola (El Engine procesa asíncronamente en el backtest)
    while not events_queue.empty():
        event = events_queue.get()
        if event.type == EventType.SIGNAL:
            await engine._process_signal_event(event)
    
    end_test = time.perf_counter()
    total_time_ms = (end_test - start_test) * 1000
    
    print(f"\n📊 RESULTADOS DE LA AUDITORÍA:")
    print(f"Total señales procesadas: 200 (100 Scalp, 100 Swing)")
    print(f"Tiempo Total de Procesamiento: {total_time_ms:.2f} ms")
    print(f"Rendimiento del Event Loop: {(200 / (total_time_ms/1000)):,.0f} signals/second")
    
    if executor.latency_log:
        avg_latency_us = np.mean(executor.latency_log)
        max_latency_us = np.max(executor.latency_log)
        min_latency_us = np.min(executor.latency_log)
        
        print(f"\n⚡ LATENCIA INTERNA (Señal -> Orden de Compra):")
        print(f"Promedio: {avg_latency_us:.2f} microsegundos")
        print(f"Mínimo:   {min_latency_us:.2f} microsegundos")
        print(f"Máximo:   {max_latency_us:.2f} microsegundos")
        
        if avg_latency_us < 1000:
            print("\n✅ VEREDICTO: VELOCIDAD NANO ALCANZADA (< 1ms per decision)")
        else:
            print("\n❌ VEREDICTO: VELOCIDAD LENTA (Se requiere optimización adicional)")
    else:
        print("\n⚠️ Las señales fueron podadas o filtradas por el Risk Manager (Comportamiento esperado si el mercado es bloqueante).")

if __name__ == "__main__":
    asyncio.run(run_latency_audit())
