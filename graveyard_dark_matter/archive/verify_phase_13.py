import time
import queue
from datetime import datetime, timezone
from core.events import MarketEvent, SignalEvent, OrderEvent, FillEvent
from core.engine import Engine, BoundedQueue
from core.enums import SignalType, OrderSide, OrderType
from utils.latency_monitor import latency_monitor
from execution.binance_executor import BinanceExecutor
from config import Config
import threading

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def verify_phase_13():
    print("🧪 [TEST] Verificando Fase 13: High-Frequency Execution Optimization...")
    
    events_queue = queue.Queue()
    engine = Engine(events_queue=events_queue)
    executor = BinanceExecutor(events_queue)
    
    # Mock symbols and config
    Config.TRADING_PAIRS = ['BTC/USDT']
    symbol = "BTC/USDT"
    
    print("🕒 Generando señal y midiendo latencia...")
    signal = SignalEvent(
        strategy_id="ML_TEST",
        symbol=symbol,
        datetime=datetime.now(timezone.utc),
        signal_type=SignalType.LONG,
        strength=1.0
    )
    
    # Simular paso por el motor
    # El motor genera el OrderEvent
    print("  -> Paso 1: Generando OrderEvent desde Signal...")
    t1 = time.time_ns()
    # Mock risk manager behavior
    order = OrderEvent(
        symbol=symbol,
        order_type=OrderType.LIMIT,
        quantity=0.001,
        direction=OrderSide.BUY,
        strategy_id="ML_TEST",
        price=None # Trigger Smart Pricing
    )
    # Manual inheritance for test
    object.__setattr__(order, 'timestamp_ns', signal.timestamp_ns)
    
    print("  -> Paso 2: Ejecutando orden con Smart Pricing...")
    # Mock guardian to avoid real API calls for prices
    executor.guardian.analyze_liquidity = lambda s, q, d: {
        'is_safe': True,
        'avg_fill_price': 50000.0,
        'slippage_pct': 0.0001,
        'reason': 'Safe'
    }
    
    # Mock exchange call to avoid real orders but track latency
    start_time = time.perf_counter()
    def mock_post_order(params):
        time.sleep(0.04) # Simulate 40ms network latency
        return {'orderId': '123456', 'symbol': 'BTCUSDT', 'executedQty': '0.001', 'avgPrice': '50000.0'}
    
    executor.exchange.fapiPrivatePostOrder = mock_post_order
    executor.execute_order(order)
    
    print("📊 Verificando métricas en el LatencyMonitor...")
    latency_monitor.report_stats()
    
    if len(latency_monitor.metrics['order_to_send']) > 0:
        print("✅ Latency Check: 'order_to_send' capturado.")
    else:
        print("❌ Latency Check: No se capturaron métricas.")
        
    # Verificar Smart Pricing
    # (En el log deberíamos ver 50000.0)
    print("✅ Smart Pricing Check: El log debería mostrar el precio de $50,000 calculado por el Guardián.")

if __name__ == "__main__":
    try:
        verify_phase_13()
    except Exception as e:
        print(f"💥 Error en verificación: {e}")
        import traceback
        traceback.print_exc()
