import os
import sys
import asyncio
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import TradingEngine
from data.data_provider import DataProvider
from core.events import MarketEvent, EventType, OrderEvent
from core.enums import OrderType, OrderSide
from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SmokeTest')

async def simulate_multi_horizon_bombardment(engine):
    """
    Bombardea el Engine con señales cruzadas (Scalping Short y Swing Long).
    """
    logger.info("💥 Iniciando Ráfaga HFT en Hedge Mode...")
    
    # Preparamos las señales
    signals = [
        OrderEvent(symbol="BTC/USDT", order_type=OrderType.MARKET, quantity=0.003, direction=OrderSide.SELL, price=None, horizon="SCALPING", strategy_id="Scalp_1"),
        OrderEvent(symbol="BTC/USDT", order_type=OrderType.MARKET, quantity=0.050, direction=OrderSide.BUY, price=None, horizon="SWING", strategy_id="Swing_1"),
        # Enviar otra de scalping casi inmediata
        OrderEvent(symbol="BTC/USDT", order_type=OrderType.MARKET, quantity=0.001, direction=OrderSide.BUY, price=None, horizon="SCALPING", strategy_id="Scalp_1"),
    ]
    
    t0 = time.time()
    
    # Emitimos las señales a la vez usando el queue del Engine si soporta ráfagas, 
    # o llamando directamente al despachador.
    for sig in signals:
        engine.events.put(sig)
        
    # Dejamos que el motor procese por un momento
    await asyncio.sleep(1.0)
    
    t1 = time.time()
    logger.info(f"⚡ Ráfaga Completada en {(t1-t0)*1000:.2f} ms")
    
    # Auditamos el Virtual Ledger
    logger.info("📓 Estado Final del Virtual Ledger:")
    for k, v in engine.portfolio.virtual_ledger.items():
        logger.info(f"   {k}: {v['quantity']} @ {v['avg_price']}")
        
    logger.info("✅ Smoke Test Finalizado exitosamente.")

async def main():
    Config.BINANCE_USE_FUTURES = True
    Config.BINANCE_USE_DEMO = True # SAFETY FIRST
    
    logger.info("Inicializando Engine en modo Hedge Concurrente...")
    engine = TradingEngine(run_web_dashboard=False)
    
    # Start engine loop in background
    task = asyncio.create_task(engine.start_async())
    
    # Wait for init
    await asyncio.sleep(2)
    
    # Run the test
    await simulate_multi_horizon_bombardment(engine)
    
    engine.running = False
    
    # Wait for shutdown
    try:
        await asyncio.wait_for(task, timeout=2.0)
    except asyncio.TimeoutError:
        from utils.error_handler import SystemIntegrityError
        raise SystemIntegrityError('Silent fallback blocked by Holographic Audit')

if __name__ == '__main__':
    asyncio.run(main())
