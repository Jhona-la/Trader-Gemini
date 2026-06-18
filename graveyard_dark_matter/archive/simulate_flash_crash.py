
import asyncio
import pandas as pd
from datetime import datetime, timezone
from core.portfolio import Portfolio
from core.events import MarketEvent, EventType
from core.enums import OrderSide
from config import Config
from utils.logger import logger
import queue

async def simulate_flash_crash():
    print("\n" + "="*50)
    print("🚨 SIMULACIÓN DE FLASH CRASH: TEST DE RESILIENCIA 🚨")
    print("="*50)
    
    # 1. SETUP: Mock Portfolio & Queue
    events_queue = queue.Queue()
    portfolio = Portfolio(initial_capital=13.50)
    
    # Capital inicial de prueba
    portfolio.current_cash = 13.50
    symbol = "BTC/USDT"
    
    print(f"📊 Estado Inicial: Balance ${portfolio.current_cash:.2f}")
    
    # 2. ABRIR POSICIÓN (Simulación de entrada en $50,000)
    entry_price = 50000.0
    qty = 0.0002 # ~$10 notional
    
    # Simulamos un evento de FILL para que el portafolio registre la posición
    class MockFillEvent:
        def __init__(self, symbol, quantity, fill_cost, direction):
            self.type = EventType.FILL
            self.symbol = symbol
            self.quantity = quantity
            self.fill_cost = fill_cost
            self.direction = direction
            self.strategy_id = "SNIPER_TEST"
            self.exchange = "BINANCE"

    fill = MockFillEvent(symbol, qty, entry_price * qty, OrderSide.BUY)
    portfolio.update_fill(fill)
    
    print(f"✅ Posición Abierta: {qty} {symbol} @ ${entry_price}")
    print(f"📈 PnL Actual: 0.00%")

    # 3. ESCENARIO DE DESPLOME (Flash Crash)
    # Bajada repentina del 1.5% (Supera el SL del 0.3%)
    crash_price = entry_price * 0.985 # $49,250
    
    print(f"\n⚡ FLASH CRASH DETECTADO: El precio cae a ${crash_price} (-1.5%)")
    
    # Actualizamos el precio en el portafolio
    portfolio.update_market_price(symbol, crash_price)
    
    # 4. EJECUTAR CHEQUEO DE SALIDA (Safety Net)
    print("🛡️ Ejecutando 'check_exits' del Portafolio...")
    
    # Mock de DataProvider para que check_exits no falle
    class MockDP:
        def get_latest_price(self, sym): return crash_price
    
    portfolio.check_exits(MockDP(), events_queue)
    
    # 5. VERIFICAR RESULTADOS
    try:
        exit_signal = events_queue.get_nowait()
        print(f"\n🔥 [RESULTADO]: ¡SALIDA DE EMERGENCIA DISPARADA!")
        print(f"   Tipo: {exit_signal.type}")
        print(f"   Símbolo: {exit_signal.symbol}")
        print(f"   Timestamp: {exit_signal.datetime}")
    except queue.Empty:
        print("\n❌ [ERROR]: El portafolio NO disparó la salida. El capital está en riesgo.")

    print("\n" + "="*50)
    print("🏁 FIN DE SIMULACIÓN")
    print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(simulate_flash_crash())
