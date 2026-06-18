from core.portfolio import Portfolio
from core.events import FillEvent
from core.events import OrderSide
import time

def test_portfolio_coherence():
    print("🚀 [TEST 2] COHERENCIA DE PORTAFOLIO Y ESTADOS")
    print("------------------------------------------------------")
    
    port = Portfolio()
    
    # Simular un FillEvent de Entrada (LONG)
    entry_fill = FillEvent(
        symbol="ETH/USDT",
        timestamp_ns=time.time_ns(),
        quantity=1.5,
        price=3000.00,
        direction=OrderSide.BUY,
        fill_cost=4500.00,
        commission=0.0,
        strategy_id="TEST_COHERENCE",
        metadata={"mock": True}
    )
    
    print("🟢 Inyectando Fill de ENTRADA: 1.5 ETH @ $3000")
    pnl = port.update_fill(entry_fill)
    pos = port.positions.get("ETH/USDT")
    print(f"Estado de Posición tras entrada: Qty: {pos.get('quantity')}, AvgPrice: {pos.get('avg_price')}")
    print(f"PnL reportado al engine: {pnl} (Debería ser None según la auditoría Fase 2)")
    
    # Simular una actualización de precio de mercado (Marca de Agua / Trailing)
    print("\n📈 El precio sube a $3100. Verificando Marca de Agua...")
    port.update_market_price("ETH/USDT", 3100.0)
    print(f"High Water Mark: {pos.get('high_water_mark')} (Debería ser 3100.0)")
    
    # Simular un FillEvent de Salida (Cierre Completo)
    exit_fill = FillEvent(
        symbol="ETH/USDT",
        timestamp_ns=time.time_ns(),
        quantity=1.5,
        price=3100.00,
        direction=OrderSide.SELL,
        fill_cost=4650.00,
        commission=0.0,
        strategy_id="TEST_COHERENCE",
        metadata={"mock": True}
    )
    
    print("\n🔴 Inyectando Fill de SALIDA: 1.5 ETH @ $3100")
    pnl = port.update_fill(exit_fill)
    pos = port.positions.get("ETH/USDT")
    print(f"Estado de Posición tras salida: Qty: {pos.get('quantity')}, AvgPrice: {pos.get('avg_price')}")
    print(f"PnL reportado al engine: {pnl} (Debería ser > 0, ganancia de ~$150)")

if __name__ == "__main__":
    test_portfolio_coherence()
