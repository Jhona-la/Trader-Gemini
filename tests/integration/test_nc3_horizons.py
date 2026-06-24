import pytest
from core.portfolio import Portfolio
from core.events import OrderEvent
from core.enums import OrderType, OrderSide

def test_hedge_mode_horizon_isolation():
    """
    NC-3: No-Colisión entre Horizontes (Hedge Mode Simulado)
    QUÉ: Verifica que la clave v_key = {symbol}_{horizon}_{pos_side} mantiene posiciones aisladas.
    POR QUÉ: Binace permite Hedge Mode (Long y Short simultáneos). En un modelo multi-horizonte,
             un LONG de Scalping no debe cancelar un SHORT de Swing.
    PARA QUÉ: Simultaneidad de horizontes comprobada matemáticamente.
    """
    portfolio = Portfolio(initial_capital=100.0)
    # Mock ZmqDictProxies as plain dicts for testing without server
    portfolio.virtual_ledger = {}
    portfolio.positions = {}
    portfolio._net_intended_positions = {}
    
    # Simular ejecución Binance de un LONG en Scalping
    order_scalp_long = OrderEvent(
        symbol="BTC/USDT",
        order_type=OrderType.MARKET,
        quantity=0.1,
        direction="LONG",
        horizon="SCALPING"
    )
    from core.events import FillEvent
    from datetime import datetime, timezone

    # Rellenar la orden a mano (mockeando BinanceExecutor callback)
    portfolio.update_fill(FillEvent(
        timeindex=datetime.now(timezone.utc),
        symbol="BTC/USDT",
        exchange="BINANCE",
        quantity=0.1,
        direction=OrderSide.BUY,
        fill_cost=0.1 * 50000,
        fill_price=50000,
        order_id="123",
        horizon="SCALPING",
        leverage=10,
        metadata={"is_close": False, "is_exit": False, "actual_order_type": "MAKER", "client_order_id": "123", "binance_position_side": "LONG", "dollar_size": 5000, "ml_confidence": 0.85, "trajectory_prediction": "MOMENTUM"}
    ))
    
    # Simular ejecución Binance de un SHORT en Swing para el MISMO SÍMBOLO
    order_swing_short = OrderEvent(
        symbol="BTC/USDT",
        order_type=OrderType.MARKET,
        quantity=0.05,
        direction="SHORT",
        horizon="SWING"
    )
    portfolio.update_fill(FillEvent(
        timeindex=datetime.now(timezone.utc),
        symbol="BTC/USDT",
        exchange="BINANCE",
        quantity=0.05,
        direction=OrderSide.SELL,
        fill_cost=0.05 * 50100,
        fill_price=50100,
        order_id="456",
        horizon="SWING",
        leverage=10,
        metadata={"is_close": False, "is_exit": False, "actual_order_type": "MAKER", "client_order_id": "456", "binance_position_side": "SHORT", "dollar_size": 2505, "ml_confidence": 0.85, "trajectory_prediction": "MOMENTUM"}
    ))
    
    # Verificamos virtual ledger (que es la fuente de verdad del bot)
    assert "BTC/USDT_SCALPING_LONG" in portfolio.virtual_ledger, "SCALPING_LONG position missing."
    assert "BTC/USDT_SWING_SHORT" in portfolio.virtual_ledger, "SWING_SHORT position missing."
    
    scalp_pos = portfolio.virtual_ledger["BTC/USDT_SCALPING_LONG"]
    swing_pos = portfolio.virtual_ledger["BTC/USDT_SWING_SHORT"]
    
    assert scalp_pos['quantity'] == 0.1, "Scalp quantity corrupted."
    assert swing_pos['quantity'] == -0.05, "Swing quantity corrupted (should be negative for SHORT)."
    
    print("✅ [NC-3] Horizon Isolation (Hedge Mode) Test Passed.")

if __name__ == "__main__":
    test_hedge_mode_horizon_isolation()
