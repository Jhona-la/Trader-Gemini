import pytest
from risk.risk_manager import RiskManager
from core.portfolio import Portfolio
from core.events import SignalEvent
from core.enums import SignalType

def test_directional_isolation():
    """
    NC-4: No-Colisión Direccional (ISN y Cross-Directions)
    QUÉ: Verifica que el RiskManager bloquee señales contradictorias en el mismo horizonte.
    POR QUÉ: Si tenemos un LONG abierto en Scalping, no podemos abrir un SHORT en Scalping.
    PARA QUÉ: Evitar invalidación de margen y comisiones inútiles.
    """
    portfolio = Portfolio(initial_capital=100.0)
    # Mock virtual_ledger as a plain dict instead of ZmqDictProxy for testing without server
    portfolio.virtual_ledger = {}
    risk_manager = RiskManager(portfolio=portfolio)
    
    # 1. Mock una posición LONG abierta en SCALPING
    portfolio.virtual_ledger["BTC/USDT_SCALPING_LONG"] = {
        "quantity": 0.1,
        "horizon": "SCALPING",
        "avg_price": 50000,
        "direction": "LONG"
    }
    
    # 2. Generar una señal SHORT para el mismo horizonte
    from datetime import datetime, timezone
    signal_short = SignalEvent(
        strategy_id="ML_MODEL",
        symbol="BTC/USDT",
        datetime=datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc),
        signal_type=SignalType.SHORT,
        horizon="SCALPING",
        strength=0.9,
        setup_type="MOMENTUM",
        metadata={}
    )
    
    # 3. Validar con el Risk Manager
    order = risk_manager.generate_order(signal_short, current_price=50100)
    
    # The order should be None because of DIRECTIONAL_SAFETY
    assert order is None, "CRITICAL COLLISION: RiskManager permitió un SHORT mientras hay un LONG abierto en el mismo horizonte."
    
    rejection_reason = signal_short.metadata.get("rejection_reason", "")
    assert "DIRECTIONAL_DUPLICATION" in rejection_reason or "DIRECTION" in rejection_reason.upper() or "ISOLATION" in rejection_reason.upper(), \
        f"Esperado rechazo direccional, pero fue: {rejection_reason}"
    
    print("✅ [NC-4] Directional Isolation Test Passed.")

if __name__ == "__main__":
    test_directional_isolation()
