import pytest
import numpy as np
from strategies.technical import HybridScalpingStrategy
from core.events import MarketEvent

def test_strategy_memory_isolation():
    """
    NC-2: No-Colisión entre Estrategias (State Isolation)
    QUÉ: Verifica que la memoria y parámetros de HybridScalpingStrategy instanciada para Scalping
         no comparta memoria con una instanciada para Swing.
    POR QUÉ: Compartición mutante en diccionarios de clases Python (Mutable Default Arguments).
    PARA QUÉ: Evitar que las señales de SWING disparen cierres de SCALPING.
    """
    import queue
    events_queue = queue.Queue()
    strat_scalp = HybridScalpingStrategy(data_provider=None, events_queue=events_queue, horizon="SCALPING")
    strat_swing = HybridScalpingStrategy(data_provider=None, events_queue=events_queue, horizon="SWING")
    strat_scalp.strategy_id = "SCALPING_STRAT"
    strat_swing.strategy_id = "SWING_STRAT"
    
    # Inyectar una 'posición' en la memoria de SCALPING
    mock_event = MarketEvent(symbol="BTC/USDT", close_price=50000)
    strat_scalp.current_horizon = "SCALPING"
    strat_swing.current_horizon = "SWING"
    
    # Manually mutate a state variable to see if it leaks
    strat_scalp.cognitive_memory["test_key"] = "leaked_value"
    
    assert "test_key" not in strat_swing.cognitive_memory, "CRITICAL COLLISION: HybridScalpingStrategy comparte `cognitive_memory` entre instancias."
    
    print("✅ [NC-2] Strategy Memory Isolation Test Passed.")

if __name__ == "__main__":
    test_strategy_memory_isolation()
