import pytest
import numpy as np
from core.asset_parameter_engine import AssetParameterEngine
from config import Config

def test_asset_parameter_engine_isolation():
    """
    NC-1: No-Colisión entre Activos (ATR Isolation)
    QUÉ: Verifica que la volatilidad (ATR) calculada para BTC no contamina la de SOL.
    POR QUÉ: Si el motor de parámetros mezcla volatilidad, los stops de SOL se pondrán con el ATR de BTC.
    PARA QUÉ: Garantizar protección matemática aislada.
    """
    engine = AssetParameterEngine()
    
    # 1. Simular mercado para BTC (Volatilidad Normal)
    # Rango de precios: ~60000, velas de 5 min con movimientos de ~150 USD (0.25% ATR)
    btc_closes = np.linspace(60000, 60100, 50)
    btc_highs = btc_closes + 150
    btc_lows = btc_closes - 150
    
    btc_bars = {
        'high': btc_highs,
        'low': btc_lows,
        'close': btc_closes
    }
    
    # 2. Simular mercado para SOL (Volatilidad Extrema)
    # Rango de precios: ~150, velas de 5 min con movimientos de ~3 USD (2.00% ATR)
    sol_closes = np.linspace(150, 155, 50)
    sol_highs = sol_closes + 3
    sol_lows = sol_closes - 3
    
    sol_bars = {
        'high': sol_highs,
        'low': sol_lows,
        'close': sol_closes
    }
    
    # Evaluar parámetros
    engine.calibrate_from_bars("BTC/USDT", btc_closes, btc_highs, btc_lows, "SCALPING")
    engine.calibrate_from_bars("SOL/USDT", sol_closes, sol_highs, sol_lows, "SCALPING")
    
    btc_params = engine.get_params("BTC/USDT", "SCALPING")
    sol_params = engine.get_params("SOL/USDT", "SCALPING")
    
    # Extraer el ratio implicado del stop loss (que depende del ATR)
    # BTC ATR es ~150/60000 = 0.25%. SL debería estar cerca de 0.25% * multiplicador
    # SOL ATR es ~3/150 = 2.00%. SL debería estar cerca de 2.00% * multiplicador
    btc_sl = btc_params['stop_loss_pct']
    sol_sl = sol_params['stop_loss_pct']
    
    assert btc_sl != sol_sl, "CRITICAL COLLISION: SL de BTC y SOL son idénticos."
    assert sol_sl > btc_sl * 2, f"EXPECTED NO-COLLISION FAULT: SOL SL ({sol_sl*100:.2f}%) debería ser mucho mayor que BTC SL ({btc_sl*100:.2f}%) debido a volatilidad."
    
    print("✅ [NC-1] Asset Parameter Engine ATR Isolation Test Passed.")

if __name__ == "__main__":
    test_asset_parameter_engine_isolation()
