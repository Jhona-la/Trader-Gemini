import os
import sys
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.portfolio import Portfolio
from core.events import SignalEvent, SignalType, FillEvent
from core.enums import OrderSide

def simulate_dual_horizon():
    print("=== Iniciando Verificación Forense: Dual Horizon ===")
    
    # 1. Setup Mock Data Provider y Portfolio
    mock_data_provider = MagicMock()
    mock_data_provider.current_time_ms = int(time.time() * 1000)
    
    portfolio = Portfolio(
        initial_capital=13.0
    )
    portfolio.data_provider = mock_data_provider
    portfolio.engine_mode = "BACKTEST"
    
    symbol = "BTC/USDT"
    current_price = 100000.0
    portfolio._last_prices = {symbol: current_price}
    
    print("\n[Paso 1] Generando Fill SCALPING LONG...")
    fill_scalp = FillEvent(
        timeindex=datetime.now(timezone.utc),
        symbol=symbol,
        exchange="BINANCE",
        quantity=0.0002, # $20 notional
        direction=OrderSide.BUY,
        fill_cost=20.0,
        strategy_id="SCALPING",
        horizon="SCALPING",
        metadata={"setup_type": "BREAKOUT", "leverage": 20, "binance_position_side": "LONG"}
    )
    
    portfolio._update_virtual_ledger(fill_scalp)
    
    print("\n[Paso 2] Generando Fill SWING LONG...")
    fill_swing = FillEvent(
        timeindex=datetime.now(timezone.utc),
        symbol=symbol,
        exchange="BINANCE",
        quantity=0.0001, # $10 notional
        direction=OrderSide.BUY,
        fill_cost=10.0,
        strategy_id="SWING",
        horizon="SWING",
        metadata={"setup_type": "PULLBACK", "leverage": 5, "binance_position_side": "LONG"}
    )
    portfolio._update_virtual_ledger(fill_swing)
    
    # Update global positions mock just like OrderManager would do
    portfolio.positions[symbol] = {
        "quantity": 0.0003, # 0.0002 + 0.0001
        "avg_price": current_price,
        "current_price": current_price
    }
    
    print("\n[Validación A] Estado del Virtual Ledger (Entradas)")
    print(f"Total Virtual Ledgers: {len(portfolio.virtual_ledger)}")
    for k, v in portfolio.virtual_ledger.items():
        print(f"  - {k}: qty={v['quantity']}, entry={v.get('avg_price')}, horizon={v.get('horizon')}")
    
    print(f"Global Position (Binance): {portfolio.positions[symbol]['quantity']}")
    assert len(portfolio.virtual_ledger) == 2, "Deberían existir 2 ledgers virtuales aislados"
    
    # 3. Simulate SCALP EXIT
    print("\n[Paso 3] Generando Fill de EXIT solo para SCALPING...")
    fill_exit = FillEvent(
        timeindex=datetime.now(timezone.utc),
        symbol=symbol,
        exchange="BINANCE",
        quantity=0.0002,
        direction=OrderSide.SELL,
        fill_cost=20.0,
        strategy_id="SCALPING",
        horizon="SCALPING",
        metadata={"exit_reason": "TAKE_PROFIT", "is_exit": True, "binance_position_side": "LONG"}
    )
    
    # Simulate Order Execution closing just the Scalping portion
    portfolio._update_virtual_ledger(fill_exit)
    portfolio.positions[symbol]["quantity"] += -0.0002
        
    print("\n[Validación C] Estado Final del Virtual Ledger")
    for k, v in portfolio.virtual_ledger.items():
         print(f"  - {k}: qty={v['quantity']}")
         
    print(f"Global Position (Binance) Final: {portfolio.positions[symbol]['quantity']}")
    
    assert portfolio.virtual_ledger[f"{symbol}_SCALPING_LONG"]['quantity'] == 0, "Scalping debe estar cerrado"
    assert portfolio.virtual_ledger[f"{symbol}_SWING_LONG"]['quantity'] == 0.0001, "Swing debe mantenerse intacto"
    
    print("\n✅ VALIDACIÓN FORENSE EXITOSA: Aislamiento total de horizontes comprobado.")

if __name__ == "__main__":
    simulate_dual_horizon()
