import os
import sys
import logging
import asyncio
from datetime import datetime, timedelta
import random
import numpy as np

# Asegurar path correcto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import logger
from core.events import OrderSide, OrderType

# Import the actual god mode script
import scripts.run_god_mode_backtest as gb

# ==============================================================================
# MONKEY PATCH PARA SLIPPAGE DINÁMICO
# ==============================================================================
# Inyectamos una simulación de slippage brutal para Market orders
original_execute = gb.BacktestExecutor.execute_order

def patched_execute_order(self, order_event, current_price=None):
    # Call original just for checks
    if order_event.metadata and (order_event.metadata.get("is_tp_limit") or order_event.metadata.get("is_sl_limit")):
        return None

    price = current_price or order_event.price
    if not price or price <= 0:
        return None

    qty = order_event.quantity
    if qty <= 0:
        return None

    is_limit = order_event.order_type == OrderType.LIMIT
    if is_limit:
        latency_ms = self._rng.lognormal(2.3, 0.5)
        if latency_ms > 50.0 and self._rng.random() < 0.30:
            return None
            
        fill_ratio = self._rng.beta(5, 1)
        if fill_ratio > 0.95:
            fill_ratio = 1.0
            
        qty = qty * fill_ratio
        if qty < 1e-8:
            return None
            
        slip_pct = 0.0
        commission = (price * qty) * 0.0002 # 0.02%
    else:
        # SLIPPAGE DINÁMICO EXTREMO (0.5% - 1.5%)
        # En vez del lognormal suave, simulamos un vacío de liquidez
        # Base: 0.5%
        # Ruido de cascada: + 0.0% a 1.0%
        slip_pct = 0.005 + random.uniform(0.0, 0.01)
        commission = (price * qty) * 0.0004 # 0.04%

    if order_event.direction == OrderSide.BUY:
        fill_price = price * (1 + slip_pct)
    else:
        fill_price = price * (1 - slip_pct)

    fill_cost = fill_price * qty
    
    self.fills_count += 1
    b_order_id = f"BT_{self.fills_count}"

    from core.events import FillEvent
    return FillEvent(
        timeindex=getattr(order_event, "datetime", datetime.utcnow()),
        symbol=order_event.symbol,
        exchange="BINANCE_MOCK",
        order_id=b_order_id,
        direction=order_event.direction,
        fill_price=fill_price,
        fill_cost=fill_cost,
        quantity=qty,
        commission=commission,
        trade_id=f"T_{self.fills_count}",
        metadata=order_event.metadata
    )

gb.BacktestExecutor.execute_order = patched_execute_order

# ==============================================================================
# EJECUCIÓN
# ==============================================================================
if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("🔬 FASE I: STRESS TEST DE SLIPPAGE DINÁMICO (VACÍO DE LIQUIDEZ)")
    logger.info("Slippage Inyectado: 0.5% - 1.5% (Taker Market Orders)")
    logger.info("=" * 80)
    
    import sys
    sys.argv = ["scripts/run_god_mode_backtest.py", "--days", "1", "--symbols", "ALL"]
    gb.main()
