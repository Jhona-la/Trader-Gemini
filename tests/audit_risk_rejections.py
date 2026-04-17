"""
Forense de Rechazo de Órdenes (RiskManager Reject Tracer)
Simularemos condiciones exactas del backtester para ver DÓNDE frena.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from core.portfolio import Portfolio
from risk.risk_manager import RiskManager
from core.events import SignalEvent, SignalType
from core.enums import OrderSide, OrderType
from datetime import datetime, timezone
import logging

# Set full debug
logging.getLogger('trader_gemini').setLevel(logging.DEBUG)

def trace_risk_manager_rejections():
    print("="*60)
    print("🛡️ INICIANDO AUDITORÍA FORENSE DE RISK MANAGER")
    print("="*60)
    
    # Simulate a micro-account Environment
    fake_portfolio = Portfolio(initial_capital=13.0) 
    # Force $13
    fake_portfolio.cash = 13.0 
    fake_portfolio._total_equity = 13.0
    
    risk_manager = RiskManager(portfolio=fake_portfolio)
    
    # Forge a high-quality mock signal
    mock_signal = SignalEvent(
        strategy_id="ML_SCALPING",
        symbol="BTC/USDT",
        datetime=datetime.now(timezone.utc),
        signal_type=SignalType.LONG,
        strength=0.85,
        horizon="SCALPING",
        atr=650.0,
        sl_pct=0.005,
        tp_pct=0.015,
        metadata={"momentum_exit_accel": -0.012}
    )
    
    current_price = 65000.0

    print(f"[TEST 1] Testing order generation for Micro Account ($13)...")
    
    # Temporarily monkey_patch the validate functions to print exactly when it fails
    def wrap_validator(func, name):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if not result:
                print(f"❌ REJECTED AT: {name}")
            return result
        return wrapper
        
    risk_manager._validate_kill_switch = wrap_validator(risk_manager._validate_kill_switch, "_validate_kill_switch")
    risk_manager._validate_frequency_limits = wrap_validator(risk_manager._validate_frequency_limits, "_validate_frequency_limits")
    risk_manager._validate_regime_veto = wrap_validator(risk_manager._validate_regime_veto, "_validate_regime_veto")
    risk_manager._validate_directional_safety = wrap_validator(risk_manager._validate_directional_safety, "_validate_directional_safety")
    risk_manager._validate_margin_ratio = wrap_validator(risk_manager._validate_margin_ratio, "_validate_margin_ratio")
    risk_manager._validate_fat_finger = wrap_validator(risk_manager._validate_fat_finger, "_validate_fat_finger")
    risk_manager._validate_slippage = wrap_validator(risk_manager._validate_slippage, "_validate_slippage")
    
    # We also need to inspect _calculate_order_params which fails internally
    # We will call it directly
    
    print("\n--- Testing `size_position` ---")
    size_margin = risk_manager.size_position(mock_signal, current_price)
    print(f"Sizing Result: ${size_margin:.4f} USD Margin generated")
    if size_margin == 0:
        print("❌ FAILED AT size_position!")
        
    print("\n--- Testing `_calculate_order_params` internal constraints ---")
    
    # Emulate the internal logic
    leverage = 10 # Let's say Safe Calc returned 10
    notional = size_margin * leverage
    print(f"Calculated Notional: ${notional:.2f} (Required Min: $5.0)")
    if notional < 5.0:
        print(f"❌ MINIMUM NOTIONAL VIOLATION: ${notional:.2f} < $5.00 Threshold!")
        required_margin = (5.0 / leverage) * 1.05
        print(f"   Require margin push to: ${required_margin:.2f}")
        size_margin = required_margin
        notional = required_margin * leverage
    
    fees = risk_manager.fee_calc.calculate_round_trip_fee(notional, order_type='LIMIT')
    expected_profit = notional * mock_signal.tp_pct
    print(f"Fees 2x = ${fees*2.0:.4f}. Expected Profit = ${expected_profit:.4f}")
    
    if expected_profit < (fees * 2.0):
         print(f"❌ FAILED AT FEE-AWARE BLOCK: Expected profit too low compared to fees!")

    print("\n--- Executing Full Pipeline ---")
    order = risk_manager.generate_order(mock_signal, current_price)
    if order is None:
        print("🚨 FULL PIPELINE FAILED (REJECTED)")
    else:
        print(f"✅ PIPELINE SUCCESS: {order}")

if __name__ == "__main__":
    trace_risk_manager_rejections()
