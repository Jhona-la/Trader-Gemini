import sys
import os

# Ensure the root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.nano_core import calculate_unrealized_pnl_fast, calculate_kelly_fraction, update_hwm_lwm
import numpy as np

def test_pnl_parity():
    # Long position
    curr_price = 105.0
    entry_price = 100.0
    qty = 2.0
    direction = 1
    
    # expected = (105-100)/100 * 2 * 100 = 0.05 * 200 = 10
    pnl = calculate_unrealized_pnl_fast(curr_price, entry_price, qty, direction)
    assert abs(pnl - 10.0) < 1e-6, f"Expected 10.0, got {pnl}"
    
    # Short position
    curr_price = 95.0
    entry_price = 100.0
    qty = 2.0
    direction = -1
    
    # expected = (100-95)/100 * 2 * 100 = 0.05 * 200 = 10
    pnl = calculate_unrealized_pnl_fast(curr_price, entry_price, qty, direction)
    assert abs(pnl - 10.0) < 1e-6, f"Expected 10.0, got {pnl}"

    print("✅ PnL scalar function test passed")

def test_kelly_parity():
    winrate = 0.55
    payoff = 2.0
    # expected (0.55 * 2 - 0.45) / 2 = (1.1 - 0.45) / 2 = 0.65 / 2 = 0.325
    k = calculate_kelly_fraction(0, 0, winrate, payoff, 0.5, 100.0, apply_mult=False)
    assert abs(k - 0.325) < 1e-4, f"Expected 0.325, got {k}"
    
    print("✅ Kelly function test passed")

def test_hwm_lwm():
    h, l = update_hwm_lwm(105.0, 100.0, 95.0)
    assert h == 105.0 and l == 95.0
    
    h, l = update_hwm_lwm(90.0, 100.0, 95.0)
    assert h == 100.0 and l == 90.0
    
    print("✅ HWM/LWM function test passed")

if __name__ == "__main__":
    print("Running Nano Core Parity Tests...")
    test_pnl_parity()
    test_kelly_parity()
    test_hwm_lwm()
    print("All tests passed.")
