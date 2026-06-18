import numpy as np
from numba import njit

@njit(fastmath=True, cache=True)
def calculate_unrealized_pnl_batch(avg_prices, current_prices, quantities):
    """
    [NANO-SPEED] Vectorized calculation of Unrealized PnL.
    quantities > 0 means LONG, quantities < 0 means SHORT.
    """
    n = len(avg_prices)
    total_pnl = 0.0
    for i in range(n):
        qty = quantities[i]
        if qty != 0:
            if qty > 0:
                total_pnl += (current_prices[i] - avg_prices[i]) * qty
            else:
                total_pnl += (avg_prices[i] - current_prices[i]) * qty # negative qty subtraction handled correctly when multiplied
                # wait, if qty is negative, qty = -abs(qty). 
                # PnL short = (Entry - Current) * abs(qty) = (avg_price - current_price) * (-qty)
                # which is equal to (current_price - avg_price) * qty
                # So we can just do (current_price - avg_price) * qty for BOTH long and short!
    return total_pnl

@njit(fastmath=True, cache=True)
def calculate_used_margin_batch(avg_prices, quantities, leverages):
    """
    [NANO-SPEED] Vectorized calculation of Used Margin.
    """
    n = len(avg_prices)
    total_margin = 0.0
    for i in range(n):
        qty = np.abs(quantities[i])
        lev = leverages[i]
        if qty > 0 and lev > 0:
            total_margin += (avg_prices[i] * qty) / lev
    return total_margin

@njit(fastmath=True, cache=True)
def calculate_portfolio_exposure_batch(avg_prices, quantities):
    """
    [NANO-SPEED] Vectorized calculation of total portfolio exposure.
    Returns: (total_long_beta, total_short_beta, net_delta)
    """
    n = len(avg_prices)
    total_long = 0.0
    total_short = 0.0
    net_delta = 0.0
    
    for i in range(n):
        qty = quantities[i]
        if qty != 0:
            notional = np.abs(qty) * avg_prices[i]
            if qty > 0:
                total_long += notional
                net_delta += notional
            else:
                total_short += notional
                net_delta -= notional
                
    return total_long, total_short, net_delta

@njit(fastmath=True, cache=True)
def calculate_omni_float_pnl_batch(avg_prices, current_prices, quantities):
    """
    [NANO-SPEED] Calculate sum of only positive floating PnL.
    """
    n = len(avg_prices)
    omni_pnl = 0.0
    for i in range(n):
        qty = quantities[i]
        if qty != 0:
            trade_pnl = (current_prices[i] - avg_prices[i]) * qty
            if trade_pnl > 0:
                omni_pnl += trade_pnl
    return omni_pnl

