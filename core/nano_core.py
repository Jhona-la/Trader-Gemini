import numpy as np
from numba import njit

@njit(fastmath=True, nogil=True)
def calculate_kelly_fraction(
    win_streak: int, 
    loss_streak: int, 
    winrate: float, 
    payoff_ratio: float, 
    max_kelly: float = 0.5,
    stress_score: float = 100.0,
    apply_mult: bool = True
) -> float:
    """
    Calculates the Kelly Criterion fraction in nanoseconds.
    Merged FASE 8 (anti-martingale) and FASE 11 (stress multiplier).
    """
    if winrate <= 0.0 or payoff_ratio <= 0.0:
        return 0.01  # Base minimum risk
        
    # Standard Kelly formula: (p * b - q) / b
    q = 1.0 - winrate
    kelly_pct = (winrate * payoff_ratio - q) / payoff_ratio
    
    if not apply_mult:
        if kelly_pct < 0.0: return 0.01
        return min(kelly_pct, max_kelly)
    
    # Defensive Scaling from math_kernel
    if stress_score < 90.0:
        kelly_pct *= 0.125  # Eighth-Kelly under extreme stress
        
    # Anti-Martingale / Streak Logic
    if win_streak > 0:
        multiplier = 1.0 + (min(win_streak, 5) * 0.1)  # Up to 50% boost on win streaks
        kelly_pct *= multiplier
    elif loss_streak > 0:
        divisor = 1.0 + (min(loss_streak, 5) * 0.2)    # Cut risk significantly on losing streaks
        kelly_pct /= divisor
        
    if kelly_pct <= 0.0:
        return 0.01  # Minimum risk floor
        
    return min(kelly_pct, max_kelly)

@njit(fastmath=True, nogil=True)
def calculate_unrealized_pnl_fast(current_price: float, entry_price: float, quantity: float, direction: int) -> float:
    """
    Scalar calculation of PnL for a single position.
    Matches the Cython (.pyx) signature used by portfolio.py.
    """
    if quantity <= 0.0:
        return 0.0
        
    if direction == 1:
        price_diff = current_price - entry_price
    else:
        price_diff = entry_price - current_price
        
    return (price_diff / entry_price) * quantity * entry_price

@njit(fastmath=True, nogil=True)
def calculate_unrealized_pnl_batch(entry_prices, current_prices, quantities, directions):
    """
    Vectorized calculation of PnL for N positions simultaneously.
    """
    n = len(entry_prices)
    pnls = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        pnls[i] = calculate_unrealized_pnl_fast(
            current_prices[i], entry_prices[i], quantities[i], directions[i]
        )
        
    return pnls

@njit(fastmath=True, nogil=True)
def update_hwm_lwm(price: float, hwm: float, lwm: float):
    """
    Fast calculation for high/low water marks.
    Returns (new_hwm, new_lwm)
    """
    new_hwm = hwm
    new_lwm = lwm
    
    if price > hwm:
        new_hwm = price
    if lwm == 0.0 or price < lwm:
        new_lwm = price
        
    return new_hwm, new_lwm

@njit(fastmath=True, nogil=True)
def check_stops_nano(
    low_price: float, 
    high_price: float, 
    close_price: float, 
    entry_price: float, 
    direction: int, 
    sl_pct: float, 
    tp_pct: float,
    trailing_activation_pct: float,
    highest_price: float,
    lowest_price: float,
    is_scalping: bool
) -> tuple:
    """
    Evaluates stop loss and take profit for a single epoch wick/bar in nanoseconds.
    Returns: (should_exit, exit_reason)
    exit_reason: 0 (None), 1 (Stop Loss), 2 (Take Profit), 3 (Trailing Stop)
    """
    should_exit = False
    exit_reason = 0
    
    if direction == 1: # LONG
        # Update HWM
        if high_price > highest_price:
            highest_price = high_price
            
        # Check Stop Loss (wick touched low)
        sl_price = entry_price * (1.0 - sl_pct)
        if low_price <= sl_price:
            return True, 1, highest_price, lowest_price
            
        # Check Take Profit
        tp_price = entry_price * (1.0 + tp_pct)
        if high_price >= tp_price and not is_scalping: # Scalping uses dynamic exits, fixed TP is fallback
            return True, 2, highest_price, lowest_price
            
        # Check Trailing Stop
        if highest_price >= entry_price * (1.0 + trailing_activation_pct):
            # Dynamic trailing calculation based on distance from entry
            profit_pct = (highest_price - entry_price) / entry_price
            trail_distance = sl_pct * 0.5 # Default tight trail
            
            # Phase 3 logic: tighten trail as profit increases
            if profit_pct > tp_pct * 0.8:
                trail_distance = sl_pct * 0.1
            elif profit_pct > tp_pct * 0.5:
                trail_distance = sl_pct * 0.2
                
            dynamic_sl = highest_price * (1.0 - trail_distance)
            if low_price <= dynamic_sl:
                return True, 3, highest_price, lowest_price
                
    elif direction == -1: # SHORT
        # Update LWM
        if low_price < lowest_price:
            lowest_price = low_price
            
        # Check Stop Loss (wick touched high)
        sl_price = entry_price * (1.0 + sl_pct)
        if high_price >= sl_price:
            return True, 1, highest_price, lowest_price
            
        # Check Take Profit
        tp_price = entry_price * (1.0 - tp_pct)
        if low_price <= tp_price and not is_scalping:
            return True, 2, highest_price, lowest_price
            
        # Check Trailing Stop
        if lowest_price <= entry_price * (1.0 - trailing_activation_pct):
            profit_pct = (entry_price - lowest_price) / entry_price
            trail_distance = sl_pct * 0.5
            
            if profit_pct > tp_pct * 0.8:
                trail_distance = sl_pct * 0.1
            elif profit_pct > tp_pct * 0.5:
                trail_distance = sl_pct * 0.2
                
            dynamic_sl = lowest_price * (1.0 + trail_distance)
            if high_price >= dynamic_sl:
                return True, 3, highest_price, lowest_price
                
    return False, 0, highest_price, lowest_price
