import numpy as np
from numba import njit
import logging

logger = logging.getLogger("VectorizedEngine")

@njit(nogil=True, cache=True)
def run_vectorized_simulation(
    open_prices, high_prices, low_prices, close_prices,
    rsi_arr, atr_arr,
    # Parameters to optimize
    sl_pct, tp_pct, rsi_oversold, rsi_overbought,
    ml_kelly_fraction, compounding_growth_factor,
    initial_capital=13.0
):
    """
    [PHASE 16] C-Level Mathematical Replica of Trader Gemini Engine
    Executes in nanoseconds using LLVM machine code.
    Returns: (final_equity, total_trades, win_rate, max_drawdown)
    """
    n = len(close_prices)
    
    # State tracking
    equity = initial_capital
    max_equity = initial_capital
    max_dd = 0.0
    
    in_position = False
    entry_price = 0.0
    position_qty = 0.0
    direction = 0  # 1 for LONG, -1 for SHORT
    
    # Kelly & Compounding state
    current_leverage = 20.0
    base_risk = 0.02
    
    # Stats
    wins = 0
    losses = 0
    total_trades = 0
    
    for i in range(1, n):
        # 1. Update existing position (Take Profit / Stop Loss)
        if in_position:
            current_price = close_prices[i]
            high = high_prices[i]
            low = low_prices[i]
            
            # Simple TP/SL evaluation
            exit_trade = False
            exit_price = 0.0
            
            if direction == 1:
                # Check SL
                if low <= entry_price * (1.0 - sl_pct):
                    exit_trade = True
                    exit_price = entry_price * (1.0 - sl_pct)
                # Check TP
                elif high >= entry_price * (1.0 + tp_pct):
                    exit_trade = True
                    exit_price = entry_price * (1.0 + tp_pct)
            else:
                # Check SL
                if high >= entry_price * (1.0 + sl_pct):
                    exit_trade = True
                    exit_price = entry_price * (1.0 + sl_pct)
                # Check TP
                elif low <= entry_price * (1.0 - tp_pct):
                    exit_trade = True
                    exit_price = entry_price * (1.0 - tp_pct)
                    
            if exit_trade:
                # Calculate PnL
                pnl_pct = (exit_price - entry_price) / entry_price * direction
                trade_pnl = (position_qty * entry_price) * pnl_pct
                
                # Apply simulated fees (Taker)
                fee = (position_qty * exit_price) * 0.000375
                net_pnl = trade_pnl - fee
                
                equity += net_pnl
                
                if net_pnl > 0:
                    wins += 1
                else:
                    losses += 1
                    
                total_trades += 1
                in_position = False
                
                if equity > max_equity:
                    max_equity = equity
                else:
                    dd = (max_equity - equity) / max_equity
                    if dd > max_dd:
                        max_dd = dd
                        
                # Check bankruptcy
                if equity < 5.0:
                    return equity, total_trades, (wins / max(1, total_trades)) * 100, max_dd

        # 2. Look for entries if not in position
        if not in_position:
            # Replicating a basic Scalping RSI strategy for vectorization test
            # If RSI crosses below Oversold -> LONG
            # If RSI crosses above Overbought -> SHORT
            signal = 0
            if rsi_arr[i] < rsi_oversold and rsi_arr[i-1] >= rsi_oversold:
                signal = 1
            elif rsi_arr[i] > rsi_overbought and rsi_arr[i-1] <= rsi_overbought:
                signal = -1
                
            if signal != 0:
                # Meritocratic Sizing Math (Replica of risk_manager.py)
                house_money = max(0.0, equity - initial_capital)
                hm_ratio = house_money / equity if equity > 0 else 0
                asymmetric_mult = 1.0 + (hm_ratio * compounding_growth_factor)
                
                # Apply Kelly Fraction
                eff_leverage = current_leverage * ml_kelly_fraction
                
                target_notional = equity * base_risk * eff_leverage * asymmetric_mult
                
                # Binance Micro limit cap
                if target_notional < 5.05:
                    target_notional = 5.05
                    
                # Margin check
                required_margin = target_notional / eff_leverage
                if required_margin > equity * 0.95:
                    target_notional = (equity * 0.95) * eff_leverage
                
                if target_notional >= 5.05:
                    entry_price = close_prices[i]
                    position_qty = target_notional / entry_price
                    direction = signal
                    in_position = True
                    
                    # Apply Maker fee for entry
                    fee = target_notional * 0.0002
                    equity -= fee

    # Close any open positions at the end of the simulation
    if in_position:
        exit_price = close_prices[n-1]
        pnl_pct = (exit_price - entry_price) / entry_price * direction
        trade_pnl = (position_qty * entry_price) * pnl_pct
        fee = (position_qty * exit_price) * 0.000375
        equity += (trade_pnl - fee)

    win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
    return equity, total_trades, win_rate, max_dd

def create_feature_matrices(df):
    """
    Precomputes vectors efficiently using pandas/numpy.
    """
    import talib
    
    # 1. Price arrays
    close_arr = df['close'].values.astype(np.float64)
    high_arr = df['high'].values.astype(np.float64)
    low_arr = df['low'].values.astype(np.float64)
    open_arr = df['open'].values.astype(np.float64)
    
    # 2. Technical arrays
    rsi_arr = talib.RSI(close_arr, timeperiod=14)
    atr_arr = talib.ATR(high_arr, low_arr, close_arr, timeperiod=14)
    
    # Fill NaNs
    rsi_arr = np.nan_to_num(rsi_arr, nan=50.0)
    atr_arr = np.nan_to_num(atr_arr, nan=0.0)
    
    return open_arr, high_arr, low_arr, close_arr, rsi_arr, atr_arr
