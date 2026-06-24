import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def quantum_grid_search_core(
    close_prices, 
    high_prices, 
    low_prices, 
    ml_scores, 
    tech_scores, 
    vol_ratios,
    params_grid  # Matrix of shape (N_combinations, M_params)
):
    """
    Motor NANO-SPEED escrito en C (vía Numba LLVM) para evaluar 
    millones de combinaciones de hiperparámetros en milisegundos.
    
    params_grid layout:
    0: ml_threshold
    1: tech_threshold
    2: vol_ratio_threshold
    3: stop_loss_pct
    4: take_profit_pct
    5: trailing_activation_pct
    6: trailing_distance_pct
    7: max_hold_bars
    8: strat_type (0: Mean Reversion, 1: Breakout)
    """
    n_candles = len(close_prices)
    n_combos = params_grid.shape[0]
    
    # Resultados: [Net PnL, WinRate, Total Trades, Max Drawdown] por combinación
    results = np.zeros((n_combos, 4), dtype=np.float64)
    
    # Comisiones Binance Taker + Slippage
    fee_rate = 0.000375 * 2.0  # Entry + Exit
    slippage = 0.0001
    
    for c in prange(n_combos):
        ml_thresh = params_grid[c, 0]
        tech_thresh = params_grid[c, 1]
        vol_thresh = params_grid[c, 2]
        sl_pct = params_grid[c, 3]
        tp_pct = params_grid[c, 4]
        trail_act = params_grid[c, 5]
        trail_dist = params_grid[c, 6]
        max_hold = int(params_grid[c, 7])
        strat_type = int(params_grid[c, 8])
        
        position = 0 # 0=flat, 1=long, -1=short
        entry_price = 0.0
        bars_held = 0
        highest_pnl = 0.0
        
        wins = 0
        losses = 0
        pnl = 0.0
        peak_pnl = 1.0 # Starting capital unit
        max_dd = 0.0
        
        capital = 1.0
        risk_fraction = 0.50
        leverage = 50.0
        for i in range(1, n_candles - 1):
            if position != 0:
                bars_held += 1
                
                # Check Exits (Stop Loss / Take Profit / Trailing / Timeout)
                current_pnl = 0.0
                if position == 1:
                    current_pnl = (close_prices[i] - entry_price) / entry_price
                else:
                    current_pnl = (entry_price - close_prices[i]) / entry_price
                
                # Update Trailing logic
                if current_pnl > highest_pnl:
                    highest_pnl = current_pnl
                    
                exit_triggered = False
                
                # 1. Hard Take Profit
                if current_pnl >= tp_pct:
                    exit_triggered = True
                # 2. Hard Stop Loss
                elif current_pnl <= -sl_pct:
                    exit_triggered = True
                # 3. Trailing Stop Loss
                elif highest_pnl >= trail_act and current_pnl <= (highest_pnl - trail_dist):
                    exit_triggered = True
                # 4. Timeout Timeout
                elif bars_held >= max_hold:
                    exit_triggered = True
                    
                if exit_triggered:
                    net_trade = current_pnl - fee_rate - slippage
                    
                    if capital > 0.0:
                        trade_roi = net_trade * leverage
                        capital_at_risk = capital * risk_fraction
                        capital += (capital_at_risk * trade_roi)
                    
                    pnl += net_trade
                    
                    if capital > peak_pnl:
                        peak_pnl = capital
                    else:
                        dd = (peak_pnl - capital) / peak_pnl
                        if dd > max_dd:
                            max_dd = dd
                            
                    if net_trade > 0:
                        wins += 1
                    else:
                        losses += 1
                        
                    position = 0
                    bars_held = 0
                    highest_pnl = 0.0
            
            if position == 0:
                if strat_type == 0:
                    # MEAN REVERSION
                    if ml_scores[i] < (1.0 - ml_thresh) and tech_scores[i] < (1.0 - tech_thresh):
                        position = 1
                        entry_price = close_prices[i]
                        bars_held = 0
                        highest_pnl = 0.0
                    elif ml_scores[i] > ml_thresh and tech_scores[i] > tech_thresh:
                        position = -1
                        entry_price = close_prices[i]
                        bars_held = 0
                        highest_pnl = 0.0
                else:
                    # BREAKOUT
                    if ml_scores[i] > ml_thresh and tech_scores[i] > tech_thresh:
                        position = 1
                        entry_price = close_prices[i]
                        bars_held = 0
                        highest_pnl = 0.0
                    elif ml_scores[i] < (1.0 - ml_thresh) and tech_scores[i] < (1.0 - tech_thresh):
                        position = -1
                        entry_price = close_prices[i]
                        bars_held = 0
                        highest_pnl = 0.0
                        
        total_trades = wins + losses
        win_rate = (wins / total_trades) if total_trades > 0 else 0.0
        
        # PENALIZACIÓN DE CERTEZA ABSOLUTA (WIN RATE 100%)
        # Si la estrategia registra UNA SOLA PÉRDIDA o capital cae a cero, el score es -999.0
        if losses > 0 or total_trades < 5 or capital <= 0.0:
            compound_pnl = -999.0
        else:
            compound_pnl = capital - 1.0
            
        results[c, 0] = compound_pnl # Guardar Compound PnL para ranking
        results[c, 1] = win_rate
        results[c, 2] = float(total_trades)
        results[c, 3] = max_dd
        
    return results
