import numpy as np
from numba import njit
import math

@njit(cache=True)
def simulate_nano_portfolio_events_jit(
    entry_ts, exit_ts, pct_ret, is_win, is_long, entry_prices, horizons, symbols,
    initial_capital, min_notional, kelly_fraction, max_concurrent, leverage, round_trip_fee
):
    """
    Simulador O(N) basado en eventos para calcular PnL exacto
    respetando los límites de margen reales de Binance y el sizing fraccional Kelly.
    Todo compilado en C vía Numba para el Algoritmo Genético.
    """
    n_trades = len(entry_ts)
    if n_trades == 0:
        return initial_capital, 0.0, 0.0, 0.0, 0
        
    # Crear un arreglo de eventos. Cada trade tiene 2 eventos: Entry (1) y Exit (0).
    n_events = n_trades * 2
    event_times = np.zeros(n_events, dtype=np.int64)
    event_types = np.zeros(n_events, dtype=np.int32) # 1 = Entry, 0 = Exit
    event_trade_ids = np.zeros(n_events, dtype=np.int32)
    
    for i in range(n_trades):
        event_times[i*2] = entry_ts[i]
        event_types[i*2] = 1
        event_trade_ids[i*2] = i
        
        event_times[i*2+1] = exit_ts[i]
        event_types[i*2+1] = 0
        event_trade_ids[i*2+1] = i
        
    # Sort events by time. If times are equal, Exits (0) should process before Entries (1).
    # Since numpy argsort doesn't natively do multi-column sort in njit easily, we can pack them:
    # packed = time * 10 + event_type  (So type 0 comes before type 1 for the same time)
    packed_events = event_times * 10 + event_types
    order = np.argsort(packed_events)
    
    # State tracking
    current_capital = float(initial_capital)
    peak_capital = float(initial_capital)
    max_drawdown = 0.0
    
    # Active slots tracking
    active_margin = np.zeros(n_trades, dtype=np.float64) # Store how much margin was allocated
    active_count = 0
    active_symbol_flags = np.zeros(100, dtype=np.int32) # Assuming symbols mapped to integers 0-99
    
    total_wins = 0
    executed_trades = 0
    
    for i in range(n_events):
        idx = order[i]
        e_type = event_types[idx]
        t_id = event_trade_ids[idx]
        sym = symbols[t_id]
        
        if e_type == 1: # Entry
            # Rejections
            if active_count >= max_concurrent:
                continue
            if sym < 100 and active_symbol_flags[sym] == 1:
                # Evitar posiciones concurrentes en el mismo símbolo
                continue
                
            # Sizing (Fractional Kelly)
            target_margin = current_capital * kelly_fraction
            notional = target_margin * leverage
            
            # Binance Rule: Minimum Order Size
            if notional < min_notional:
                # Si no nos alcanza el margen ni con todo el capital disponible
                max_possible_notional = current_capital * leverage
                if max_possible_notional < min_notional:
                    # Cuenta quebrada o margin insufficient
                    continue
                else:
                    # Ajustar al mínimo de Binance
                    target_margin = min_notional / leverage
                    
            # Si target_margin > current_capital, solo podemos usar current_capital
            if target_margin > current_capital:
                target_margin = current_capital
                
            # Verificar de nuevo con el ajuste final
            if (target_margin * leverage) < min_notional:
                continue
                
            # EXECUTE ENTRY
            active_margin[t_id] = target_margin
            current_capital -= target_margin
            active_count += 1
            if sym < 100:
                active_symbol_flags[sym] = 1
                
        else: # Exit
            margin_used = active_margin[t_id]
            if margin_used > 0: # Trade was actually executed
                # Calculate absolute PnL based on leveraged return percentage
                trade_ret = pct_ret[t_id]
                # The pct_ret from quantum_engine already includes leverage multiplier
                # e.g., pct_ret = ((exit - entry)/entry * lev) - fees
                # Wait, if pct_ret is already leveraged, absolute PnL is (margin_used * trade_ret).
                # Example: Margin $2. Lev 10x. Notional $20. Price moves +1%.
                # Notional profit = $0.20.
                # If pct_ret is given as leveraged return: 1% * 10x = 10%. 
                # PnL = Margin * 10% = 2 * 0.10 = $0.20. Correct.
                
                pnl = margin_used * trade_ret
                current_capital += (margin_used + pnl)
                
                # Update stats
                executed_trades += 1
                if is_win[t_id] == 1:
                    total_wins += 1
                    
                if current_capital > peak_capital:
                    peak_capital = current_capital
                
                dd = (peak_capital - current_capital) / peak_capital if peak_capital > 0 else 0
                if dd > max_drawdown:
                    max_drawdown = dd
                    
                # Free slot
                active_count -= 1
                active_margin[t_id] = 0.0
                if sym < 100:
                    active_symbol_flags[sym] = 0
                    
    global_pnl = current_capital - initial_capital
    win_rate = (total_wins / executed_trades * 100.0) if executed_trades > 0 else 0.0
    
    return current_capital, global_pnl, max_drawdown, win_rate, executed_trades

@njit(fastmath=True, cache=True)
def evaluate_sl_tp_trailing_jit(
    current_price: float, 
    entry_price: float, 
    hwm: float, 
    lwm: float, 
    qty: float, 
    sl_pct: float, 
    tp_pct: float, 
    atr_pct: float, 
    is_zombie_chaser: bool,
    elastic_tp_expansion: bool,
    trailing_atr_mult: float
) -> int:
    """
    [NANO-SPEED] Core SL/TP and Trailing Stop evaluation.
    Returns:
        0: KEEP_OPEN
        1: EXIT_HARD_SL
        2: EXIT_HARD_TP
        3: EXIT_TRAILING_STOP
        4: EXIT_TURBO_BREAKEVEN
        5: EXIT_ZOMBIE_CHASER
    """
    if qty == 0:
        return 0
        
    is_long = qty > 0
    pnl_pct = ((current_price - entry_price) / entry_price) if is_long else ((entry_price - current_price) / entry_price)
    
    # 1. HARD SL
    if pnl_pct <= -sl_pct:
        return 1
        
    # 2. HARD TP
    # If elastic TP expansion is active, we push TP up by 50%
    eff_tp = tp_pct * 1.5 if elastic_tp_expansion else tp_pct
    if pnl_pct >= eff_tp:
        return 2
        
    # 3. ZOMBIE CHASER TRAILING
    if is_zombie_chaser:
        trailing_dist = (atr_pct * 0.5)
        if is_long:
            trail_stop = hwm * (1.0 - trailing_dist)
            if current_price < trail_stop and pnl_pct > 0.0008:
                return 5
        else:
            trail_stop = lwm * (1.0 + trailing_dist)
            if current_price > trail_stop and pnl_pct > 0.0008:
                return 5
                
    # 4. STANDARD TRAILING STOP & TURBO BREAKEVEN
    # We trigger Turbo-Breakeven if we cross 1x ATR profit
    if pnl_pct > atr_pct:
        # Distance dynamically adjusted by horizon (via trailing_atr_mult)
        trail_dist = atr_pct * trailing_atr_mult
        if is_long:
            trail_stop = hwm * (1.0 - trail_dist)
            if current_price < trail_stop and pnl_pct > (atr_pct * 0.4): # Buffer for Maker fees
                return 3
        else:
            trail_stop = lwm * (1.0 + trail_dist)
            if current_price > trail_stop and pnl_pct > (atr_pct * 0.4):
                return 3
                
    return 0

