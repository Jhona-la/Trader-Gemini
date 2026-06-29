import numpy as np
try:
    import cython
    IS_CYTHON = cython.compiled
except ImportError:
    IS_CYTHON = False

if IS_CYTHON:
    def njit(*args, **kwargs):
        def decorator(func): return func
        if len(args) == 1 and callable(args[0]) and not kwargs: return args[0]
        return decorator
else:
    from numba import njit

@njit(fastmath=True, nogil=True)
def vectorized_backtest_core(
    highs, lows, closes, signals,
    sl_pct, tp_pct, leverage, fee_rate,
    max_hold_bars
):
    """
    Motor Nano-Cuántico en Numba C++.
    Itera arreglos pre-cacheados a velocidad de la luz.
    Retorna PnL % de cada trade, barras de duración, y winrate.
    """
    n = len(closes)
    
    # Pre-allocate output arrays (max trades = n)
    pnl_pcts = np.zeros(n, dtype=np.float32)
    durations = np.zeros(n, dtype=np.int32)
    
    trade_count = 0
    in_position = False
    entry_price = 0.0
    position_side = 0 # 1 = LONG, -1 = SHORT
    bars_held = 0
    
    for i in range(n):
        # 1. Manejo de Posición Abierta
        if in_position:
            bars_held += 1
            current_high = highs[i]
            current_low = lows[i]
            current_close = closes[i]
            
            exit_price = 0.0
            exit_reason = 0 # 0=None, 1=TP, 2=SL, 3=Time
            
            if position_side == 1: # LONG
                tp_price = entry_price * (1.0 + tp_pct)
                sl_price = entry_price * (1.0 - sl_pct)
                
                # Check Stop Loss First (Pessimistic intra-candle)
                if current_low <= sl_price:
                    exit_price = sl_price
                    exit_reason = 2
                elif current_high >= tp_price:
                    exit_price = tp_price
                    exit_reason = 1
                elif bars_held >= max_hold_bars:
                    exit_price = current_close
                    exit_reason = 3
                    
            elif position_side == -1: # SHORT
                tp_price = entry_price * (1.0 - tp_pct)
                sl_price = entry_price * (1.0 + sl_pct)
                
                if current_high >= sl_price:
                    exit_price = sl_price
                    exit_reason = 2
                elif current_low <= tp_price:
                    exit_price = tp_price
                    exit_reason = 1
                elif bars_held >= max_hold_bars:
                    exit_price = current_close
                    exit_reason = 3
            
            # Si hubo salida
            if exit_reason > 0:
                raw_pnl_pct = 0.0
                if position_side == 1:
                    raw_pnl_pct = (exit_price - entry_price) / entry_price
                else:
                    raw_pnl_pct = (entry_price - exit_price) / entry_price
                
                # Aplicar apalancamiento y comisiones (entrada + salida)
                # entry_fee = fee_rate (Maker o Taker)
                # exit_fee = Taker siempre en backtest pesimista
                net_pnl_pct = (raw_pnl_pct * leverage) - (fee_rate * 2 * leverage)
                
                pnl_pcts[trade_count] = net_pnl_pct
                durations[trade_count] = bars_held
                trade_count += 1
                
                in_position = False
                bars_held = 0
                continue # No abrir nueva posición en la misma vela que cerramos
                
        # 2. Apertura de Posición
        if not in_position:
            sig = signals[i]
            if sig == 1:
                in_position = True
                position_side = 1
                entry_price = closes[i] # Asumimos apertura en cierre de vela o apertura siguiente
                bars_held = 0
            elif sig == -1:
                in_position = True
                position_side = -1
                entry_price = closes[i]
                bars_held = 0

    return pnl_pcts[:trade_count], durations[:trade_count]

@njit(fastmath=True, nogil=True)
def calculate_ema(closes, window):
    n = len(closes)
    ema = np.zeros(n, dtype=np.float32)
    alpha = 2.0 / (window + 1.0)
    ema[0] = closes[0]
    for i in range(1, n):
        ema[i] = closes[i] * alpha + ema[i-1] * (1.0 - alpha)
    return ema

@njit(fastmath=True, nogil=True)
def calculate_rsi(closes, window):
    n = len(closes)
    rsi = np.zeros(n, dtype=np.float32)
    if n <= window:
        return rsi
        
    gain = 0.0
    loss = 0.0
    for i in range(1, window):
        diff = closes[i] - closes[i-1]
        if diff > 0:
            gain += diff
        else:
            loss -= diff
    gain /= window
    loss /= window
    
    rsi[window-1] = 100.0 - (100.0 / (1.0 + gain / loss)) if loss != 0 else 100.0
    
    for i in range(window, n):
        diff = closes[i] - closes[i-1]
        if diff > 0:
            gain = (gain * (window - 1) + diff) / window
            loss = (loss * (window - 1)) / window
        else:
            gain = (gain * (window - 1)) / window
            loss = (loss * (window - 1) - diff) / window
            
        rs = gain / loss if loss != 0 else 0
        rsi[i] = 100.0 - (100.0 / (1.0 + rs)) if loss != 0 else 100.0
    return rsi

@njit(fastmath=True, nogil=True)
def vectorized_signals(
    closes, 
    rsi_window, rsi_os, rsi_ob,
    macd_f, macd_s
):
    """
    Calcula señales técnicas en picosegundos.
    """
    n = len(closes)
    signals = np.zeros(n, dtype=np.int8)
    
    rsi = calculate_rsi(closes, rsi_window)
    ema_fast = calculate_ema(closes, macd_f)
    ema_slow = calculate_ema(closes, macd_s)
    
    for i in range(1, n):
        # RSI Crosses y EMA Trend
        # LONG: RSI < Oversold y EMA Fast > EMA Slow
        if rsi[i] < rsi_os and ema_fast[i] > ema_slow[i]:
            signals[i] = 1
        # SHORT: RSI > Overbought y EMA Fast < EMA Slow
        elif rsi[i] > rsi_ob and ema_fast[i] < ema_slow[i]:
            signals[i] = -1
            
    return signals

@njit(fastmath=True, nogil=True)
def combine_signals(
    tech_signals, 
    ml_probas_long, ml_probas_short,
    w_tech, w_ml, master_threshold
):
    n = len(tech_signals)
    final_signals = np.zeros(n, dtype=np.int8)
    
    for i in range(n):
        tech = tech_signals[i]
        ml_l = ml_probas_long[i]
        ml_s = ml_probas_short[i]
        
        score_long = 0.0
        score_short = 0.0
        
        if tech == 1:
            score_long += w_tech
        elif tech == -1:
            score_short += w_tech
            
        if ml_l > 0.5:
            score_long += w_ml * ml_l
            
        if ml_s > 0.5:
            score_short += w_ml * ml_s
            
        if score_long >= master_threshold and score_long > score_short:
            final_signals[i] = 1
        elif score_short >= master_threshold and score_short > score_long:
            final_signals[i] = -1
            
    return final_signals

def simulate_portfolio_vectorized(pnl_pcts, initial_capital=13.0, size_pct=0.3):
    """
    Calcula la curva de equity asumiendo trades secuenciales (para Optuna fitness).
    """
    if len(pnl_pcts) == 0:
        return initial_capital, 0.0, 0, 0, 0.0
        
    capital = initial_capital
    peak = initial_capital
    max_dd = 0.0
    wins = 0
    losses = 0
    
    for pnl in pnl_pcts:
        # Size = 30% del capital actual
        trade_size = capital * size_pct
        pnl_usd = trade_size * pnl
        
        capital += pnl_usd
        
        if pnl_usd > 0:
            wins += 1
        else:
            losses += 1
            
        if capital > peak:
            peak = capital
            
        dd = (peak - capital) / peak
        if dd > max_dd:
            max_dd = dd
            
        if capital <= 0:
            capital = 0
            break
            
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0
    return capital, max_dd, wins, losses, win_rate
