import numpy as np
from numba import njit, prange

@njit(cache=True)
def technical_simulation_loop_njit(
    closes: np.ndarray,
    tp_pct: float,
    sl_pct: float,
    window: int,
    fast_window: int,
    trend_conf_threshold: float,
    actual_start: int,
    actual_end: int
) -> np.ndarray:
    """
    Simulación HFT Bare Metal vectorizada con Numba.
    Ignora el Garbage Collector de Python. No se instancian objetos.
    Retorna un array pre-alocado de trades: [pnl, duration, is_win].
    """
    # Max posibles trades (teóricamente, uno cada dos barras)
    max_trades = (actual_end - actual_start) // 2 + 1
    trades = np.empty((max_trades, 3), dtype=np.float64)
    trade_count = 0
    
    position = 0  # 0: NONE, 1: LONG, -1: SHORT
    entry_price = 0.0
    entry_idx = 0
    
    # Pre-calcular el divisor de confirmación para evitar división por 1000 iterada
    trend_scalar_long = 1.0 + (trend_conf_threshold / 1000.0)
    trend_scalar_short = 1.0 - (trend_conf_threshold / 1000.0)

    # Optimizador de medias móviles (Suma deslizante en $O(1)$)
    # Inicialización del rolling sum
    if actual_start <= window:
        actual_start = window + 1

    sum_fast = np.sum(closes[actual_start - fast_window:actual_start])
    sum_slow = np.sum(closes[actual_start - window:actual_start])
    
    for i in range(actual_start, actual_end):
        current_close = closes[i]
        
        # Mantenimiento de sumas deslizantes en O(1)
        # Se resta el elemento saliente y se suma el entrante
        out_fast = closes[i - fast_window]
        out_slow = closes[i - window]
        
        sum_fast = sum_fast - out_fast + current_close
        sum_slow = sum_slow - out_slow + current_close
        
        fast_sma = sum_fast / fast_window
        slow_sma = sum_slow / window

        if position == 0:
            # ENTRY LOGIC
            if fast_sma > slow_sma * trend_scalar_long:
                position = 1
                entry_price = current_close
                entry_idx = i
            elif fast_sma < slow_sma * trend_scalar_short:
                position = -1
                entry_price = current_close
                entry_idx = i
        else:
            # EXIT LOGIC
            pnl = (current_close - entry_price) / entry_price if position == 1 else (entry_price - current_close) / entry_price
            
            # 1. Hard Stops (SL/TP)
            if pnl <= -sl_pct or pnl >= tp_pct:
                if trade_count < max_trades:
                    trades[trade_count, 0] = pnl
                    trades[trade_count, 1] = float((i - entry_idx) * 60) # 60 seconds per bar approx
                    trades[trade_count, 2] = 1.0 if pnl > 0 else 0.0
                    trade_count += 1
                position = 0
                entry_idx = 0
                continue
                
            # 2. Technical Reversal
            if position == 1 and fast_sma < slow_sma:
                if trade_count < max_trades:
                    trades[trade_count, 0] = pnl
                    trades[trade_count, 1] = float((i - entry_idx) * 60)
                    trades[trade_count, 2] = 1.0 if pnl > 0 else 0.0
                    trade_count += 1
                position = 0
                entry_idx = 0
            elif position == -1 and fast_sma > slow_sma:
                if trade_count < max_trades:
                    trades[trade_count, 0] = pnl
                    trades[trade_count, 1] = float((i - entry_idx) * 60)
                    trades[trade_count, 2] = 1.0 if pnl > 0 else 0.0
                    trade_count += 1
                position = 0
                entry_idx = 0

    return trades[:trade_count]

@njit(cache=True, fastmath=True)
def extract_features_njit(closes: np.ndarray, window: int) -> np.ndarray:
    """
    Pre-computa el tensor de características 25D para TODA la matriz de precios.
    En lugar de extraer fila a fila en un bucle Python, Numba vectoriza
    la extracción creando un array [N_velas, 25].
    """
    n = len(closes)
    # 25D Tensor: [Returns (10), SMA Diffs (5), Volatilidad (5), Momentum (5)]
    features = np.zeros((n, 25), dtype=np.float64)
    
    # Rellenar con aproximaciones para la red neuronal
    for i in range(window, n):
        # Base de precios local
        base_price = closes[i-1] if closes[i-1] != 0 else 1.0
        
        # 10 Returns
        for j in range(10):
            if i - j - 1 >= 0:
                features[i, j] = (closes[i-j] - closes[i-j-1]) / base_price
                
        # 5 SMA Diffs (Simple moving averages de 3, 5, 8, 13, 21 periodos)
        periods = (3, 5, 8, 13, 21)
        for idx, p in enumerate(periods):
            if i - p >= 0:
                sma = np.mean(closes[i-p:i])
                features[i, 10 + idx] = (closes[i] - sma) / sma
                
        # 5 Volatility (Std Dev)
        for idx, p in enumerate(periods):
            if i - p >= 0:
                features[i, 15 + idx] = np.std(closes[i-p:i]) / base_price
                
        # 5 Momentum (ROC)
        for idx, p in enumerate(periods):
            if i - p >= 0:
                features[i, 20 + idx] = (closes[i] - closes[i-p]) / base_price
                
    return features

@njit(cache=True, fastmath=True)
def neural_feedforward_njit(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Cómputo matricial directo en Numba. Features (N, 25) @ Weights (25, 4) = Logits (N, 4).
    Luego aplica Softmax estabilizado.
    """
    n = features.shape[0]
    out_dim = weights.shape[1]
    probs = np.zeros((n, out_dim), dtype=np.float64)
    
    for i in range(n):
        # 1. Logits = Input @ Weights
        logits = np.zeros(out_dim, dtype=np.float64)
        for j in range(out_dim):
            sum_val = 0.0
            for k in range(features.shape[1]):
                sum_val += features[i, k] * weights[k, j]
            logits[j] = sum_val
            
        # 2. Stable Softmax
        max_logit = np.max(logits)
        exp_logits = np.exp(logits - max_logit)
        probs[i, :] = exp_logits / np.sum(exp_logits)
        
    return probs

@njit(cache=True, fastmath=True)
def execution_loop_njit(probs: np.ndarray, closes: np.ndarray, sl_pct: float, tp_pct: float, start_idx: int, end_idx: int) -> np.ndarray:
    """
    Sustituye el "Neural Execution Loop" de Python.
    Ejecuta el Bucle de trading sobre arrays estáticos Numba con 0 asignaciones.
    probs[:, 0] = FLAT
    probs[:, 1] = LONG
    probs[:, 2] = SHORT
    probs[:, 3] = CLOSE
    """
    max_trades = (end_idx - start_idx) // 2 + 1
    trades = np.empty((max_trades, 3), dtype=np.float64)
    trade_count = 0
    
    position = 0 # 0=FLAT, 1=LONG, -1=SHORT
    entry_price = 0.0
    entry_idx = 0
    
    for i in range(start_idx, end_idx):
        current_close = closes[i]
        
        # Action Decoder (Argmax for hard threshold)
        action_idx = np.argmax(probs[i])
        conf = probs[i, action_idx]
        
        if position == 0:
            if conf > 0.5:
                if action_idx == 1: # LONG
                    position = 1
                    entry_price = current_close
                    entry_idx = i
                elif action_idx == 2: # SHORT
                    position = -1
                    entry_price = current_close
                    entry_idx = i
        else:
            # EXIT LOGIC
            pnl = (current_close - entry_price)/entry_price if position == 1 else (entry_price - current_close)/entry_price
            
            # Hard Risk
            if pnl <= -sl_pct or pnl >= tp_pct:
                if trade_count < max_trades:
                    trades[trade_count, 0] = pnl
                    trades[trade_count, 1] = float((i - entry_idx)*60)
                    trades[trade_count, 2] = 1.0 if pnl > 0 else 0.0
                    trade_count += 1
                position = 0
                entry_idx = 0
                continue
                
            # Neural Exit
            is_exit = False
            if action_idx == 3: # CLOSE
                is_exit = True
            elif position == 1 and action_idx == 2: # LONG reversal to SHORT
                is_exit = True
            elif position == -1 and action_idx == 1: # SHORT reversal to LONG
                is_exit = True
                
            if is_exit:
                if trade_count < max_trades:
                    trades[trade_count, 0] = pnl
                    trades[trade_count, 1] = float((i - entry_idx)*60)
                    trades[trade_count, 2] = 1.0 if pnl > 0 else 0.0
                    trade_count += 1
                position = 0
                entry_idx = 0
                
    return trades[:trade_count]
