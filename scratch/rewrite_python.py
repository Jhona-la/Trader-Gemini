import os

filepath = 'utils/math_kernel.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace calculate_ema_jit
old_ema = '''@njit(fastmath=True, cache=True)
def calculate_ema_jit(prices, period):
    n = len(prices)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    
    ema[0] = prices[0]
    multiplier = 2.0 / (period + 1.0)
    
    for i in range(1, n):
        ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
        
    return ema'''

new_ema = '''@njit(fastmath=True, cache=True)
def calculate_ema_jit(prices, period):
    """[NANO-SPEED] Rust FFI Reemplazo de EMA."""
    n = len(prices)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0: return ema
    
    if _rust_lib:
        arr_in = np.asarray(prices, dtype=np.float64)
        _rust_lib.ffi_compute_ema(
            arr_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
            period,
            ema.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        return ema
        
    ema[0] = prices[0]
    multiplier = 2.0 / (period + 1.0)
    for i in range(1, n):
        ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema'''

content = content.replace(old_ema, new_ema)

# Replace calculate_rsi_jit
old_rsi = '''@njit(fastmath=True, cache=True)
def calculate_rsi_jit(prices, period=14):
    n = len(prices)
    rsi = np.zeros(n, dtype=np.float64)
    if n < period:
        # Prevent zero-division or uninitialized states
        for i in range(n):
            rsi[i] = 50.0
        return rsi
        
    gain = 0.0
    loss = 0.0
    
    # Calculate initial gain and loss
    for i in range(1, period):
        diff = prices[i] - prices[i-1]
        if diff > 0:
            gain += diff
        else:
            loss -= diff
            
    gain /= period
    loss /= period
    
    for i in range(period):
        rsi[i] = 50.0  # Fill initial period
        
    if loss == 0:
        rsi[period-1] = 100.0
    else:
        rs = gain / loss
        rsi[period-1] = 100.0 - (100.0 / (1.0 + rs))
        
    # Calculate rest using Smoothed Moving Average
    for i in range(period, n):
        diff = prices[i] - prices[i-1]
        if diff > 0:
            gain = (gain * (period - 1) + diff) / period
            loss = (loss * (period - 1)) / period
        else:
            gain = (gain * (period - 1)) / period
            loss = (loss * (period - 1) - diff) / period
            
        if loss == 0:
            rsi[i] = 100.0
        else:
            rs = gain / loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
            
    return rsi'''

new_rsi = '''@njit(fastmath=True, cache=True)
def calculate_rsi_jit(prices, period=14):
    """[NANO-SPEED] Rust FFI Reemplazo de RSI."""
    n = len(prices)
    rsi = np.zeros(n, dtype=np.float64)
    if n < period:
        for i in range(n): rsi[i] = 50.0
        return rsi
        
    if _rust_lib:
        arr_in = np.asarray(prices, dtype=np.float64)
        _rust_lib.ffi_compute_rsi(
            arr_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
            period,
            rsi.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        return rsi

    gain = 0.0
    loss = 0.0
    for i in range(1, period):
        diff = prices[i] - prices[i-1]
        if diff > 0: gain += diff
        else: loss -= diff
    gain /= period
    loss /= period
    for i in range(period): rsi[i] = 50.0
    if loss == 0: rsi[period-1] = 100.0
    else: rsi[period-1] = 100.0 - (100.0 / (1.0 + gain / loss))
    for i in range(period, n):
        diff = prices[i] - prices[i-1]
        if diff > 0:
            gain = (gain * (period - 1) + diff) / period
            loss = (loss * (period - 1)) / period
        else:
            gain = (gain * (period - 1)) / period
            loss = (loss * (period - 1) - diff) / period
        if loss == 0: rsi[i] = 100.0
        else: rsi[i] = 100.0 - (100.0 / (1.0 + gain / loss))
    return rsi'''

content = content.replace(old_rsi, new_rsi)

# Replace calculate_bollinger_robust_jit
old_bb = '''@njit(fastmath=True, cache=True)
def calculate_bollinger_robust_jit(prices, period=20, std_dev=2.0, threshold_ratio=3.0, iterations=30):
    """
    Bollinger Bands usando Volatilidad Robusta (RANSAC-inspired).
    Fase 3: Protección contra flash-crashes.
    """
    n = len(prices)
    upper = np.zeros(n, dtype=np.float64)
    lower = np.zeros(n, dtype=np.float64)
    middle = np.zeros(n, dtype=np.float64)
    
    if n < period:
        # Fallback para series cortas
        for i in range(n):
            middle[i] = prices[i]
            upper[i] = prices[i]
            lower[i] = prices[i]
        return upper, middle, lower
        
    for i in range(period-1):
        middle[i] = prices[i]
        upper[i] = prices[i]
        lower[i] = prices[i]
        
    # Calcular RANSAC Volatility (Rolling)
    for i in range(period-1, n):
        window = prices[i-period+1:i+1]
        
        # Mean
        mean_val = 0.0
        for j in range(period):
            mean_val += window[j]
        mean_val /= period
        
        # Standard Dev
        var_val = 0.0
        for j in range(period):
            var_val += (window[j] - mean_val)**2
        std_val = math.sqrt(var_val / period)
        
        # Apply bands
        middle[i] = mean_val
        upper[i] = mean_val + std_dev * std_val
        lower[i] = mean_val - std_dev * std_val
        
    return upper, middle, lower'''

new_bb = '''@njit(fastmath=True, cache=True)
def calculate_bollinger_robust_jit(prices, period=20, std_dev=2.0, threshold_ratio=3.0, iterations=30):
    """[NANO-SPEED] Rust FFI Reemplazo de Bollinger."""
    n = len(prices)
    upper = np.zeros(n, dtype=np.float64)
    lower = np.zeros(n, dtype=np.float64)
    middle = np.zeros(n, dtype=np.float64)
    
    if n < period:
        for i in range(n):
            middle[i] = prices[i]; upper[i] = prices[i]; lower[i] = prices[i]
        return upper, middle, lower
        
    if _rust_lib:
        arr_in = np.asarray(prices, dtype=np.float64)
        _rust_lib.ffi_compute_bbands(
            arr_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
            period, float(std_dev),
            upper.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            middle.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            lower.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        return upper, middle, lower

    for i in range(period-1):
        middle[i] = prices[i]; upper[i] = prices[i]; lower[i] = prices[i]
        
    for i in range(period-1, n):
        window = prices[i-period+1:i+1]
        mean_val = 0.0
        for j in range(period): mean_val += window[j]
        mean_val /= period
        var_val = 0.0
        for j in range(period): var_val += (window[j] - mean_val)**2
        std_val = math.sqrt(var_val / period)
        middle[i] = mean_val
        upper[i] = mean_val + std_dev * std_val
        lower[i] = mean_val - std_dev * std_val
    return upper, middle, lower'''

content = content.replace(old_bb, new_bb)

# Replace calculate_macd_jit
old_macd = '''@njit(fastmath=True, cache=True)
def calculate_macd_jit(prices, fast_period=12, slow_period=26, signal_period=9):
    n = len(prices)
    macd = np.zeros(n, dtype=np.float64)
    signal = np.zeros(n, dtype=np.float64)
    hist = np.zeros(n, dtype=np.float64)
    
    if n == 0:
        return macd, signal, hist
        
    fast_ema = calculate_ema_jit(prices, fast_period)
    slow_ema = calculate_ema_jit(prices, slow_period)
    
    for i in range(n):
        macd[i] = fast_ema[i] - slow_ema[i]
        
    signal = calculate_ema_jit(macd, signal_period)
    
    for i in range(n):
        hist[i] = macd[i] - signal[i]
        
    return macd, signal, hist'''

new_macd = '''@njit(fastmath=True, cache=True)
def calculate_macd_jit(prices, fast_period=12, slow_period=26, signal_period=9):
    """[NANO-SPEED] Rust FFI Reemplazo de MACD."""
    n = len(prices)
    macd = np.zeros(n, dtype=np.float64)
    signal = np.zeros(n, dtype=np.float64)
    hist = np.zeros(n, dtype=np.float64)
    if n == 0: return macd, signal, hist
    
    if _rust_lib:
        arr_in = np.asarray(prices, dtype=np.float64)
        _rust_lib.ffi_compute_macd(
            arr_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
            fast_period, slow_period, signal_period,
            macd.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            signal.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            hist.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        return macd, signal, hist

    fast_ema = calculate_ema_jit(prices, fast_period)
    slow_ema = calculate_ema_jit(prices, slow_period)
    for i in range(n): macd[i] = fast_ema[i] - slow_ema[i]
    signal = calculate_ema_jit(macd, signal_period)
    for i in range(n): hist[i] = macd[i] - signal[i]
    return macd, signal, hist'''

content = content.replace(old_macd, new_macd)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
