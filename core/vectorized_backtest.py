import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def vectorized_backtest_core(close_prices, rsi_period, rsi_lower, rsi_upper, stop_loss_pct, take_profit_pct):
    """
    Motor hiper-rápido de backtest escrito en C/LLVM via Numba JIT.
    Capaz de evaluar millones de velas en milisegundos.
    """
    n = len(close_prices)
    returns = np.zeros(n)
    
    # Pre-calcular RSI de forma vectorizada (aproximación rápida para JIT)
    rsi = np.zeros(n)
    gains = np.zeros(n)
    losses = np.zeros(n)
    
    for i in prange(1, n):
        change = close_prices[i] - close_prices[i-1]
        if change > 0:
            gains[i] = change
        else:
            losses[i] = -change
            
    # Calcular SMA de las ganancias y pérdidas
    for i in prange(rsi_period, n):
        avg_gain = np.mean(gains[i-rsi_period:i])
        avg_loss = np.mean(losses[i-rsi_period:i])
        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))

    # Simulación de Trading
    position = 0 # 0=flat, 1=long
    entry_price = 0.0
    
    for i in range(rsi_period, n-1):
        # Lógica de Salida (Stop Loss / Take Profit)
        if position == 1:
            pnl_pct = (close_prices[i] - entry_price) / entry_price
            if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct:
                position = 0
                returns[i] = pnl_pct
                
        # Lógica de Entrada
        if position == 0 and rsi[i] < rsi_lower:
            position = 1
            entry_price = close_prices[i]
            
    return returns

def run_backtest_fidelity(fidelity_level, params):
    """
    Wrapper multi-fidelidad.
    F1: 100 velas (microsegundos)
    F2: 1000 velas
    F3: 5000 velas
    F4: 10000 velas (Validación final)
    """
    fidelity_map = {
        'F1': 100,
        'F2': 1000,
        'F3': 5000,
        'F4': 10000
    }
    
    n_candles = fidelity_map.get(fidelity_level, 1000)
    
    # Generar datos sintéticos (Random Walk simulando mercado)
    np.random.seed(42) # Fijo para reproducibilidad entre iteraciones
    returns = np.random.normal(0.0001, 0.002, n_candles)
    close_prices = 100.0 * np.exp(np.cumsum(returns))
    
    # Extraer parámetros
    rsi_p = params.get('rsi_period', 14)
    rsi_l = params.get('rsi_lower', 30.0)
    rsi_u = params.get('rsi_upper', 70.0)
    sl = params.get('stop_loss', 0.02)
    tp = params.get('take_profit', 0.04)
    
    # Ejecutar JIT
    trade_returns = vectorized_backtest_core(close_prices, rsi_p, rsi_l, rsi_u, sl, tp)
    
    # Calcular Geometric Mean Return (Evitar NaN)
    valid_returns = trade_returns[trade_returns != 0.0]
    if len(valid_returns) == 0:
        return -1.0 # Penalizar inactividad
        
    geo_mean = np.prod(1 + valid_returns) ** (1/max(1, len(valid_returns))) - 1
    
    return float(geo_mean)
