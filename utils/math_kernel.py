import numpy as np
from numba import njit, prange, float64, int64

# ==============================================================================
# 🧠 FASE 10: QUANTITATIVE MASTERY (Hurst & RANSAC)
# ==============================================================================

@njit(cache=True)
def compute_alpha_decay_jit(time_held_sec: float, ttl_sec: float) -> float:
    """
    Computes a bayesian probability decay multiplier based on holding time.
    Uses a smooth decay curve that accelerates towards 0 as time_held approaches TTL.
    Returns: A multiplier between 0.0 and 1.0
    """
    if ttl_sec <= 0: return 1.0
    if time_held_sec >= ttl_sec: return 0.0
    
    # Smooth decay using an inverse sigmoid shape
    # At t=0 -> 1.0. At t=ttl -> 0.0. 
    ratio = time_held_sec / ttl_sec
    decay_factor = 1.0 - (ratio ** 1.5)
    return max(0.0, min(1.0, decay_factor))


@njit(cache=True)
def kahan_sum(arr):
    """
    [PRECISION-AXIOMA] Kahan Summation Algorithm.
    Prevents catastrophic loss of significance when summing large arrays of floating point numbers.
    Maintains a running compensation for accumulated rounding errors.
    """
    sum_val = 0.0
    c = 0.0 # Running compensation for lost low-order bits
    for i in range(len(arr)):
        y = arr[i] - c
        t = sum_val + y
        c = (t - sum_val) - y
        sum_val = t
    return sum_val

@njit(cache=True)
def calculate_hurst_exponent(prices, max_lags=20):
    """
    Hurst Exponent (Variance Ratio Method) - Fast Numba Vectorized
    Calcula el exponente de Hurst para determinar el régimen de mercado.
    
    Interpretación:
    - H < 0.5: Reversión a la Media (Rango)
    - H ≈ 0.5: Random Walk (Ruido Geométrico Browniano)
    - H > 0.5: Tendencia Persistente (Trending)
    
    Args:
        prices: Array numpy de precios
        max_lags: Profundidad máxima de retardo
    """
    n = len(prices)
    if n < max_lags * 2:
        return 0.5 # Insufficient data, assume random walk
        
    # Standard lags
    lags = np.arange(2, max_lags + 1)
    tau = np.zeros(len(lags), dtype=np.float64)
    
    for k in range(len(lags)):
        lag = lags[k]
        # Restar los precios con un desplazamiento 'lag'
        # Std( P(t+lag) - P(t) ) proporcional a lag^H
        # diffs tiene tamaño n - lag
        diffs = np.empty(n - lag, dtype=np.float64)
        for i in range(n - lag):
            diffs[i] = prices[i + lag] - prices[i]
            
        tau[k] = np.std(diffs)
        
    # Linear Regression en el espacio Log-Log
    # log(Std) = H * log(lag) + c
    valid = tau > 0
    if np.sum(valid) < 3:
        return 0.5
        
    log_lags = np.log(lags[valid])
    log_tau = np.log(tau[valid])
    
    # Regress log_tau on log_lags to find slope H
    mean_x = np.mean(log_lags)
    mean_y = np.mean(log_tau)
    
    cov_xy = np.mean((log_lags - mean_x) * (log_tau - mean_y))
    var_x = np.mean((log_lags - mean_x)**2)
    
    if var_x == 0:
        return 0.5
        
    H = cov_xy / var_x
    
    # Clip H to theoretical limits 0-1
    if H < 0.0: return 0.0
    if H > 1.0: return 1.0
    return H

@njit(cache=True)
def calculate_ransac_volatility(prices, threshold_ratio=3.0, min_samples=0.5, iterations=50):
    """
    RANSAC (Random Sample Consensus) 1D Volatility - Numba Vectorized
    Calcula una Desviación Estándar Robusta ignorando outliers (spikes/flash crashes).
    Ideal para canales de volatilidad inmunes al ruido de Binance.
    
    Args:
        prices: Array numpy de precios (ej: ventana de 20 periodos).
        threshold_ratio: Multiplicador del MAD (Median Absolute Deviation) para clasificar inliers.
        min_samples: Fracción mínima de inliers requerida.
        iterations: Número de muestras aleatorias a probar.
        
    Returns:
        robust_std, robust_mean
    """
    n = len(prices)
    if n < 5:
        # Fallback to standard stats if window is too small
        return np.std(prices), np.mean(prices)
        
    best_inlier_count = 0
    best_inliers = np.empty(n, dtype=np.bool_)
    best_inliers[:] = False
    
    # Calculate global MAD for baseline threshold
    med = np.median(prices)
    mad = np.median(np.abs(prices - med))
    if mad == 0:
        mad = 1e-8
    threshold = mad * threshold_ratio
    
    for _ in range(iterations):
        # 1. Random Sample (2 points to define a 1D "model" mean)
        i1 = np.random.randint(0, n)
        i2 = np.random.randint(0, n)
        while i1 == i2:
            i2 = np.random.randint(0, n)
            
        sample_mean = (prices[i1] + prices[i2]) / 2.0
        
        # 2. Evaluate consensus (count inliers)
        inliers = np.abs(prices - sample_mean) <= threshold
        inlier_count = np.sum(inliers)
        
        # 3. Update best model
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inliers = inliers
            
    # 4. Filter and return robust stats
    # If we couldn't find a consensus, fallback to standard stats
    if best_inlier_count < n * min_samples:
        return np.std(prices), np.mean(prices)
        
    # Extract inliers
    inlier_prices = prices[best_inliers]
    
    return np.std(inlier_prices), np.mean(inlier_prices)

# ==============================================================================
# EXISTING TECHNICAL INDICATORS
# ==============================================================================

@njit(fastmath=True, cache=True)
def calculate_ema_jit(prices, period):
    """
    Exponential Moving Average - JIT Compiled (O(N))
    """
    n = len(prices)
    ema = np.empty(n, dtype=np.float64)
    alpha = 2.0 / (period + 1)
    
    # Initialize with SMA of first 'period' elements
    # Optim: Just use first price as seed if N < period? 
    # Standard: SMA of first 'period'
    if n < period:
        ema[:] = np.nan
        return ema
        
    # SMA initialization
    sma = 0.0
    for i in range(period):
        sma += prices[i]
    ema[period-1] = sma / period
    
    # Fill Pre-EMA with NaNs
    ema[:period-1] = np.nan
    
    # EMA Calculation
    for i in range(period, n):
        ema[i] = (prices[i] - ema[i-1]) * alpha + ema[i-1]
        
    return ema

@njit(fastmath=True, parallel=True, cache=True)
def calculate_rsi_jit(prices, period=14):
    """
    Relative Strength Index - JIT Compiled
    Parallelized element-wise ops where possible, but recursive dependency limits parallel gains.
    However, gain/loss array creation IS parallelizable.
    """
    n = len(prices)
    rsi = np.full(n, np.nan, dtype=np.float64)
    
    if n <= period:
        return rsi
        
    deltas = np.empty(n, dtype=np.float64)
    # Vectorized difference (could use np.diff but manual for JIT speed)
    # Parallelize this loop? No, generic np.diff is fast enough? 
    # Let's keep it simple JIT loop.
    for i in range(1, n):
        deltas[i] = prices[i] - prices[i-1]
    
    # Initial gain/loss
    gain = 0.0
    loss = 0.0
    
    for i in range(1, period + 1):
        d = deltas[i]
        gain += np.fmax(0.0, d)
        loss += np.fmax(0.0, -d)
            
    avg_gain = gain / period
    avg_loss = loss / period
    
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))
        
    # Smoothing Smoothed Moving Average
    for i in range(period + 1, n):
        d = deltas[i]
        current_gain = np.fmax(0.0, d)
        current_loss = np.fmax(0.0, -d)
        
        avg_gain = (avg_gain * (period - 1) + current_gain) / period
        avg_loss = (avg_loss * (period - 1) + current_loss) / period
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
            
    return rsi

@njit(fastmath=True, parallel=True, cache=True)
def calculate_bollinger_jit(prices, period=20, std_dev=2.0):
    """
    Bollinger Bands - JIT Compiled
    Highly Parallelizable (SMA and STD over windows can be parallelized via prange? No, sliding window dependency on index).
    Wait, sliding window IS independent per output pixel if calculated brute force, but that's O(N*W).
    Incremental Welford or simple rolling is O(N).
    We stick to O(N) serial optimization with fastmath.
    """
    n = len(prices)
    upper = np.full(n, np.nan, dtype=np.float64)
    middle = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    if n < period:
        return upper, middle, lower
        
    # 1. Calculate Middle Band (SMA) & Std Dev
    # Using window sum for O(1) rolling update
    window_sum = 0.0
    window_sum_sq = 0.0
    
    # Init
    for i in range(period):
        val = prices[i]
        window_sum += val
        window_sum_sq += val * val
        
    # Rest
    for i in range(period - 1, n):
        # We need sum from i-period+1 to i
        if i >= period:
            # Add new, remove old
            val_new = prices[i]
            val_old = prices[i-period]
            window_sum += val_new - val_old
            window_sum_sq += val_new*val_new - val_old*val_old
            
        # Compute Stats
        mean = window_sum / period
        variance = (window_sum_sq / period) - (mean * mean)
        
        if variance < 0: variance = 0.0 # Float precision safety
        std = np.sqrt(variance)
        
        middle[i] = mean
        upper[i] = mean + (std * std_dev)
        lower[i] = mean - (std * std_dev)
        
    return upper, middle, lower

@njit(parallel=True, cache=True)
def calculate_bollinger_robust_jit(prices, period=20, std_dev=2.0, threshold_ratio=3.0, iterations=30):
    """
    Bollinger Bands Robustos (RANSAC Volatility) - Fase 10
    Calcula las bandas ignorando outliers (flash crashes) en la std.
    O(N * W), aceptable bajo compilación Numba.
    """
    n = len(prices)
    upper = np.full(n, np.nan, dtype=np.float64)
    middle = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    if n < period:
        return upper, middle, lower
        
    for i in prange(period - 1, n):
        window = prices[i - period + 1 : i + 1]
        
        # Calcular robust stats
        rob_std, rob_mean = calculate_ransac_volatility(
            window, threshold_ratio=threshold_ratio, min_samples=0.5, iterations=iterations
        )
        
        middle[i] = rob_mean
        upper[i] = rob_mean + (rob_std * std_dev)
        lower[i] = rob_mean - (rob_std * std_dev)
        
    return upper, middle, lower

@njit(fastmath=True, cache=True)
def calculate_zscore_jit(prices, period=20):
    """
    Rolling Z-Score - JIT Compiled (O(N) Optimization).
    Calculates moving mean and variance in a single pass to trigger SIMD.
    """
    n = len(prices)
    zscores = np.zeros(n, dtype=np.float64)  # F6: Was float32, causing precision loss
    
    if n < period:
        return zscores
        
    # Initial window sum
    window_sum = 0.0
    window_sum_sq = 0.0
    for i in range(period):
        val = prices[i]
        window_sum += val
        window_sum_sq += val * val
        
    # Main loop (O(N))
    for i in range(period - 1, n):
        if i >= period:
            val_new = prices[i]
            val_old = prices[i - period]
            window_sum += val_new - val_old
            window_sum_sq += val_new * val_new - val_old * val_old
            
        mean = window_sum / period
        variance = (window_sum_sq / period) - (mean * mean)
        
        # Numerical stability check
        if variance < 1e-10:
            std = 0.0
        else:
            std = np.sqrt(variance)
            
        if std > 1e-8:
            zscores[i] = (prices[i] - mean) / std
        else:
            zscores[i] = 0.0
            
    return zscores

@njit(fastmath=True, cache=True)
def vector_zscore(prices, period=20):
    """
    [NANO-SPEED] Vectorized Z-Score calculation for feature engineering.
    """
    return calculate_zscore_jit(prices, period)

@njit(fastmath=True, cache=True)
def bayesian_probability_jit(signal_strength, trend_strength, volatility_z):
    """
    Calcula la probabilidad bayesiana de éxito de un trade dado el contexto.
    Prior: 0.5 (Neutral)
    Evidence: Signal + Trend + Volatility
    """
    # Prior odds
    prior_prob = 0.5
    
    # Likelihood ratios (Simplified Naive Bayes)
    # Signal Strength (0.0 to 1.0) -> Multiplier 0.5x to 2.0x
    lr_signal = 0.5 + (signal_strength * 1.5)
    
    # Trend alignment (-1.0 to 1.0) -> Multiplier 0.5x to 1.5x
    # If aligned with signal (both pos or both neg), boost.
    lr_trend = 1.0
    if trend_strength > 0.5: lr_trend = 1.3
    elif trend_strength < -0.5: lr_trend = 0.7
    
    # Volatility Z-Score (Mean Reversion vs Breakout)
    # High Z (>2) means extreme, Higher risk of reversion unless breakout strategy
    lr_vol = 1.0
    if abs(volatility_z) > 3.0:
        lr_vol = 0.6 # Too stretched, risky
    elif abs(volatility_z) > 1.5:
        lr_vol = 1.2 # Good momentum
    else:
        lr_vol = 0.9 # Low noise
        
    # Posterior Odds = Prior Odds * LR1 * LR2 * LR3
    posterior_odds = (prior_prob / (1.0 - prior_prob)) * lr_signal * lr_trend * lr_vol
    
    # Probability = Odds / (1 + Odds)
    probability = posterior_odds / (1.0 + posterior_odds)
    
    return probability

@njit(fastmath=True, cache=True)
def calculate_correlation_matrix_jit(price_matrix):
    """
    Fast Pearson Correlation Matrix - SIMD Optimized.
    Input: (N_samples, M_assets) array.
    Output: (M, M) correlation matrix.
    Uses centralized normalization to leverage instruction-level parallelism.
    """
    n_samples, m_assets = price_matrix.shape
    # Ensure memory alignment for SIMD — F6: Changed to float64 (price precision)
    price_matrix_f64 = price_matrix.astype(np.float64)
    
    # 1. Compute Means & Normalize (Vectorized)
    # Numba will auto-vectorize these col-wise ops
    norm_matrix = np.zeros((n_samples, m_assets), dtype=np.float64)
    for j in range(m_assets):
        col = price_matrix_f64[:, j]
        mean = np.mean(col)
        std = np.std(col)
        if std > 1e-8:
            norm_matrix[:, j] = (col - mean) / std
        else:
            norm_matrix[:, j] = 0.0
            
    # 2. Compute Correlation via Dot Product (BLAS SIMD)
    # Correlation of normalized variables is just (X' * X) / N
    corr_matrix = np.dot(norm_matrix.T, norm_matrix) / n_samples
    
    return corr_matrix
@njit(fastmath=True, cache=True)
def calculate_macd_jit(prices, fast_period=12, slow_period=26, signal_period=9):
    """
    MACD - JIT Compiled.
    Returns: macd, signal, hist
    """
    ema_fast = calculate_ema_jit(prices, fast_period)
    ema_slow = calculate_ema_jit(prices, slow_period)
    
    macd = ema_fast - ema_slow
    
    # Signal is EMA of MACD
    # Need to handle NaNs from EMA slow
    signal = np.full(len(macd), np.nan, dtype=np.float64)
    valid_start = slow_period - 1
    if len(macd) > valid_start + signal_period:
        macd_valid = macd[valid_start:]
        sig_ema = calculate_ema_jit(macd_valid, signal_period)
        signal[valid_start:] = sig_ema
        
    hist = macd - signal
    return macd, signal, hist

@njit(fastmath=True, cache=True)
def calculate_atr_jit(high, low, close, period=14):
    """
    ATR - JIT Compiled.
    """
    n = len(close)
    tr = np.zeros(n, dtype=np.float64)
    atr = np.full(n, np.nan, dtype=np.float64)
    
    if n < period:
        return atr
        
    # 1. Calculate True Range [BRANCHLESS]
    for i in range(1, n):
        h_l = high[i] - low[i]
        h_pc = abs(high[i] - close[i-1])
        l_pc = abs(low[i] - close[i-1])
        tr[i] = np.fmax(h_l, np.fmax(h_pc, l_pc))
    
    # 2. Initial ATR (SMA of TR)
    tr_sum = 0.0
    for i in range(1, period + 1):
        tr_sum += tr[i]
    atr[period] = tr_sum / period
    
    # 3. Smoothed ATR
    for i in range(period + 1, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
    return atr

@njit(fastmath=True, cache=True)
def calculate_adx_jit(high, low, close, period=14):
    """
    ADX - JIT Compiled.
    """
    n = len(close)
    adx = np.full(n, np.nan, dtype=np.float64)
    
    if n < 2 * period:
        return adx
        
    up_move = np.zeros(n, dtype=np.float64)
    down_move = np.zeros(n, dtype=np.float64)
    dm_pos = np.zeros(n, dtype=np.float64)
    dm_neg = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        
        if up > down and up > 0:
            dm_pos[i] = up
        if down > up and down > 0:
            dm_neg[i] = down
            
    # Smoothing techniques similar to ATR (Wilder's)
    # We need ATR too
    atr = calculate_atr_jit(high, low, close, period)
    
    smooth_dm_pos = np.zeros(n, dtype=np.float64)
    smooth_dm_neg = np.zeros(n, dtype=np.float64)
    
    # Init
    sdm_p = 0.0
    sdm_n = 0.0
    for i in range(1, period + 1):
        sdm_p += dm_pos[i]
        sdm_n += dm_neg[i]
        
    smooth_dm_pos[period] = sdm_p
    smooth_dm_neg[period] = sdm_n
    
    di_pos = np.zeros(n, dtype=np.float64)
    di_neg = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)
    
    for i in range(period, n):
        if i > period:
            smooth_dm_pos[i] = smooth_dm_pos[i-1] - (smooth_dm_pos[i-1] / period) + dm_pos[i]
            smooth_dm_neg[i] = smooth_dm_neg[i-1] - (smooth_dm_neg[i-1] / period) + dm_neg[i]
            
        if atr[i] > 0:
            di_pos[i] = 100 * (smooth_dm_pos[i] / atr[i])
            di_neg[i] = 100 * (smooth_dm_neg[i] / atr[i])
        
        denom = di_pos[i] + di_neg[i]
        if denom > 0:
            dx[i] = 100 * abs(di_pos[i] - di_neg[i]) / denom
            
    # ADX is smoothing of DX
    dx_sum = 0.0
    for i in range(period, 2 * period):
        dx_sum += dx[i]
    adx[2*period - 1] = dx_sum / period
    
    for i in range(2 * period, n):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
        
    return adx

@njit(fastmath=True, cache=True)
def calculate_hurst_jit(prices, period=20):
    """
    Hurst Exponent - JIT Compiled (Simplified R/S Analysis via Variance Ratio).
    0.5 = Random Walk
    > 0.5 = Trending (Persistent)
    < 0.5 = Mean Reverting (Anti-Persistent)
    Uses Variance Difference method for O(N) estimation.
    """
    n = len(prices)
    if n < period:
        return 0.5

    # Variance Difference Method (Generalized Hurst) is robust for small samples
    # Log[Var(tau)] ~ 2H * Log[tau]
    # We estimate for tau=1 and tau=period/2
    
    # Lag 1
    tau1 = 1
    sum_sq_diff1 = 0.0
    count1 = 0
    for i in range(tau1, n):
        d = prices[i] - prices[i-tau1]
        sum_sq_diff1 += d * d
        count1 += 1
        
    if count1 == 0: return 0.5
    var1 = sum_sq_diff1 / count1
    
    # Lag 2 (Adaptive, roughly 1/4 to 1/2 of period for stability)
    tau2 = max(2, period // 4)
    if tau2 >= n: tau2 = n // 2
    if tau2 <= tau1: tau2 = tau1 + 1
    
    sum_sq_diff2 = 0.0
    count2 = 0
    for i in range(tau2, n):
        d = prices[i] - prices[i-tau2]
        sum_sq_diff2 += d * d
        count2 += 1
        
    if count2 == 0: return 0.5
    var2 = sum_sq_diff2 / count2
    
    # Avoid log(0)
    if var1 < 1e-12 or var2 < 1e-12:
        return 0.5
        
    # H approx = 0.5 * (log(var2) - log(var1)) / (log(tau2) - log(tau1))
    log_tau1 = np.log(float(tau1))
    log_tau2 = np.log(float(tau2))
    log_var1 = np.log(var1)
    log_var2 = np.log(var2)
    
    denom = log_tau2 - log_tau1
    if abs(denom) < 1e-9: return 0.5
    
    h = 0.5 * (log_var2 - log_var1) / denom
    
    # Clamp to theoretical bounds
    if h < 0.0: return 0.0
    if h > 1.0: return 1.0
    return h

@njit(fastmath=True, cache=True)
def calculate_expectancy_jit(win_rate, avg_win, avg_loss):
    """
    Mathematical Expectation = (WinRate * AvgWin) - (LossRate * Abs(AvgLoss))
    Expectancy Ratio = Expectancy / Abs(AvgLoss)  (Optional, but raw value is safer)
    """
    loss_rate = 1.0 - win_rate
    # Ensure positive AvgWin, positive AvgLoss (magnitude)
    aw = abs(avg_win)
    al = abs(avg_loss)
    
    ev = (win_rate * aw) - (loss_rate * al)
    return ev

@njit(fastmath=True, cache=True)
def calculate_garch_jit(returns, omega=1e-6, alpha=0.05, beta=0.90):
    """
    Simulates GARCH(1,1) variance forecast process.
    sigma^2_t = omega + alpha * epsilon^2_{t-1} + beta * sigma^2_{t-1}
    Returns: array of conditional variances (sigma^2)
    """
    n = len(returns)
    variances = np.zeros(n, dtype=np.float64)
    
    if n < 2: return variances
    
    # Initialize with sample variance
    variances[0] = np.var(returns)
    
    for t in range(1, n):
        resid = returns[t-1]
        variances[t] = omega + alpha * (resid * resid) + beta * variances[t-1]
        
    return variances

# ==============================================================================
# 🚀 A1 FIX: BATCH VECTORIZED QUANTUM FEATURES (Nano-Speed)
# ==============================================================================
# QUÉ: Calcula Hurst, RANSAC-Volatility y Bayesian Probability para TODOS
#       los bars de una vez en un solo loop JIT compilado.
# POR QUÉ: La versión anterior hacía 4980 llamadas Python→JIT individuales 
#           con overhead de ~40μs por llamada = 200ms por símbolo.
# PARA QUÉ: Reducir latencia de feature engineering de ~200ms a ~5ms por símbolo.
# CÓMO: Un solo loop Numba que itera internamente sin volver a Python.
# CUÁNDO: Cada vez que se llama prepare_features() en feature_engineering.py.
# DÓNDE: utils/math_kernel.py
# QUIÉN: FeatureEngineering.prepare_features() → calculate_quantum_features_batch_jit()

@njit(fastmath=True, cache=True)
def calculate_quantum_features_batch_jit(close, z_scores, returns_5, period=20):
    """
    BATCH Quantum Features: Hurst + RANSAC + Bayesian for entire array.
    Single JIT call replaces ~N individual Python→JIT calls.
    
    Args:
        close: float64 array of close prices
        z_scores: float64 array of pre-computed z-scores
        returns_5: float64 array of 5-bar returns (may contain NaN)
        period: lookback window size
        
    Returns:
        hurst_arr, ransac_arr, bayes_arr (all float64 arrays of len(close))
    """
    n = len(close)
    hurst_arr = np.full(n, 0.5, dtype=np.float64)
    ransac_arr = np.zeros(n, dtype=np.float64)
    bayes_arr = np.full(n, 0.5, dtype=np.float64)
    
    for i in range(period, n):
        window = close[i - period:i + 1]
        w_len = len(window)
        
        # === INLINE HURST (Variance Difference Method) ===
        # Avoids function call overhead
        if w_len >= period:
            # Lag 1 variance
            sum_sq_1 = 0.0
            cnt_1 = 0
            for k in range(1, w_len):
                d = window[k] - window[k - 1]
                sum_sq_1 += d * d
                cnt_1 += 1
            
            # Lag tau2 variance
            tau2 = max(2, period // 4)
            if tau2 >= w_len:
                tau2 = w_len // 2
            if tau2 < 2:
                tau2 = 2
            
            sum_sq_2 = 0.0
            cnt_2 = 0
            for k in range(tau2, w_len):
                d = window[k] - window[k - tau2]
                sum_sq_2 += d * d
                cnt_2 += 1
            
            if cnt_1 > 0 and cnt_2 > 0:
                var1 = sum_sq_1 / cnt_1
                var2 = sum_sq_2 / cnt_2
                
                if var1 > 1e-12 and var2 > 1e-12:
                    log_tau1 = 0.0  # log(1) = 0
                    log_tau2 = np.log(float(tau2))
                    denom_h = log_tau2 - log_tau1
                    
                    if abs(denom_h) > 1e-9:
                        h = 0.5 * (np.log(var2) - np.log(var1)) / denom_h
                        if h < 0.0:
                            h = 0.0
                        elif h > 1.0:
                            h = 1.0
                        hurst_arr[i] = h
        
        # === INLINE RANSAC VOLATILITY (Simplified — 30 iterations) ===
        if w_len >= 5:
            med = np.median(window)
            abs_devs = np.empty(w_len, dtype=np.float64)
            for k in range(w_len):
                abs_devs[k] = abs(window[k] - med)
            mad = np.median(abs_devs)
            if mad < 1e-8:
                mad = 1e-8
            threshold = mad * 3.0
            
            best_count = 0
            best_inliers = np.empty(w_len, dtype=np.bool_)
            best_inliers[:] = False
            
            for _ in range(30):  # Reduced from 50
                i1 = np.random.randint(0, w_len)
                i2 = np.random.randint(0, w_len)
                while i1 == i2:
                    i2 = np.random.randint(0, w_len)
                
                sample_mean = (window[i1] + window[i2]) / 2.0
                count = 0
                inliers = np.abs(window - sample_mean) <= threshold
                for k in range(w_len):
                    if inliers[k]:
                        count += 1
                
                if count > best_count:
                    best_count = count
                    best_inliers = inliers
            
            if best_count >= w_len // 2:
                inlier_sum = 0.0
                inlier_sq_sum = 0.0
                cnt = 0
                for k in range(w_len):
                    if best_inliers[k]:
                        inlier_sum += window[k]
                        inlier_sq_sum += window[k] * window[k]
                        cnt += 1
                if cnt > 1:
                    mean_val = inlier_sum / cnt
                    var_val = (inlier_sq_sum / cnt) - (mean_val * mean_val)
                    if var_val < 0:
                        var_val = 0.0
                    ransac_arr[i] = np.sqrt(var_val)
                else:
                    ransac_arr[i] = np.std(window)
            else:
                ransac_arr[i] = np.std(window)
        
        # === INLINE BAYESIAN PROBABILITY ===
        r_val = returns_5[i]
        if np.isnan(r_val):
            r_val = 0.0
        
        sig_str = abs(r_val) / 0.02
        if sig_str > 1.0:
            sig_str = 1.0
        
        z_val = z_scores[i]
        trend_str = 0.0
        if r_val > 0 and z_val > 0:
            trend_str = 1.0
        elif r_val < 0 and z_val < 0:
            trend_str = -1.0
        
        # Likelihood ratios
        lr_signal = 0.5 + (sig_str * 1.5)
        
        lr_trend = 1.0
        if trend_str > 0.5:
            lr_trend = 1.3
        elif trend_str < -0.5:
            lr_trend = 0.7
        
        lr_vol = 1.0
        abs_z = abs(z_val)
        if abs_z > 3.0:
            lr_vol = 0.6
        elif abs_z > 1.5:
            lr_vol = 1.2
        else:
            lr_vol = 0.9
        
        posterior_odds = 1.0 * lr_signal * lr_trend * lr_vol  # prior_odds = 1.0 (50/50)
        bayes_arr[i] = posterior_odds / (1.0 + posterior_odds)
    
    return hurst_arr, ransac_arr, bayes_arr

# ==============================================================================
# 🧠 FASE 11: NANO-LATENCY JIT KERNELS FOR RISK & SOPHIA
# ==============================================================================

@njit(fastmath=True, cache=True)
def compute_kelly_fraction_jit(p, b, apply_mult=True, kelly_mult=0.25, stress_score=100.0, max_exposure=0.40):
    """
    [NANO-SPEED] Reemplazo de `Decimal` de Python para el Criterio de Kelly.
    Velocidad de ejecución en nanosegundos (vs milisegundos de Decimal Python).
    Fórmula: (p * b - q) / b donde q = 1 - p
    """
    if b <= 0.0:
        return 0.0
        
    q = 1.0 - p
    kelly = (p * b - q) / b
    
    if not apply_mult:
        return kelly
        
    # Defensive Scaling
    mult = kelly_mult
    if stress_score < 90.0:
        mult = 0.125  # Eighth-Kelly under extreme stress
        
    fractional_kelly = kelly * mult
    if fractional_kelly < 0.0:
        fractional_kelly = 0.0
        
    clamped = fractional_kelly
    if clamped > max_exposure:
        clamped = max_exposure
        
    return clamped

@njit(fastmath=True, cache=True)
def extract_kelly_stats_jit(pnl_array, is_win_array):
    """
    Deduce win rate (p) and payoff ratio (b) from a set of trade returns.
    """
    n = len(pnl_array)
    if n == 0:
        return 0.5, 1.0
        
    wins = 0.0
    losses = 0.0
    sum_wins = 0.0
    sum_losses = 0.0
    
    for i in range(n):
        if is_win_array[i]:
            wins += 1.0
            sum_wins += pnl_array[i]
        else:
            losses += 1.0
            sum_losses += abs(pnl_array[i])
            
    p = wins / n if n > 0 else 0.5
    avg_win = sum_wins / wins if wins > 0 else 0.01
    avg_loss = sum_losses / losses if losses > 0 else 0.01
    b = avg_win / avg_loss if avg_loss > 0 else 1.0
    
    return p, b

@njit(fastmath=True, cache=True)
def compute_cvar_jit(loss_history, confidence_level=0.95):
    """
    [NANO-SPEED] Conditional Value at Risk calculation.
    Uses numpy backend rather than python `sorted()` list logic.
    """
    n = len(loss_history)
    if n < 10:
        return 0.05
        
    # Sort in descending order
    sorted_losses = np.sort(loss_history)[::-1]
    var_index = max(1, int(n * (1.0 - confidence_level)))
    
    # Calculate mean of the tail (worst cases)
    sum_loss = 0.0
    for i in range(var_index):
        sum_loss += sorted_losses[i]
        
    return sum_loss / float(var_index)

@njit(fastmath=True, cache=True)
def compute_shannon_entropy_jit(probs):
    """
    [NANO-SPEED] Shannon Entropy para Sophia Intelligence.
    H = -Σ p_i × log2(p_i)
    """
    h = 0.0
    for i in range(len(probs)):
        p = probs[i]
        if p > 1e-10:
            h -= p * np.log2(p)
    return h

@njit(fastmath=True, cache=True)
def compute_alpha_decay_jit(signal_strength, elapsed_seconds, ttl_seconds):
    """
    [NANO-SPEED] Decaimiento Exponencial Temporal de las señales en Sophia.
    Reemplaza la lenta llamada de math.exp().
    """
    if ttl_seconds <= 0.0:
        return 0.0
    lam = 1.0 / ttl_seconds
    return signal_strength * np.exp(-lam * elapsed_seconds)

# ==============================================================================
# 🧠 FASE 12: NANO-LATENCY JIT KERNELS FOR REGIME & WEBSOCKET LEAD-LAG
# ==============================================================================

@njit(fastmath=True, cache=True)
def compute_fuzzy_regime_scores_jit(adx, hurst, tm_multiplier, is_bullish):
    """
    [NANO-SPEED] Reemplaza la lógica serial Fuzzy para Regímenes.
    Devuelve índice del régimen ganador y su probabilidad:
    0: TRENDING_BEAR, 1: TRENDING_BULL, 2: MEAN_REVERTING, 3: RANGING, 4: CHOPPY
    """
    adx_base = 20.0 * tm_multiplier
    adx_range = 10.0 * tm_multiplier
    mr_adx_base = 22.0 * tm_multiplier
    
    # 1. P(Trending)
    p_trend_adx = max(0.0, min(1.0, (adx - adx_base) / adx_range))
    p_trend_hurst = max(0.0, min(1.0, (hurst - 0.5) / 0.15))
    score_trending = (p_trend_adx * 0.6) + (p_trend_hurst * 0.4)
    
    # 2. P(Mean-Reverting)
    p_mr_hurst = max(0.0, min(1.0, (0.45 - hurst) / 0.1))
    p_mr_adx = max(0.0, min(1.0, (mr_adx_base - adx) / 7.0))
    score_mean_reverting = (p_mr_hurst * 0.7) + (p_mr_adx * 0.3)
    
    # 3. P(Ranging)
    p_range_adx = max(0.0, min(1.0, (mr_adx_base - adx) / 7.0))
    dist_to_neutral = abs(hurst - 0.5)
    p_range_hurst = max(0.0, min(1.0, (0.1 - dist_to_neutral) / 0.1))
    score_ranging = (p_range_adx * 0.5) + (p_range_hurst * 0.5)
    
    # 4. P(Choppy)
    max_3 = max(score_trending, score_mean_reverting, score_ranging)
    score_choppy = max(0.0, 1.0 - max_3)
    
    # Map index
    best_idx = 4 # CHOPPY (Default)
    best_score = score_choppy
    
    if score_trending > best_score:
        best_score = score_trending
        best_idx = 1 if is_bullish else 0
        
    if score_mean_reverting > best_score:
        best_score = score_mean_reverting
        best_idx = 2
        
    if score_ranging > best_score:
        best_score = score_ranging
        best_idx = 3
        
    if best_score < 0.35:
        best_idx = 4 # Force Choppy
        
    return best_idx, best_score

@njit(fastmath=True, cache=True)
def pearson_correlation_jit(x, y):
    """
    [NANO-SPEED] Pearson Correlation Coefficient para reemplazar np.corrcoef()
    Extremadamente útil para cálculos Lead-Lag sin overhead overhead O(n^2) del GC.
    """
    n = len(x)
    if n < 2 or n != len(y):
        return 0.0
        
    sum_x = 0.0
    sum_y = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0
    sum_xy = 0.0
    
    for i in range(n):
        xi = x[i]
        yi = y[i]
        sum_x += xi
        sum_y += yi
        sum_x2 += xi * xi
        sum_y2 += yi * yi
        sum_xy += xi * yi
        
    numerator = (n * sum_xy) - (sum_x * sum_y)
    denominator = np.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
    
    if denominator == 0.0:
        return 0.0
    return numerator / denominator
