
import numpy as np
from numba import njit, prange, float64, int64

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
