# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np

cdef struct BacktestResult:
    double pnl
    int trades
    double win_rate
    double profit_factor
    double max_dd

cdef BacktestResult backtest_core(
    long long[:] t, double[:] o, double[:] h, double[:] l, double[:] c, 
    double[:] v, double[:] oi, double[:] f,
    int n, int lookback, double price_drop_pct, double vol_spike_mult, 
    double oi_drop_pct, double sl_pct, double tp_pct
) nogil:
    
    cdef int in_position = 0
    cdef int direction = 0
    cdef double entry_price = 0.0
    cdef double stop_loss = 0.0
    cdef double take_profit = 0.0
    
    cdef double total_pnl = 0.0
    cdef double peak_pnl = 0.0
    cdef double max_dd = 0.0
    
    cdef int trades = 0
    cdef int wins = 0
    
    cdef double gross_profit = 0.0
    cdef double gross_loss = 0.0
    
    cdef int i, j
    cdef double sum_vol, avg_vol, current_close, past_close, price_change
    cdef double current_oi, past_oi, oi_change, pnl_pct, dd
    
    for i in range(lookback, n - 1):
        if in_position == 1:
            pnl_pct = 0.0
            
            if direction == 1:
                if l[i] <= stop_loss:
                    pnl_pct = ((stop_loss - entry_price) / entry_price) - 0.01 
                    in_position = 0
                elif h[i] >= take_profit:
                    pnl_pct = (take_profit - entry_price) / entry_price
                    in_position = 0
                elif f[i] < -0.0001:
                    pnl_pct = (c[i] - entry_price) / entry_price
                    in_position = 0
            else:
                if h[i] >= stop_loss:
                    pnl_pct = ((entry_price - stop_loss) / entry_price) - 0.01
                    in_position = 0
                elif l[i] <= take_profit:
                    pnl_pct = (entry_price - take_profit) / entry_price
                    in_position = 0
                elif f[i] > 0.0001:
                    pnl_pct = (entry_price - c[i]) / entry_price
                    in_position = 0
                    
            if in_position == 0:
                total_pnl += pnl_pct
                trades += 1
                if pnl_pct > 0:
                    wins += 1
                    gross_profit += pnl_pct
                else:
                    gross_loss -= pnl_pct
                    
                if total_pnl > peak_pnl:
                    peak_pnl = total_pnl
                dd = peak_pnl - total_pnl
                if dd > max_dd:
                    max_dd = dd
            continue
            
        sum_vol = 0.0
        for j in range(i - lookback, i):
            sum_vol += v[j]
        avg_vol = sum_vol / lookback
        
        current_close = c[i]
        past_close = c[i - lookback]
        price_change = (current_close - past_close) / past_close
        
        current_oi = oi[i]
        past_oi = oi[i - lookback]
        oi_change = 0.0
        if past_oi > 0:
            oi_change = (current_oi - past_oi) / past_oi
            
        if price_change <= -price_drop_pct and v[i] > avg_vol * vol_spike_mult and oi_change <= -oi_drop_pct:
            if f[i] > 0.0:
                in_position = 1
                direction = 1
                entry_price = o[i+1] * 1.005 
                stop_loss = l[i] * (1.0 - sl_pct)
                take_profit = entry_price * (1.0 + tp_pct)
                continue
                
        if price_change >= price_drop_pct and v[i] > avg_vol * vol_spike_mult and oi_change <= -oi_drop_pct:
            if f[i] < 0.0:
                in_position = 1
                direction = -1
                entry_price = o[i+1] * 0.995 
                stop_loss = h[i] * (1.0 + sl_pct)
                take_profit = entry_price * (1.0 - tp_pct)
                continue
                
    cdef BacktestResult res
    res.pnl = total_pnl
    res.trades = trades
    res.win_rate = <double>wins / <double>trades if trades > 0 else 0.0
    res.profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    res.max_dd = max_dd
    
    return res

def run_macro_cpp(
    np.ndarray[np.int64_t, ndim=1] t,
    np.ndarray[np.float64_t, ndim=1] o,
    np.ndarray[np.float64_t, ndim=1] h,
    np.ndarray[np.float64_t, ndim=1] l,
    np.ndarray[np.float64_t, ndim=1] c,
    np.ndarray[np.float64_t, ndim=1] v,
    np.ndarray[np.float64_t, ndim=1] oi,
    np.ndarray[np.float64_t, ndim=1] f,
    int lookback, double price_drop_pct, double vol_spike_mult, 
    double oi_drop_pct, double sl_pct, double tp_pct
):
    cdef int n = t.shape[0]
    cdef BacktestResult res = backtest_core(
        t, o, h, l, c, v, oi, f, n,
        lookback, price_drop_pct, vol_spike_mult, oi_drop_pct, sl_pct, tp_pct
    )
    
    return res.pnl, res.trades, res.win_rate, res.profit_factor, res.max_dd
