# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

cimport cython
from libc.math cimport sqrt, exp, log, fabs

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double compute_kelly_fraction(double win_rate, double win_loss_ratio):
    """
    Cython implementation of the Kelly Criterion.
    win_rate: Decimal (e.g. 0.55 for 55%)
    win_loss_ratio: Reward/Risk ratio (e.g. 1.5)
    Returns fraction to risk (e.g. 0.25 for 25%).
    """
    if win_loss_ratio <= 0.0 or win_rate <= 0.0:
        return 0.0
    
    cdef double q = 1.0 - win_rate
    cdef double k = win_rate - (q / win_loss_ratio)
    
    if k < 0.0:
        return 0.0
    return k

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double compute_dynamic_sizing(double current_balance, double kelly_fraction, double max_risk_limit, double atr_pct):
    """
    Computes absolute position size in USDT based on:
    - current balance
    - computed kelly fraction
    - the maximum allowable risk
    - current ATR (volatility measure)
    """
    cdef double risk_budget = current_balance * kelly_fraction
    # Clamp risk budget to absolute max
    if risk_budget > (current_balance * max_risk_limit):
        risk_budget = current_balance * max_risk_limit
        
    # How much position can we take given volatility?
    if atr_pct <= 0.0:
        return 0.0
        
    cdef double position_size = risk_budget / atr_pct
    return position_size

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int check_drawdown_limit(double peak_balance, double current_balance, double max_dd_pct):
    """
    Returns 1 if drawdown limit is exceeded, 0 otherwise.
    """
    if peak_balance <= 0.0:
        return 0
        
    cdef double dd = (peak_balance - current_balance) / peak_balance
    if dd >= max_dd_pct:
        return 1
    return 0
