import numpy as np
from numba import njit
from core.nano_risk_engine import evaluate_sl_tp_trailing_jit

@njit(fastmath=True, nogil=True)
def batch_check_stops(
    lows: np.ndarray, 
    highs: np.ndarray, 
    closes: np.ndarray,
    entry_prices: np.ndarray,
    hwms: np.ndarray,
    lwms: np.ndarray,
    qtys: np.ndarray,
    sl_pcts: np.ndarray,
    tp_pcts: np.ndarray,
    atr_pcts: np.ndarray,
    is_zombie_chasers: np.ndarray,
    elastic_tp_expansions: np.ndarray,
    trailing_atr_mults: np.ndarray
) -> np.ndarray:
    """
    [NANO-SPEED] Evaluates ALL active positions against Low, High, and Close wicks in a single pass.
    Simulates the pessimistic intra-bar wick order: Low -> High -> Close.
    
    Returns an array of shape (N, 2) where:
    - col 0: exit_reason (0=KEEP, 1=SL, 2=TP, 3=TRAIL, 5=ZOMBIE)
    - col 1: exit_price
    """
    n = len(qtys)
    results = np.zeros((n, 2), dtype=np.float64)
    
    for i in range(n):
        if qtys[i] == 0:
            continue
            
        exit_reason = 0
        exit_price = 0.0
        
        # 1. Test LOW price (Pessimistic approach first)
        hwm_test = max(hwms[i], lows[i]) if qtys[i] > 0 else hwms[i]
        lwm_test = min(lwms[i], lows[i]) if qtys[i] < 0 else lwms[i]
        
        reason = evaluate_sl_tp_trailing_jit(
            lows[i], entry_prices[i], hwm_test, lwm_test, qtys[i],
            sl_pcts[i], tp_pcts[i], atr_pcts[i], 
            is_zombie_chasers[i] > 0, elastic_tp_expansions[i] > 0, trailing_atr_mults[i]
        )
        
        if reason > 0:
            exit_reason = reason
            exit_price = lows[i]
        else:
            # 2. Test HIGH price
            hwm_test = max(hwm_test, highs[i]) if qtys[i] > 0 else hwm_test
            lwm_test = min(lwm_test, highs[i]) if qtys[i] < 0 else lwm_test
            
            reason = evaluate_sl_tp_trailing_jit(
                highs[i], entry_prices[i], hwm_test, lwm_test, qtys[i],
                sl_pcts[i], tp_pcts[i], atr_pcts[i], 
                is_zombie_chasers[i] > 0, elastic_tp_expansions[i] > 0, trailing_atr_mults[i]
            )
            
            if reason > 0:
                exit_reason = reason
                exit_price = highs[i]
            else:
                # 3. Test CLOSE price
                hwm_test = max(hwm_test, closes[i]) if qtys[i] > 0 else hwm_test
                lwm_test = min(lwm_test, closes[i]) if qtys[i] < 0 else lwm_test
                
                reason = evaluate_sl_tp_trailing_jit(
                    closes[i], entry_prices[i], hwm_test, lwm_test, qtys[i],
                    sl_pcts[i], tp_pcts[i], atr_pcts[i], 
                    is_zombie_chasers[i] > 0, elastic_tp_expansions[i] > 0, trailing_atr_mults[i]
                )
                
                if reason > 0:
                    exit_reason = reason
                    exit_price = closes[i]
                    
        results[i, 0] = float(exit_reason)
        results[i, 1] = exit_price
        
    return results
