import numpy as np
from core.simulation_numba import technical_simulation_loop_njit

def run_fidelity(
    closes_arr: np.ndarray,
    tp_pct: float,
    sl_pct: float,
    window: int,
    fast_window: int,
    trend_conf: float,
    fidelity_level: str
) -> dict:
    """
    Capa de Multi-Fidelidad que encapsula el simulador HFT (Numba).
    Ejecuta el backtest solo sobre la porción de datos correspondiente al nivel.
    """
    total_bars = len(closes_arr)
    
    if fidelity_level == 'F1':
        # F1: Exploración rápida (Aprox 100 velas útiles post-window)
        end_idx = min(100 + window, total_bars)
    elif fidelity_level == 'F2':
        # F2: Filtrado intermedio (1000 velas)
        end_idx = min(1000 + window, total_bars)
    elif fidelity_level == 'F3':
        # F3: Validación (5000 velas)
        end_idx = min(5000 + window, total_bars)
    else:
        # F4: Certificación completa (Todo el array)
        end_idx = total_bars
        
    if end_idx <= window:
        # Not enough data for this fidelity
        return {'pnl': -999.0, 'max_dd': 1.0, 'win_rate': 0.0, 'trades': 0}
        
    trades_arr = technical_simulation_loop_njit(
        closes_arr,
        tp_pct,
        sl_pct,
        window,
        fast_window,
        trend_conf,
        0,
        end_idx
    )
    
    num_trades = trades_arr.shape[0]
    if num_trades == 0:
        return {'pnl': -999.0, 'max_dd': 1.0, 'win_rate': 0.0, 'trades': 0}
        
    # Calculate aggregate metrics for this fidelity
    pnls = trades_arr[:, 0]
    wins = trades_arr[:, 2]
    
    total_pnl = np.sum(pnls)
    win_rate = (np.sum(wins) / num_trades) * 100.0
    
    # Calculate approximate Max Drawdown from PnL series
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
    
    return {
        'pnl': float(total_pnl),
        'max_dd': float(max_dd),
        'win_rate': float(win_rate),
        'trades': num_trades
    }
