import numpy as np
from numba import njit
import math

# ═════════════════════════════════════════════════════════════════════════════
# ⚡ THE QUANTUM ENGINE (NUMBA @NJIT) - DUAL HORIZON INTEGRAL
# ═════════════════════════════════════════════════════════════════════════════
# Este simulador maneja concurrencia de estados: SCALPING y SWING al mismo tiempo.
# Evita pisadas de patas ("vetos internos") aislando los límites de TP/SL 
# y deduciendo la reserva de margen compartida vectorizadamente.
# ═════════════════════════════════════════════════════════════════════════════

@njit
def _run_numba_simulation(
    closes, highs, lows, opens, 
    ml_signals_scalp, tech_signals_scalp, 
    ml_signals_swing, tech_signals_swing,
    atr_pct, macro_multiplier,
    initial_capital, 
    kelly_fraction, maker_fee, taker_fee,
    leverage_scalp, base_tp_scalp, base_sl_scalp,
    leverage_swing, base_tp_swing, base_sl_swing
):
    """
    Simulador concurrente determinista de estado (Dual-Horizon State Machine)
    Returns: PnL Timeline, Trades Array (Scalp), Trades Array (Swing)
    """
    n = len(closes)
    
    capital = np.zeros(n, dtype=np.float32)
    capital[0] = initial_capital
    
    # State SCALPING
    pos_qty_scalp = 0.0
    pos_price_scalp = 0.0
    pos_side_scalp = 0  # 1 = LONG, -1 = SHORT
    tp_price_scalp = 0.0
    sl_price_scalp = 0.0
    
    # State SWING
    pos_qty_swing = 0.0
    pos_price_swing = 0.0
    pos_side_swing = 0
    tp_price_swing = 0.0
    sl_price_swing = 0.0
    
    # Trades Logging
    trades_pnl_scalp = np.zeros(n, dtype=np.float32)
    trades_pnl_swing = np.zeros(n, dtype=np.float32)
    trade_count_scalp = 0
    trade_count_swing = 0
    
    current_capital = initial_capital
    free_margin = initial_capital
    
    for i in range(1, n):
        capital[i] = current_capital
        
        c = closes[i]
        h = highs[i]
        l = lows[i]
        
        # ════════════════════════════════════════════════════
        # 1. CHECK EXITS (SCALPING)
        # ════════════════════════════════════════════════════
        if pos_side_scalp != 0:
            exit_triggered = False
            exit_price = 0.0
            
            if pos_side_scalp == 1:
                if l <= sl_price_scalp:
                    exit_triggered = True
                    exit_price = sl_price_scalp
                elif h >= tp_price_scalp:
                    exit_triggered = True
                    exit_price = tp_price_scalp
            elif pos_side_scalp == -1:
                if h >= sl_price_scalp:
                    exit_triggered = True
                    exit_price = sl_price_scalp
                elif l <= tp_price_scalp:
                    exit_triggered = True
                    exit_price = tp_price_scalp
                    
            if exit_triggered:
                if pos_side_scalp == 1:
                    gross_pnl = (exit_price - pos_price_scalp) * pos_qty_scalp
                else:
                    gross_pnl = (pos_price_scalp - exit_price) * pos_qty_scalp
                    
                fee = (exit_price * pos_qty_scalp) * taker_fee
                net_pnl = gross_pnl - fee
                
                current_capital += net_pnl
                free_margin += (pos_price_scalp * pos_qty_scalp) / leverage_scalp + net_pnl
                trades_pnl_scalp[trade_count_scalp] = net_pnl
                trade_count_scalp += 1
                
                pos_side_scalp = 0
                pos_qty_scalp = 0.0

        # ════════════════════════════════════════════════════
        # 2. CHECK EXITS (SWING)
        # ════════════════════════════════════════════════════
        if pos_side_swing != 0:
            exit_triggered = False
            exit_price = 0.0
            
            if pos_side_swing == 1:
                if l <= sl_price_swing:
                    exit_triggered = True
                    exit_price = sl_price_swing
                elif h >= tp_price_swing:
                    exit_triggered = True
                    exit_price = tp_price_swing
            elif pos_side_swing == -1:
                if h >= sl_price_swing:
                    exit_triggered = True
                    exit_price = sl_price_swing
                elif l <= tp_price_swing:
                    exit_triggered = True
                    exit_price = tp_price_swing
                    
            if exit_triggered:
                if pos_side_swing == 1:
                    gross_pnl = (exit_price - pos_price_swing) * pos_qty_swing
                else:
                    gross_pnl = (pos_price_swing - exit_price) * pos_qty_swing
                    
                fee = (exit_price * pos_qty_swing) * taker_fee
                net_pnl = gross_pnl - fee
                
                current_capital += net_pnl
                free_margin += (pos_price_swing * pos_qty_swing) / leverage_swing + net_pnl
                trades_pnl_swing[trade_count_swing] = net_pnl
                trade_count_swing += 1
                
                pos_side_swing = 0
                pos_qty_swing = 0.0

        # ════════════════════════════════════════════════════
        # 3. CHECK ENTRIES (SCALPING)
        # ════════════════════════════════════════════════════
        if pos_side_scalp == 0 and free_margin > 0:
            signal = 0
            if ml_signals_scalp[i] == 1 and tech_signals_scalp[i] >= 0:
                signal = 1
            elif ml_signals_scalp[i] == -1 and tech_signals_scalp[i] <= 0:
                signal = -1
                
            if signal != 0:
                volatility_scaler = min(1.5, 0.005 / max(0.001, atr_pct[i]/100.0))
                # Macro Multiplier ajusta agresividad
                size_pct = min(0.95, kelly_fraction * volatility_scaler * macro_multiplier[i])
                
                margin_reserved = free_margin * size_pct
                notional = margin_reserved * leverage_scalp
                
                pos_price_scalp = c 
                fee = notional * taker_fee
                
                # Update Capital and free margin directly on entry
                current_capital -= fee 
                free_margin -= margin_reserved + fee
                
                pos_qty_scalp = notional / pos_price_scalp
                pos_side_scalp = signal
                
                tp_dist = c * (base_tp_scalp * max(1.0, (atr_pct[i]/100.0) / 0.005))
                sl_dist = c * (base_sl_scalp * max(1.0, (atr_pct[i]/100.0) / 0.005))
                
                if pos_side_scalp == 1:
                    tp_price_scalp = c + tp_dist
                    sl_price_scalp = c - sl_dist
                else:
                    tp_price_scalp = c - tp_dist
                    sl_price_scalp = c + sl_dist

        # ════════════════════════════════════════════════════
        # 4. CHECK ENTRIES (SWING)
        # ════════════════════════════════════════════════════
        if pos_side_swing == 0 and free_margin > 0:
            signal = 0
            if ml_signals_swing[i] == 1 and tech_signals_swing[i] >= 0:
                signal = 1
            elif ml_signals_swing[i] == -1 and tech_signals_swing[i] <= 0:
                signal = -1
                
            if signal != 0:
                volatility_scaler = min(1.5, 0.005 / max(0.001, atr_pct[i]/100.0))
                # Swing tiene un kelly fraccionado al 50% de agresividad vs Scalp
                size_pct = min(0.95, (kelly_fraction*0.5) * volatility_scaler * macro_multiplier[i])
                
                margin_reserved = free_margin * size_pct
                notional = margin_reserved * leverage_swing
                
                pos_price_swing = c 
                fee = notional * taker_fee
                
                current_capital -= fee 
                free_margin -= margin_reserved + fee
                
                pos_qty_swing = notional / pos_price_swing
                pos_side_swing = signal
                
                tp_dist = c * (base_tp_swing * max(1.0, (atr_pct[i]/100.0) / 0.005))
                sl_dist = c * (base_sl_swing * max(1.0, (atr_pct[i]/100.0) / 0.005))
                
                if pos_side_swing == 1:
                    tp_price_swing = c + tp_dist
                    sl_price_swing = c - sl_dist
                else:
                    tp_price_swing = c - tp_dist
                    sl_price_swing = c + sl_dist
                    
    return capital, trades_pnl_scalp[:trade_count_scalp], trades_pnl_swing[:trade_count_swing]

class NumbaEngine:
    @staticmethod
    def run_simulation(df, ml_sigs_scalp, tech_sigs_scalp, ml_sigs_swing, tech_sigs_swing, macro_multiplier, config):
        closes = np.ascontiguousarray(df['close'].values, dtype=np.float32)
        highs = np.ascontiguousarray(df['high'].values, dtype=np.float32)
        lows = np.ascontiguousarray(df['low'].values, dtype=np.float32)
        opens = np.ascontiguousarray(df['open'].values, dtype=np.float32)
        atr_pct = np.ascontiguousarray(df['atr_pct'].values, dtype=np.float32)
        macro_mult = np.ascontiguousarray(macro_multiplier, dtype=np.float32)
        
        ml_s = np.ascontiguousarray(ml_sigs_scalp, dtype=np.int8)
        ts_s = np.ascontiguousarray(tech_sigs_scalp, dtype=np.int8)
        
        ml_sw = np.ascontiguousarray(ml_sigs_swing, dtype=np.int8)
        ts_sw = np.ascontiguousarray(tech_sigs_swing, dtype=np.int8)
        
        capital_curve, trades_scalp, trades_swing = _run_numba_simulation(
            closes, highs, lows, opens,
            ml_s, ts_s,
            ml_sw, ts_sw,
            atr_pct, macro_mult,
            initial_capital=config.get('initial_capital', 13.0),
            kelly_fraction=config.get('kelly_fraction', 0.19),
            maker_fee=config.get('maker_fee', 0.0002),
            taker_fee=config.get('taker_fee', 0.00075),
            leverage_scalp=config.get('leverage_scalp', 50.0),
            base_tp_scalp=config.get('base_tp_scalp', 0.0076),
            base_sl_scalp=config.get('base_sl_scalp', 0.0162),
            leverage_swing=config.get('leverage_swing', 30.0),
            base_tp_swing=config.get('base_tp_swing', 0.1732),
            base_sl_swing=config.get('base_sl_swing', 0.0313)
        )
        return capital_curve, trades_scalp, trades_swing
