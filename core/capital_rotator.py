import logging
import time
from typing import Optional, Dict, Any, List
from core.events import SignalEvent, SignalType

logger = logging.getLogger("CapitalRotator")

class CapitalRotator:
    """
    ⚡ CAPITAL ROTATOR — Hyper-Rotational Capital Matrix
    
    QUÉ: Oráculo de costo de oportunidad que asesina posiciones Zombie para liberar margen.
    POR QUÉ: Si llega una señal de altísimo EV (Expected Value) pero la cuenta de $13 no tiene margen,
            es imperativo liquidar posiciones estancadas para inyectar el capital donde importa.
    PARA QUÉ: Maximizar la rotación y alcanzar 100% ROI cada 3 días.
    """
    
    def __init__(self):
        self.min_trade_age_s = 60  # Trade must be at least 60s old to be considered a zombie
        self.max_zombie_pnl = 0.001  # 0.10% profit is still considered zombie if moving slow
        self.min_confidence_to_rotate = 0.85  # Only rotate for VERY high confidence signals
        self.enabled = True
        
    def evaluate_opportunity(self, incoming_signal: SignalEvent, portfolio, required_margin: float) -> Optional[SignalEvent]:
        if not self.enabled or not portfolio:
            return None
            
        confidence = getattr(incoming_signal, "confidence", 0.0)
        if confidence < self.min_confidence_to_rotate:
            return None
            
        horizon = getattr(incoming_signal, "horizon", "SCALPING")
        if horizon not in ("SCALPING", "MICROSCALPING"):
            # Rotator is only extremely aggressive for Scalping
            return None
            
        # Inspect Virtual Ledger
        candidates = []
        now_ts = time.time()
        
        for v_key, pos in portfolio.virtual_ledger.items():
            qty = pos['quantity']
            if qty == 0:
                continue
                
            pos_horizon = pos['horizon']
            if pos_horizon != horizon:
                continue
                
            # If the signal is for the same symbol, don't kill it (might be a scale-in)
            symbol = pos['symbol']
            if symbol == incoming_signal.symbol:
                continue
                
            entry_ts = pos['entry_time']
            if not entry_ts:
                continue
                
            if hasattr(entry_ts, 'timestamp'):
                entry_ts = entry_ts.timestamp()
                
            age_s = now_ts - entry_ts
            if age_s < self.min_trade_age_s:
                continue
                
            entry_price = pos['avg_price']
            current_price = pos['current_price']
            
            pnl_pct = (current_price - entry_price) / entry_price if qty > 0 else (entry_price - current_price) / entry_price
            
            if pnl_pct < self.max_zombie_pnl:
                # Calculate EV heuristic: pnl_pct / age_s
                ev_velocity = pnl_pct / age_s if age_s > 0 else 0
                candidates.append((v_key, symbol, pos_horizon, age_s, pnl_pct, ev_velocity, abs(qty) * current_price))
                
        if not candidates:
            return None
            
        # Sort candidates by lowest EV velocity (the most stagnant ones)
        candidates.sort(key=lambda x: x[5])
        
        # Select the absolute worst zombie
        worst_zombie = candidates[0]
        z_symbol = worst_zombie[1]
        z_horizon = worst_zombie[2]
        
        logger.warning(
            f"⚡ [CAPITAL ROTATOR] High EV Signal on {incoming_signal.symbol} (Conf: {confidence:.2f}) "
            f"requires margin. ASSASSINATING Zombie Trade: {z_symbol} {z_horizon} "
            f"(Age: {worst_zombie[3]:.0f}s, PnL: {worst_zombie[4]*100:.3f}%)"
        )
        
        return SignalEvent(
            strategy_id="CAPITAL_ROTATOR",
            symbol=z_symbol,
            datetime=time.time(),
            signal_type=SignalType.EXIT,
            strength=1.0,
            horizon=z_horizon,
            metadata={
                "exit_reason": "CAPITAL_ROTATION_ZOMBIE_ASSASSINATION",
                "rotated_into": incoming_signal.symbol
            }
        )

# Singleton instance
capital_rotator = CapitalRotator()
