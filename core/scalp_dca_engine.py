import logging
import time
from typing import Dict, Any, Optional
from core.events import SignalEvent, SignalType

logger = logging.getLogger("ScalpDCAEngine")

class ScalpDCAEngine:
    """
    📉 SCALP DCA ENGINE — Recovery Automático para Scalping
    A diferencia del SwingDCAEngine, este es un micro-DCA rápido
    diseñado para recuperar posiciones de scalping antes de que toquen el SL.
    
    Reglas:
    1. Solo 1 capa de DCA (vs 3 en Swing)
    2. Trigger dinámico: a mitad del camino hacia el SL (e.g., -0.175% si SL es -0.35%)
    3. Tamaño reducido: 50% del tamaño original
    4. Cooldown hiper-rápido: 30 segundos
    """
    
    def __init__(self):
        self.enabled = True
        self.max_layers = 1
        self.size_mult = 0.50
        self.cooldown_s = 30
        
        # State: {trade_id: {'layers': int, 'last_dca_ts': float}}
        self._dca_state: Dict[str, Dict[str, Any]] = {}
        
    def evaluate_position(self, pos: dict, current_price: float, available_cash: float, dh=None) -> Optional[SignalEvent]:
        if not self.enabled:
            return None
            
        symbol = pos['symbol']
        if not symbol:
            return None
            
        pos_horizon = pos['horizon']
        if pos_horizon not in ("SCALPING", "MICROSCALPING"):
            return None
            
        qty = pos['quantity']
        if qty == 0:
            return None
            
        entry_price = pos['entry_price']
        sl_pct = pos['sl_pct']  # Default 0.35% si no hay
        
        unrealized_pnl_pct = (
            ((current_price - entry_price) / entry_price) * 100
            if qty > 0
            else ((entry_price - current_price) / entry_price) * 100
        )
        
        # Trigger is at 50% of SL (e.g., if SL is 0.40%, trigger is -0.20%)
        # Note: sl_pct is typically absolute (e.g., 0.004). Pnl_pct is 100-based.
        trigger_pct = - (sl_pct * 100) * 0.50
        
        if unrealized_pnl_pct > trigger_pct:
            return None  # No en draw down suficiente
            
        # Check layers
        trade_id = pos['trade_id']
        state = self._dca_state[trade_id]
        
        if state['layers'] >= self.max_layers:
            return None
            
        # Check cooldown
        now_ts = time.time()
        if now_ts - state['last_dca_ts'] < self.cooldown_s:
            return None
            
        # Check margin (min notional $5)
        dca_qty = abs(qty) * self.size_mult
        if dca_qty * current_price < 5.0:
            return None
            
        # Issue DCA Signal
        stype = SignalType.LONG if qty > 0 else SignalType.SHORT
        
        logger.warning(f"🛡️ [SCALP-DCA] {symbol} {pos_horizon} PnL: {unrealized_pnl_pct:.2f}% | Firing Recovery DCA (Layer {state['layers']+1})")
        
        # Update State (Optimistic)
        state['layers'] += 1
        state['last_dca_ts'] = now_ts
        self._dca_state[trade_id] = state
        
        return SignalEvent(
            strategy_id="SCALP_DCA_ENGINE",
            symbol=symbol,
            datetime=time.time(),
            signal_type=stype,
            strength=0.99,
            horizon=pos_horizon,
            metadata={
                "exit_reason": "SCALP_RECOVERY_DCA",
                "dca_qty": dca_qty,
                "dca_layer": state['layers'],
                "original_qty": abs(qty)
            }
        )

# Singleton instance
scalp_dca_engine = ScalpDCAEngine()
