import logging
import time
from typing import Dict, Any, Optional
from core.events import SignalEvent, SignalType
from config import Config

logger = logging.getLogger("PyramidEngine")

class PyramidEngine:
    """
    📈 PYRAMID ENGINE — Asymmetric Compounding para Scalping
    
    QUÉ: Inyecta capital extra (Scale-In) a posiciones que ya están en ganancias considerables.
    POR QUÉ: Para maximizar retornos con un modelo de apalancamiento compuesto sin aumentar riesgo base.
    PARA QUÉ: Crecimiento exponencial. $13 a infinito.
    
    Reglas:
    1. Solo funciona si PnL > threshold (default 0.35%).
    2. Máximo N capas (definido en config, default 1).
    3. Tamaño inyectado: 50% del tamaño original de la posición.
    """
    
    def __init__(self):
        self.enabled = getattr(Config.Strategies, "ENABLE_PYRAMIDING", True)
        self.max_layers = getattr(Config.Strategies, "MAX_PYRAMID_SCALE_INS", 1)
        self.size_mult = 0.50
        self.cooldown_s = 60  # Al menos 60s entre pyramiding para no spam de órdenes
        self.trigger_pct = 0.35  # % de ganancia requerido
        
        # State: {trade_id: {'layers': int, 'last_pyramid_ts': float}}
        self._pyramid_state: Dict[str, Dict[str, Any]] = {}
        
    def evaluate_position(self, pos: dict, current_price: float, available_cash: float, dh=None) -> Optional[SignalEvent]:
        if not self.enabled:
            return None
            
        symbol = pos.get('symbol')
        if not symbol:
            return None
            
        pos_horizon = pos.get('horizon', 'SCALPING')
        if pos_horizon not in ("SCALPING", "MICROSCALPING"):
            return None
            
        qty = pos.get('quantity', 0.0)
        if qty == 0:
            return None
            
        entry_price = pos.get('avg_price', pos.get('entry_price', 0.0))
        if entry_price == 0:
            return None
            
        unrealized_pnl_pct = (
            ((current_price - entry_price) / entry_price) * 100
            if qty > 0
            else ((entry_price - current_price) / entry_price) * 100
        )
        
        # Requiere ganancia para activar pirámide
        if unrealized_pnl_pct < self.trigger_pct:
            return None
            
        # Check layers
        trade_id = pos.get('trade_id', f"{symbol}_{pos_horizon}")
        
        # Hibridamos estado interno con estado de la posición (si hubo cierre parcial se reinicia)
        state = self._pyramid_state.get(trade_id, {'layers': 0, 'last_pyramid_ts': 0.0})
        actual_layers = max(state['layers'], pos.get('scale_count', 0))
        
        if actual_layers >= self.max_layers:
            return None
            
        # Check cooldown
        now_ts = time.time()
        if now_ts - state['last_pyramid_ts'] < self.cooldown_s:
            return None
            
        # Check margin
        dca_qty = abs(qty) * self.size_mult
        required_margin = (dca_qty * current_price) / getattr(Config, "BINANCE_LEVERAGE", 20)
        if available_cash < required_margin:
            # Not enough margin to pyramid, we don't spam errors. Rotator might free cash.
            return None
            
        # Issue Pyramid Signal
        stype = SignalType.LONG if qty > 0 else SignalType.SHORT
        
        logger.warning(f"📈 [PYRAMID ENGINE] {symbol} {pos_horizon} PnL: {unrealized_pnl_pct:.2f}% | 💥 Firing Pyramid Scale-In (Layer {actual_layers+1})")
        
        # Update State (Optimistic)
        state['layers'] = actual_layers + 1
        state['last_pyramid_ts'] = now_ts
        self._pyramid_state[trade_id] = state
        
        return SignalEvent(
            strategy_id="PYRAMID_ENGINE",
            symbol=symbol,
            datetime=time.time(),
            signal_type=stype,
            strength=0.99,
            horizon=pos_horizon,
            metadata={
                "exit_reason": "PYRAMIDING_SCALE_IN",
                "is_pyramid": True,
                "pyramid_qty": dca_qty,
                "pyramid_layer": state['layers'],
                "original_qty": abs(qty)
            }
        )

# Singleton instance
pyramid_engine = PyramidEngine()
