import logging
import datetime
from typing import Dict, Optional, Tuple, Any

logger = logging.getLogger("TraderGemini")

class PositionLifecycleManager:
    """
    🧠 [MUTACIÓN 25] CEREBRO CENTRAL DE POSICIONES (Dynamic Position Lifecycle)
    Erradica los Time-Stops rígidos y gestiona el ciclo de vida de los trades
    mediante estructura de mercado y momentum en lugar de un reloj.
    """
    
    def __init__(self):
        self._states: Dict[str, str] = {}
        # Historial de máxima excursión a favor para expansión elástica de TPs
        self._mfe_history: Dict[str, float] = {}

    def evaluate_health(self, pos: dict, market_data: dict, current_price: float, now: datetime.datetime) -> Tuple[str, Optional[str]]:
        """
        Evalúa la inercia y estructura de una posición en tiempo real.
        Retorna (Action, Reason), donde Action puede ser:
        - "HOLD": Mantener la posición viva.
        - "EXIT": Cerrar inmediatamente por rotura de estructura o verdadero estado Zombie.
        - "SHIFT_UP": Mover TP más arriba (Expansión Elástica por Momentum).
        """
        pos_id = pos.get('trade_id') or pos.get('symbol')
        if not pos_id:
            return "HOLD", None

        # Datos base de la posición
        entry_price = pos.get('avg_price', 0.0)
        if entry_price <= 0:
            return "HOLD", None

        qty = pos.get('quantity', 0.0)
        direction = 1 if qty > 0 else -1
        
        # PnL bruto porcentual
        pnl_pct = ((current_price - entry_price) / entry_price) * direction * 100.0
        
        # Actualizar MFE (Maximum Favorable Excursion)
        current_mfe = self._mfe_history.get(pos_id, 0.0)
        if pnl_pct > current_mfe:
            self._mfe_history[pos_id] = pnl_pct
            current_mfe = pnl_pct

        # Extraer métricas de mercado
        tick_volatility = market_data.get('tick_volatility', 0.0)
        vpin = market_data.get('toxicity_index', 0.5)
        obi_velocity = market_data.get('obi_velocity', 0.0)
        
        # Calcular tiempo abierto
        entry_time = pos.get('entry_time', now)
        try:
            duration_mins = (now - entry_time).total_seconds() / 60.0
        except TypeError:
            duration_mins = 0.0

        # =========================================================================
        # 1. ANTI-ZOMBIE MATEMÁTICO (Trade Perdido + Sin Momentum)
        # =========================================================================
        # En lugar de matar al minuto 60 a ciegas, matamos si y solo si:
        # PnL no justifica estar dentro (< 0.10%) Y han pasado > 45 minutos Y el mercado está muerto (TickVol < 0.0005)
        is_losing_or_flat = pnl_pct <= 0.10
        
        # FIX PARA BACKTESTING: Si no hay order_flow reales, evitamos falsos positivos de mercado muerto
        has_order_flow = 'tick_volatility' in market_data
        if not has_order_flow:
            is_stagnant_market = False  # Asumimos que no está muerto si no tenemos los datos
        else:
            is_stagnant_market = tick_volatility < 0.0005 and abs(obi_velocity) < 0.1
        
        if duration_mins > 45.0 and is_losing_or_flat and is_stagnant_market:
            logger.warning(f"🧟 [LIFECYCLE] {pos_id} marcado como TRUE ZOMBIE (Dur:{duration_mins:.1f}m, PnL:{pnl_pct:.2f}%, Vol:{tick_volatility:.6f}). Ejecutando.")
            return "EXIT", "TRUE_ZOMBIE_STRUCTURAL"

        # Si el trade está estancado en tiempo, PERO es un trade ganador (> 0.20%), NO es zombie.
        if duration_mins > 90.0 and pnl_pct > 0.20:
            self._states[pos_id] = "STAGNANT_WINNING"
            # Si el VPIN o OBI de repente tira con fuerza en contra, cerramos.
            if direction == 1 and vpin > 0.8: # Long pero flujo de ventas agresivo
                return "EXIT", "WINNER_MOMENTUM_REVERSAL"
            if direction == -1 and vpin < 0.2: # Short pero flujo de compras agresivo
                return "EXIT", "WINNER_MOMENTUM_REVERSAL"

        # =========================================================================
        # 2. DYNAMIC HORIZON SHIFT (Expansión Elástica de Ganancias)
        # =========================================================================
        # Si un trade de MICROSCALPING agarra tendencia, no lo estrangulamos con un TP de 0.5%.
        horizon = pos.get('horizon', 'SCALPING')
        if horizon == "MICROSCALPING" and current_mfe > 0.40:
            # Momentum violento a favor
            if direction == 1 and obi_velocity > 0.5: # Compras fuertes
                self._states[pos_id] = "ACCELERATING"
                return "SHIFT_UP", "MOMENTUM_EXPANSION"
            if direction == -1 and obi_velocity < -0.5: # Ventas fuertes
                self._states[pos_id] = "ACCELERATING"
                return "SHIFT_UP", "MOMENTUM_EXPANSION"

        self._states[pos_id] = "HEALTHY"
        return "HOLD", None

    def get_state(self, pos_id: str) -> str:
        return self._states.get(pos_id, "UNKNOWN")
    
    def clear_state(self, pos_id: str):
        self._states.pop(pos_id, None)
        self._mfe_history.pop(pos_id, None)

lifecycle_manager = PositionLifecycleManager()
