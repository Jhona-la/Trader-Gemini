from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

@dataclass(frozen=True, slots=True, kw_only=True)
class TradeIntent:
    """
    Representa la "intención" bruta generada por una Estrategia o el motor IA (Sophia).
    Habla sobre QUÉ hacer y POR QUÉ, pero NO sobre CÓMO ejecutarlo.
    """
    symbol: str
    direction: str  # 'LONG', 'SHORT', 'EXIT'
    confidence: float  # 0.0 to 1.0
    expected_mfe: float  # Expected Maximum Favorable Excursion (Take Profit base)
    expected_mae: float  # Expected Maximum Adverse Excursion (Stop Loss base)
    horizon: str  # 'SCALPING', 'SWING'
    regime_compatibility: float  # 0.0 to 1.0 (Qué tan alineada está la señal con el régimen)
    liquidity_score: float  # 0.0 to 1.0 (Profundidad del libro para evitar slippage)
    
    # Metadatos del emisor
    strategy_id: str
    timestamp_ns: int
    metadata: Optional[Dict[str, Any]] = None

@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionPlan:
    """
    Representa el plan táctico emitido por el Meta-Coordinator.
    Traduce un TradeIntent en un plan ejecutable que el RiskManager y Executor deben seguir estrictamente.
    """
    intent_id: str  # Link al TradeIntent original
    symbol: str
    direction: str
    entry_type: str  # 'MARKET', 'LIMIT', 'TWAP', etc.
    capital_allocation: float  # Monto en USD o % de equity a arriesgar
    tp_structure: List[float]  # Puntos de Take Profit en porcentaje de precio [0.5, 1.0, 1.5]
    sl_structure: float  # Porcentaje de Stop Loss inicial
    trailing_logic: str  # 'TIGHT', 'STRUCTURED', 'MEAN_REV'
    max_hold_seconds: int  # Time-to-Live (TTL) de la operación
    
    timestamp_ns: int

@dataclass(slots=True, kw_only=True)
class PositionState:
    """
    Representa la realidad actual de una posición abierta en el SSOT.
    Es un objeto mutable que el PortfolioState actualiza por tick.
    """
    symbol: str
    direction: str
    quantity: float
    entry_price: float
    current_price: float
    horizon: str = "UNKNOWN"
    
    pnl: float = 0.0
    pnl_pct: float = 0.0
    
    high_water_mark: float = 0.0
    low_water_mark: float = 0.0
    
    time_in_trade_ms: int = 0
    health_score: float = 1.0  # 1.0 (Sana), <0.5 (En peligro), 0.0 (Liquidable)
    hazard_rate: float = 0.0   # Probabilidad instantánea de reversión en contra
    
    # Vinculación al plan original
    execution_plan_id: str = ""
    
    def update(self, current_price: float, timestamp_ms: int):
        self.current_price = current_price
        
        # MFE / MAE Updates
        if self.direction == 'LONG':
            self.high_water_mark = max(self.high_water_mark, current_price)
            self.low_water_mark = min(self.low_water_mark, current_price)
            self.pnl = (current_price - self.entry_price) * self.quantity
        elif self.direction == 'SHORT':
            self.high_water_mark = min(self.high_water_mark, current_price)
            self.low_water_mark = max(self.low_water_mark, current_price)
            self.pnl = (self.entry_price - current_price) * self.quantity
            
        self.pnl_pct = (self.pnl / (self.entry_price * self.quantity)) if self.quantity > 0 else 0.0
        self.time_in_trade_ms += timestamp_ms
