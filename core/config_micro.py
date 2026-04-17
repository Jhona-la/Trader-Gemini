"""
Configuración optimizada para micro cuentas en futures
"""
from dataclasses import dataclass

@dataclass
class MicroFuturesConfig:
    # Gestión de riesgo
    MAX_LEVERAGE: int = 10
    RISK_PER_TRADE: float = 0.03
    MAX_DAILY_RISK: float = 0.10
    MIN_NOTIONAL: float = 5.0
    
    # Optimización de costos
    FEE_OPTIMIZATION: bool = True
    PREFER_MAKER_ORDERS: bool = True
    MAX_FEE_RATIO: float = 0.35
    
    # Estrategias adaptadas
    SCALPING_MIN_TARGET: float = 0.015
    SCALPING_MAX_DURATION: int = 30
    SCALPING_MAX_TRADES_HOUR: int = 4
    
    SWING_MIN_TARGET: float = 0.035
    SWING_MAX_DURATION: int = 48
    SWING_MAX_POSITIONS: int = 2
    
    # Protecciones
    LIQUIDATION_PROTECTION: bool = True
    AUTO_REDUCE_LEVERAGE: bool = True
    EMERGENCY_STOP_LOSS: float = 0.02
