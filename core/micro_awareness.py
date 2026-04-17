import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class MicroAccountConfig:
    # Configuración específica para micro cuentas
    INITIAL_BALANCE: float = 13.0
    MIN_NOTIONAL: float = 5.0  # Mínimo de Binance Futures
    FEE_RATE_TAKER: float = 0.0006  # 0.06%
    FEE_RATE_MAKER: float = 0.0004  # 0.04%
    LEVERAGE: int = 10
    RISK_PER_TRADE: float = 0.03  # 3%
    MAX_DAILY_DRAWDOWN: float = 0.005  # 0.5%

class MicroAccountAwareness:
    def __init__(self, config: MicroAccountConfig = None):
        self.config = config or MicroAccountConfig()
        self.balance = self.config.INITIAL_BALANCE
        self.last_sync_time = 0
        self.is_synced = False
        
    def update_balance(self, new_balance: float):
        """QUÉ: Actualiza el saldo interno con datos reales del Exchange."""
        import time
        self.balance = new_balance
        self.last_sync_time = time.time()
        self.is_synced = True
        
    def calculate_viable_trade_size(self, symbol: str, current_price: float) -> Tuple[float, bool]:
        """
        QUÉ: Calcula el tamaño de posición basado en el saldo REAL.
        POR QUÉ: Con $13, un error de $1 (fees) es el 7.7% del capital.
        """
        risk_amount = self.balance * self.config.RISK_PER_TRADE
        base_size = (risk_amount * self.config.LEVERAGE) / current_price
        notional_value = base_size * current_price
        
        # Ajuste por Mínimo Notional de Binance ($5 USDT)
        if notional_value < self.config.MIN_NOTIONAL:
            # Forzamos el tamaño al mínimo notional para evitar reyecciones
            adjusted_size = self.config.MIN_NOTIONAL / current_price
            return adjusted_size, True
            
        return base_size, False
    
    def calculate_breakeven_threshold(self, size: float, entry_price: float) -> float:
        """Calcula el movimiento de precio necesario para cubrir fees"""
        entry_fee = size * entry_price * self.config.FEE_RATE_TAKER
        exit_fee = size * entry_price * self.config.FEE_RATE_TAKER
        total_fees = entry_fee + exit_fee
        return total_fees / (size * entry_price)
    
    def is_trade_viable(self, symbol: str, entry_price: float, target_profit: float) -> Tuple[bool, str]:
        """Determina si un trade es económicamente viable"""
        size, adjusted = self.calculate_viable_trade_size(symbol, entry_price)
        breakeven = self.calculate_breakeven_threshold(size, entry_price)
        
        if target_profit < breakeven * 1.5:
            return False, f"Target muy pequeño. Breakeven: {breakeven:.4f}"
            
        return True, "Trade viable"
