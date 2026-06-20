"""
Position Cleaner - Sistema Centralizado de Dust Detection
Evita posiciones residuales que consumen margen sin valor

Este módulo resuelve la inconsistencia en la detección de posiciones "dust"
que existía en portfolio.py (dos métodos diferentes)
"""
from typing import Dict, Optional
from decimal import Decimal, ROUND_DOWN
import logging

logger = logging.getLogger(__name__)


class PositionCleaner:
    """
    Gestión profesional de posiciones residuales (dust).
    
    Criterios para considerar dust:
    1. Valor de posición < $1 USD
    2. Valor de posición < 0.1% del capital total
    3. Cantidad < minimum notional del exchange
    """
    
    def __init__(self, min_position_value: float = 1.0):
        """
        Args:
            min_position_value: Valor mínimo en USD para mantener posición
        """
        self.MIN_POSITION_VALUE = min_position_value
        self.MIN_POSITION_PCT = 0.001  # 0.1% del capital
        
        # Exchange-specific minimums (Binance)
        self.EXCHANGE_MINIMUMS = {
            'BINANCE_SPOT': 10.0,      # $10 notional mínimo
            'BINANCE_FUTURES': 5.0,     # $5 notional mínimo
        }
        
        # Statistics
        self.dust_cleaned_count = 0
        self.dust_cleaned_value = 0.0
    
    def is_dust(self, quantity: float, current_price: float, 
                total_capital: Optional[float] = None,
                exchange: str = 'BINANCE_FUTURES') -> tuple:
        """
        Determinar si una posición es dust.
        
        Returns:
            (is_dust: bool, reason: str)
        """
        # Validación básica
        if quantity == 0:
            return True, "Zero quantity"
        
        if current_price <= 0:
            return False, "Invalid price - cannot determine"
        
        # 1. VALUE-BASED CHECK (Primary)
        position_value = abs(quantity) * current_price
        
        if position_value < self.MIN_POSITION_VALUE:
            return True, f"Value ${position_value:.4f} < MIN ${self.MIN_POSITION_VALUE}"
        
        # 2. CAPITAL PERCENTAGE CHECK (Secondary)
        if total_capital and total_capital > 0:
            position_pct = position_value / total_capital
            if position_pct < self.MIN_POSITION_PCT:
                return True, f"Position {position_pct*100:.3f}% < MIN {self.MIN_POSITION_PCT*100}%"
        
        # 3. EXCHANGE MINIMUM CHECK (Tertiary)
        exchange_min = self.EXCHANGE_MINIMUMS.get(exchange, 5.0)
        if position_value < exchange_min:
            return True, f"Value ${position_value:.2f} < Exchange MIN ${exchange_min}"
        
        return False, "OK"
    
    def clean_position(self, symbol: str, position: Dict) -> Dict:
        """
        Limpiar posición si es dust.
        
        Args:
            symbol: Trading pair
            position: Position dict con keys: quantity, current_price, avg_price
        
        Returns:
            Cleaned position dict
        """
        quantity = position['quantity']
        current_price = position['current_price']
        
        # Skip si ya está en 0
        if quantity == 0:
            return position
        
        # Check si es dust
        is_dust_pos, reason = self.is_dust(
            quantity=quantity,
            current_price=current_price,
            total_capital=None,  # Se puede pasar desde portfolio
            exchange='BINANCE_FUTURES'
        )
        
        if is_dust_pos:
            position_value = abs(quantity) * current_price
            
            logger.info(
                f"🧹 Cleaning DUST position: {symbol} | "
                f"Qty: {quantity:.8f} | Value: ${position_value:.4f} | "
                f"Reason: {reason}"
            )
            
            # Clean position
            position['quantity'] = 0
            position['high_water_mark'] = 0
            position['low_water_mark'] = 0
            position['stop_distance'] = 0
            
            # Statistics
            self.dust_cleaned_count += 1
            self.dust_cleaned_value += position_value
        
        return position
    
    def clean_all_positions(self, positions: Dict[str, Dict], 
                           total_capital: float) -> Dict[str, Dict]:
        """
        Limpiar todas las posiciones dust de un portfolio.
        
        Args:
            positions: Dict de posiciones {symbol: position_dict}
            total_capital: Capital total del portfolio
        
        Returns:
            Cleaned positions dict
        """
        cleaned_positions = {}
        dust_found = []
        
        for symbol, position in positions.items():
            quantity = position['quantity']
            current_price = position['current_price']
            
            # Skip empty positions
            if quantity == 0:
                cleaned_positions[symbol] = position
                continue
            
            # Check dust with capital context
            is_dust_pos, reason = self.is_dust(
                quantity=quantity,
                current_price=current_price,
                total_capital=total_capital,
                exchange='BINANCE_FUTURES'
            )
            
            if is_dust_pos:
                position_value = abs(quantity) * current_price
                dust_found.append({
                    'symbol': symbol,
                    'quantity': quantity,
                    'value': position_value,
                    'reason': reason
                })
                
                # Clean
                position = self.clean_position(symbol, position)
            
            cleaned_positions[symbol] = position
        
        # Report
        if dust_found:
            total_dust_value = sum(d['value'] for d in dust_found)
            logger.info(
                f"🧹 DUST CLEANING REPORT: {len(dust_found)} positions cleaned | "
                f"Total value: ${total_dust_value:.4f}"
            )
            for d in dust_found:
                logger.debug(f"   - {d['symbol']}: ${d['value']:.4f} ({d['reason']})")
        
        return cleaned_positions
    
    def adjust_quantity_to_notional(self, quantity: float, price: float, 
                                    min_notional: float = 5.0,
                                    round_down: bool = True) -> float:
        """
        Ajustar cantidad para cumplir con notional mínimo.
        
        Útil al crear órdenes.
        """
        if price <= 0:
            return 0.0
        
        current_notional = quantity * price
        
        if current_notional < min_notional:
            # Calcular cantidad mínima necesaria
            required_quantity = min_notional / price
            
            # Round according to preference
            from decimal import ROUND_UP
            if round_down:
                # Para sells, round down (conservador)
                return float(Decimal(str(required_quantity)).quantize(
                    Decimal('0.00000001'), rounding=ROUND_DOWN
                ))
            else:
                # Para buys, ROUND_UP estricto para sobrepasar siempre MIN_NOTIONAL
                return float(Decimal(str(required_quantity)).quantize(
                    Decimal('0.00000001'), rounding=ROUND_UP
                ))
        
        return quantity
    
    async def clean_orphan_orders(self, executor, symbol: str) -> bool:
        """
        Orphan Order Cleaner - Cancela órdenes pendientes en Binance.
        Invocado tras el cierre total de una posición local, para liberar margen retenido.
        """
        try:
            if not executor or not hasattr(executor, 'exchange'):
                return False
                
            logger.info(f"🧹 [ORPHAN CLEANER] Cancelando órdenes pendientes para {symbol} (Liberación Margen Micro-Cuenta)")
            
            import asyncio
            if hasattr(executor.exchange, 'cancel_all_orders'):
                if asyncio.iscoroutinefunction(executor.exchange.cancel_all_orders):
                    await executor.exchange.cancel_all_orders(symbol)
                else:
                    await asyncio.to_thread(executor.exchange.cancel_all_orders, symbol)
                return True
            return False
        except Exception as e:
            logger.error(f"⚠️ Failed to clean orphan orders for {symbol}: {e}")
            return False
    
    def get_statistics(self) -> dict:
        """Obtener estadísticas de limpieza"""
        return {
            'total_cleaned': self.dust_cleaned_count,
            'total_value_cleaned': self.dust_cleaned_value,
            'avg_dust_value': self.dust_cleaned_value / max(1, self.dust_cleaned_count)
        }
    
    def reset_statistics(self):
        """Reset statistics counters"""
        self.dust_cleaned_count = 0
        self.dust_cleaned_value = 0.0


# Singleton instance for global access
position_cleaner = PositionCleaner(min_position_value=1.0)
