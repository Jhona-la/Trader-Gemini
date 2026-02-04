"""
Safe Leverage Calculator - Optimizado para cuentas micro ($10-$100)
Evita liquidaciones y protege capital con reglas conservadoras

REGLA DE ORO: Con $12, leverage > 5x es extremadamente peligroso
"""
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LeverageConfig:
    """Configuración de leverage por fase de cuenta"""
    min_leverage: int
    max_leverage: int
    default_leverage: int
    max_drawdown_pct: float
    max_position_size_pct: float


class SafeLeverageCalculator:
    """
    Sistema de leverage adaptativo basado en:
    1. Capital disponible (account size)
    2. Volatilidad del asset (ATR)
    3. Win rate histórico
    4. Drawdown actual
    
    REGLA DE ORO: Con $12, leverage > 5x es suicida
    """
    
    # Configuraciones por fase
    PHASES = {
        'MICRO': {  # $0 - $50
            'min_leverage': 2,
            'max_leverage': 5,
            'default_leverage': 3,
            'max_drawdown_pct': 0.25,      # 25% max DD
            'max_position_size_pct': 0.30, # 30% max position
            'max_trades': 2                # Fixed: Max 2 positions for micro
        },
        'SMALL': {  # $50 - $200
            'min_leverage': 3,
            'max_leverage': 8,
            'default_leverage': 5,
            'max_drawdown_pct': 0.20,
            'max_position_size_pct': 0.25,
            'max_trades': 4
        },
        'MEDIUM': {  # $200 - $1000
            'min_leverage': 4,
            'max_leverage': 12,
            'default_leverage': 8,
            'max_drawdown_pct': 0.15,
            'max_position_size_pct': 0.20,
            'max_trades': 8
        },
        'LARGE': {  # $1000+
            'min_leverage': 5,
            'max_leverage': 20,
            'default_leverage': 10,
            'max_drawdown_pct': 0.12,
            'max_position_size_pct': 0.15,
            'max_trades': 15
        }
    }
    
    def __init__(self, initial_capital: float = 12.0, portfolio=None):
        """
        Args:
            initial_capital: Capital inicial
            portfolio: Referencia al portfolio para obtener capital actualizado
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.portfolio = portfolio
        
        # Performance tracking
        self.win_count = 0
        self.loss_count = 0
        self.consecutive_losses = 0
        
        # Safety limits
        self.LIQUIDATION_BUFFER = 0.15  # 15% buffer de liquidación
        self.MAX_RISK_PER_TRADE = 0.02  # 2% del capital
    
    def get_capital(self) -> float:
        """Obtener capital actual desde portfolio (SINGLE SOURCE OF TRUTH)"""
        if self.portfolio is not None:
            try:
                return self.portfolio.get_total_equity()
            except:
                pass
        return self.current_capital
    
    def get_phase(self, capital: Optional[float] = None) -> str:
        """Determinar fase actual de cuenta"""
        if capital is None:
            capital = self.get_capital()
            
        if capital < 50:
            return 'MICRO'
        elif capital < 200:
            return 'SMALL'
        elif capital < 1000:
            return 'MEDIUM'
        else:
            return 'LARGE'
    
    def get_max_trades(self, capital: Optional[float] = None) -> int:
        """Obtener número máximo de posiciones sugeridas para el capital actual"""
        phase = self.get_phase(capital)
        return self.PHASES[phase]['max_trades']
    
    def calculate_safe_leverage(self, atr: float, price: float, 
                                win_rate: Optional[float] = None) -> Dict:
        """
        Calcular leverage seguro basado en múltiples factores.
        
        Args:
            atr: Average True Range del asset
            price: Precio actual
            win_rate: Win rate histórico (0-1), None para usar calculado
        
        Returns:
            Dict con leverage recomendado y metadata
        """
        # Obtener capital actualizado
        capital = self.get_capital()
        
        # 1. Determinar fase
        phase = self.get_phase(capital)
        config = self.PHASES[phase]
        
        # 2. Calcular volatilidad
        # FIXED: Defensive None check for ATR
        if atr is None:
            atr = (price * 0.02) if price else 0.0
            
        atr_pct = (atr / price) * 100 if price and price > 0 else 999
        
        # 3. Calcular win rate si no se proporciona
        if win_rate is None:
            total = self.win_count + self.loss_count
            win_rate = self.win_count / total if total > 0 else 0.5
        
        # 4. Base leverage por volatilidad (CONSERVATIVE UPDATE Feb 2026)
        if atr_pct > 2.0:
            # Extrema volatilidad - SAFETY MODE
            leverage = 3
            reason = "Extreme volatility (>2%) - Capped at 3x"
        elif atr_pct > 1.0:
            # Alta volatilidad - CAUTION
            leverage = 5
            reason = "High volatility (1-2%) - Capped at 5x"
        elif atr_pct > 0.5:
            # Media volatilidad - NORMAL
            leverage = 8
            reason = "Normal volatility (0.5-1%) - 8x"
        else:
            # Baja volatilidad - AGGRESSIVE
            leverage = 10
            reason = "Low volatility (<0.5%) - 10x" # Up to 10x, clipped by max_leverage later
        
        # Clip by phase limits immediately
        leverage = min(leverage, config['max_leverage'])
        leverage = max(leverage, config['min_leverage'])
        
        # 5. Ajustar por performance
        if win_rate < 0.40:
            # Mala performance - reducir leverage 50%
            leverage = max(config['min_leverage'], int(leverage * 0.5))
            reason += " | Poor performance penalty"
        elif win_rate > 0.60:
            # Buena performance - bonus conservador
            leverage = min(config['max_leverage'], int(leverage * 1.2))
            reason += " | Good performance bonus"
        
        # 6. Ajustar por drawdown
        drawdown = (self.peak_capital - capital) / self.peak_capital if self.peak_capital > 0 else 0
        if drawdown > config['max_drawdown_pct']:
            leverage = max(config['min_leverage'], int(leverage * 0.6))
            reason += f" | Drawdown penalty ({drawdown*100:.1f}%)"
        
        # 7. Ajustar por pérdidas consecutivas
        if self.consecutive_losses >= 3:
            leverage = config['min_leverage']
            reason += f" | {self.consecutive_losses} consecutive losses"
        
        # 8. Asegurar que leverage sea al menos 1 si es válido
        if leverage > 0:
            leverage = max(1, leverage)
        
        # 9. Calcular liquidation price
        liq_price_long = self._calculate_liquidation_price(
            price, leverage, 'LONG'
        ) if leverage > 0 else 0
        liq_price_short = self._calculate_liquidation_price(
            price, leverage, 'SHORT'
        ) if leverage > 0 else 0
        
        # 10. Validar que liquidation está suficientemente lejos
        distance_to_liq = 0
        if leverage > 0 and price > 0:
            distance_to_liq_long = abs(price - liq_price_long) / price if liq_price_long > 0 else 1
            distance_to_liq_short = abs(price - liq_price_short) / price if liq_price_short > 0 else 1
            distance_to_liq = min(distance_to_liq_long, distance_to_liq_short)
            
            if distance_to_liq < self.LIQUIDATION_BUFFER:
                # Demasiado cerca de liquidación - reducir leverage
                old_leverage = leverage
                leverage = max(config['min_leverage'], int(leverage * 0.7))
                reason += f" | Liquidation too close (reduced {old_leverage}→{leverage})"
        
        return {
            'leverage': leverage,
            'reason': reason,
            'phase': phase,
            'capital': capital,
            'atr_pct': atr_pct,
            'win_rate': win_rate,
            'drawdown': drawdown,
            'consecutive_losses': self.consecutive_losses,
            'liquidation_long': liq_price_long,
            'liquidation_short': liq_price_short,
            'distance_to_liq': distance_to_liq * 100,
            'is_safe': leverage >= config['min_leverage'] and leverage > 0,
            'max_position_size_pct': config['max_position_size_pct']
        }
    
    def _calculate_liquidation_price(self, entry_price: float, 
                                     leverage: int, direction: str) -> float:
        """
        Calcular precio de liquidación (Binance Futures Isolated Margin).
        
        Formula:
        LONG:  Liq = Entry * (1 - 1/leverage + MMR)
        SHORT: Liq = Entry * (1 + 1/leverage - MMR)
        
        MMR = Maintenance Margin Rate (0.4% para la mayoría)
        """
        if leverage <= 0:
            return 0.0
        
        MMR = 0.004  # 0.4% maintenance margin
        
        if direction == 'LONG':
            liq_price = entry_price * (1 - (1 / leverage) + MMR)
        else:  # SHORT
            liq_price = entry_price * (1 + (1 / leverage) - MMR)
        
        return liq_price
    
    def calculate_position_size(self, price: float, leverage: Optional[int] = None,
                               capital: Optional[float] = None) -> Dict:
        """
        Calcular tamaño de posición seguro.
        
        Args:
            price: Precio actual del asset
            leverage: Leverage a usar (se calcula si no se proporciona)
            capital: Capital a usar (se obtiene del portfolio si no se proporciona)
        
        Returns:
            Dict con margin_required, notional_value, quantity
        """
        if capital is None:
            capital = self.get_capital()
        
        phase = self.get_phase(capital)
        config = self.PHASES[phase]
        
        if leverage is None:
            leverage = config['default_leverage']
        
        # Máximo que podemos usar del capital
        max_capital_use = capital * config['max_position_size_pct']
        
        # Notional value con leverage
        notional_value = max_capital_use * leverage
        
        # Margin requerido
        margin_required = notional_value / leverage if leverage > 0 else 0
        
        # Cantidad de asset
        quantity = notional_value / price if price > 0 else 0
        
        # Calcular riesgo por trade (stop loss 2%)
        stop_loss_distance = price * 0.02  # 2% stop
        risk_per_trade = quantity * stop_loss_distance
        risk_pct = (risk_per_trade / capital) * 100 if capital > 0 else 0
        
        return {
            'margin_required': margin_required,
            'notional_value': notional_value,
            'quantity': quantity,
            'risk_per_trade': risk_per_trade,
            'risk_pct': risk_pct,
            'max_capital_use_pct': config['max_position_size_pct'] * 100,
            'leverage_used': leverage,
            'is_safe': risk_pct <= (self.MAX_RISK_PER_TRADE * 100)
        }
    
    def update_performance(self, is_win: bool, pnl: float = 0):
        """Actualizar performance tracking"""
        if is_win:
            self.win_count += 1
            self.consecutive_losses = 0
        else:
            self.loss_count += 1
            self.consecutive_losses += 1
        
        # Actualizar capital tracking
        capital = self.get_capital()
        self.current_capital = capital
        self.peak_capital = max(self.peak_capital, capital)
    
    def update_capital(self, new_capital: float):
        """Actualizar capital tracking manualmente"""
        self.current_capital = new_capital
        self.peak_capital = max(self.peak_capital, new_capital)
    
    def get_recommendations(self) -> str:
        """Obtener recomendaciones para el trader"""
        capital = self.get_capital()
        phase = self.get_phase(capital)
        config = self.PHASES[phase]
        
        total = self.win_count + self.loss_count
        win_rate = self.win_count / max(1, total)
        
        recommendations = []
        
        # Leverage
        recommendations.append(
            f"📊 Phase: {phase} (${capital:.2f}) | "
            f"Safe leverage range: {config['min_leverage']}-{config['max_leverage']}x"
        )
        
        # Performance
        if total >= 5 and win_rate < 0.50:
            recommendations.append(
                f"⚠️ Win rate {win_rate*100:.1f}% is low. "
                f"Consider: reducing leverage, reviewing strategy"
            )
        
        # Drawdown
        drawdown = (self.peak_capital - capital) / self.peak_capital if self.peak_capital > 0 else 0
        if drawdown > 0.15:
            recommendations.append(
                f"🔴 Drawdown {drawdown*100:.1f}% is high. "
                f"Consider REDUCING position sizes"
            )
        
        # Consecutive losses
        if self.consecutive_losses >= 2:
            recommendations.append(
                f"⚠️ {self.consecutive_losses} consecutive losses. "
                f"Review strategy before next trade"
            )
        
        # Capital phase advice
        if phase == 'MICRO':
            recommendations.append(
                "💡 MICRO account: Focus on capital preservation. "
                f"Target: 2-3% per day, max {config['max_leverage']}x leverage"
            )
        
        return "\n".join(recommendations)
    
    def reset(self, initial_capital: float = 12.0):
        """Reset calculator state"""
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.win_count = 0
        self.loss_count = 0
        self.consecutive_losses = 0


# Singleton instance for global access
safe_leverage_calculator = SafeLeverageCalculator(initial_capital=15.0)
