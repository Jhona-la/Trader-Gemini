import logging
import numpy as np
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("PMB-Simulator")

@dataclass
class FillResult:
    filled_price: float
    filled_pct: float
    filled_notional: float
    fee_usdt: float
    funding_cost_usdt: float
    execution_latency_ms: float
    fill_quality_score: float

class RealisticExecutionSimulator:
    """
    AXIOMA: GAP 2 - EJECUCIÓN IDEALIZADA
    Simula la ejecución de órdenes con imperfecciones reales (Slippage, Latencia, Fills parciales).
    """
    
    def __init__(self):
        # Latencias base por horizonte (Media, Std)
        self.latency_profiles = {
            'MICRO': (5.0, 2.0),
            'SCALP': (50.0, 20.0),
            'SWING': (200.0, 50.0)
        }
        
    def _simulate_latency(self, horizon: str) -> float:
        mu, sigma = self.latency_profiles[horizon]
        return max(1.0, np.random.normal(mu, sigma))
        
    def _get_price_after_latency(self, current_price: float, latency_ms: float, direction: str, volatility: float = 0.0001) -> float:
        """
        Simula el movimiento de precio durante los milisegundos de latencia.
        """
        # Un modelo muy simplificado de Random Walk para unos milisegundos
        drift = np.random.normal(0, volatility * (latency_ms / 1000.0))
        return current_price * (1.0 + drift)
        
    def _calculate_slippage(self, order_size_usdt: float, asset: str, horizon: str, order_type: str) -> float:
        """
        Calcula el slippage en función del tamaño de orden y liquidez.
        """
        if order_type == 'LIMIT':
            return 0.0 # Las limit no tienen slippage en precio (solo riesgo de no ejecución)
            
        # Simulación de Market Order slippage
        base_slippage = 0.0001 # 1 bps base
        if asset == 'BTC' or asset == 'ETH':
            multiplier = 1.0
        else:
            multiplier = 3.0 # Altcoins tienen menos liquidez
            
        size_impact = (order_size_usdt / 100000.0) * 0.0005 # 5 bps por cada 100k
        
        return base_slippage * multiplier + size_impact
        
    def _calculate_fill_probability(self, order_type: str, spread: float, timeout_seconds: int) -> float:
        """
        Probabilidad de fill de orden Limit. Market es siempre 1.0.
        """
        if order_type == 'MARKET':
            return 1.0
            
        # Limit Orders
        # A mayor spread o timeout, mayor probabilidad
        prob = 0.5 + (timeout_seconds / 60.0) * 0.1
        return min(0.99, prob)
        
    def simulate_fill(self, order: Any, market_state: Any) -> FillResult:
        """
        Simula el fill de una orden.
        order.horizon, order.direction, order.notional_value, order.asset, order.order_type
        market_state.mid_price, market_state.spread
        """
        # Extraer variables con default (Mocking real attributes if not present)
        horizon = getattr(order, 'horizon', 'SCALP')
        direction = getattr(order, 'direction', 'LONG')
        notional = getattr(order, 'notional_value', 1000.0)
        asset = getattr(order, 'asset', 'BTC')
        order_type = getattr(order, 'order_type', 'MARKET')
        timeout = getattr(order, 'timeout_seconds', 60)
        
        mid_price = getattr(market_state, 'mid_price', 50000.0)
        spread = getattr(market_state, 'spread', 0.0001)
        
        # 1. Latencia
        latency_ms = self._simulate_latency(horizon)
        
        # 2. Precio tras latencia
        price_at_fill = self._get_price_after_latency(mid_price, latency_ms, direction)
        
        # 3. Slippage
        slippage_pct = self._calculate_slippage(notional, asset, horizon, order_type)
        if direction == 'LONG':
            final_price = price_at_fill * (1.0 + slippage_pct)
        else:
            final_price = price_at_fill * (1.0 - slippage_pct)
            
        # 4. Fill Parcial
        fill_prob = self._calculate_fill_probability(order_type, spread, timeout)
        
        if order_type == 'LIMIT':
            # Simulando Beta distribution para fill parcial
            actual_fill_pct = min(1.0, np.random.beta(fill_prob * 10, (1 - fill_prob) * 10))
        else:
            actual_fill_pct = 1.0
            
        # 5. Fees
        if order_type == 'MARKET' or (actual_fill_pct == 1.0 and final_price != order.price if hasattr(order, 'price') else False):
            fee_rate = 0.0004 # Taker
        else:
            fee_rate = 0.0002 # Maker
            
        filled_notional = notional * actual_fill_pct
        fee_usdt = filled_notional * fee_rate
        
        # 6. Funding Cost (mock 0.01% por hold de 8h)
        funding_cost = filled_notional * 0.0001 * (getattr(order, 'expected_hold_hours', 8) / 8.0)
        
        return FillResult(
            filled_price=final_price,
            filled_pct=actual_fill_pct,
            filled_notional=filled_notional,
            fee_usdt=fee_usdt,
            funding_cost_usdt=funding_cost,
            execution_latency_ms=latency_ms,
            fill_quality_score=1.0 - slippage_pct * 100.0
        )
