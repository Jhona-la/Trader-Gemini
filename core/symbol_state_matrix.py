"""
Symbol State Matrix
Centralized in-memory store for global market state per symbol.
Updated continuously by the data handlers and accessed by the Meta-Arbitrator.
"""
import time
from typing import Dict, Any, Optional
from utils.logger import logger

class SymbolStateMatrix:
    """
    Maintains a real-time vector of state for each traded symbol.
    Provides O(1) access to symbol health, liquidity, and regime compatibility.
    """
    def __init__(self):
        self._matrix: Dict[str, Dict[str, Any]] = {}
        self._last_update: float = time.time()
        
    def get_state(self, symbol: str) -> Dict[str, Any]:
        """Returns a copy of the current state for a symbol, or default values."""
        return self._matrix.get(symbol, self._get_default_state(symbol)).copy()
        
    def _get_default_state(self, symbol: str) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "trend_score": 0.0,
            "micro_volatility": 0.0,
            "orderflow_pressure": 0.0,
            "liquidity_score": 0.5,
            "regime_class": "UNKNOWN",
            "funding_bias": 0.0,
            "correlation_cluster": 0,
            "signal_density": 0.0,
            "last_update": time.time()
        }

    def update_from_market_event(self, event) -> None:
        """
        Updates the matrix using a new MarketEvent.
        Called directly from engine._process_market_event.
        """
        symbol = getattr(event, 'symbol', None)
        if not symbol:
            return
            
        if symbol not in self._matrix:
            self._matrix[symbol] = self._get_default_state(symbol)
            
        state = self._matrix[symbol]
        
        # In a full implementation, these would be extracted from event.order_flow,
        # event.health_metrics or calculated via rolling window.
        # Here we provide a lightweight integration hook.
        
        # Example pseudo-extraction if metrics exist:
        if hasattr(event, 'order_flow') and event.order_flow:
            state['orderflow_pressure'] = event.order_flow.get('ofi', 0.0)
            
        if hasattr(event, 'health_metrics') and event.health_metrics:
            state['liquidity_score'] = event.health_metrics.get('liquidity', 0.5)
            
        state['last_update'] = time.time()
        self._last_update = time.time()

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Returns the entire matrix."""
        return self._matrix.copy()

# Singleton instance
symbol_state_matrix = SymbolStateMatrix()
