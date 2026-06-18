from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from core.events import OrderEvent

class ExchangeInterface(ABC):
    """
    OMEGA PROTOCOL: MULTI-EXCHANGE ABSTRACTION LAYER (Phase 58)
    Standardizes interaction with any crypto exchange (Binance, Bybit, Kraken).
    """
    
    @abstractmethod
    async def connect(self):
        """Establish connection to API/Websockets."""
        pass
        
    @abstractmethod
    async def get_orderbook(self, symbol: str, limit: int = 10) -> Dict:
        """Fetch Level 2 Orderbook."""
        pass
        
    @abstractmethod
    async def place_order(self, order: OrderEvent) -> Optional[Dict]:
        """Submit order to matching engine."""
        pass
        
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel specific order."""
        pass
        
    @abstractmethod
    async def get_balance(self, asset: str) -> float:
        """Get available balance for asset."""
        pass
        
    @abstractmethod
    async def get_historical_data(self, symbol: str, timeframe: str, limit: int) -> List:
        """Fetch OHLCV candles."""
        pass
