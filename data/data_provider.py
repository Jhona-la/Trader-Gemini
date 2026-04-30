from abc import ABC, abstractmethod

class DataProvider(ABC):
    """
    Abstract base class for all data handlers.
    """
    
    @abstractmethod
    def get_latest_bars(self, symbol, n=1):
        """
        Returns the last N bars from the latest_symbol list.
        """
        raise NotImplementedError("Should implement get_latest_bars()")

    @abstractmethod
    def update_bars(self):
        """
        Pushes the latest bars to the bars_queue for each symbol
        in a tuple format: (symbol, datetime, open, high, low, close, volume).
        """
        raise NotImplementedError("Should implement update_bars()")

    @abstractmethod
    def get_order_flow_metrics(self, symbol: str) -> dict:
        """
        Returns real-time order flow and microstructure metrics.
        """
        raise NotImplementedError("Should implement get_order_flow_metrics()")

    @abstractmethod
    def get_derivatives_metrics(self, symbol: str) -> dict:
        """
        Returns futures derivatives metrics (Funding, Open Interest, Liquidations).
        Expected format: {'funding_rate': float, 'oi': float, 'oi_delta': float, 'liquidations': float}
        """
        raise NotImplementedError("Should implement get_derivatives_metrics()")
        
    @abstractmethod
    def get_orderbook(self, symbol: str):
        """
        Returns the OrderBook instance for a symbol to access L2 metrics (OFI, Spread, Microprice).
        """
        pass
