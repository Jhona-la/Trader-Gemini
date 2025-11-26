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
