from abc import ABC, abstractmethod

class Strategy(ABC):
    """
    Abstract base class for all strategies.
    """
    def __init__(self):
        self.sophia = None

    @abstractmethod
    def calculate_signals(self, event):
        """
        Calculate signals based on market data.
        """
        raise NotImplementedError("Should implement calculate_signals()")

    def stop(self):
        """
        Signal the strategy to stop processing and cleanup resources.
        """
        pass
