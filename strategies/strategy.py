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

    def check_exit(self, position, current_price, data_provider, now=None):
        """
        Evaluate and return an exit SignalEvent if this strategy determines the position should close.
        Returns None if position should remain open.
        
        [INTELLIGENT EXIT]: Las estrategias deben sobreescribir este método 
        consultando a su IA interna (ej. self.sophia) para determinar la salud del trade
        en tiempo real (win_probability).
        """
        return None
