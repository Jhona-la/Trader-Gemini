from abc import ABC, abstractmethod

class Strategy(ABC):
    """
    Abstract base class for all strategies.
    Supports Dual-Horizon categorization: 'scalping' or 'swing'.
    """
    def __init__(self, horizon_type: str = "scalping"):
        if horizon_type not in ["scalping", "swing"]:
            raise ValueError(f"Invalid horizon_type: {horizon_type}")
        self.horizon_type = horizon_type
        self.sophia = None

    @abstractmethod
    def calculate_signals(self, event):
        """
        Calculate signals based on market data.
        Returns a list of SignalEvents or a single SignalEvent.
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
