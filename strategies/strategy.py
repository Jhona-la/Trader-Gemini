from abc import ABC, abstractmethod

class Strategy(ABC):
    """
    Abstract base class for all strategies.
    Supports Dual-Horizon categorization: 'scalping' or 'swing'.
    Now automatically binds Universal dependencies via **kwargs.
    """
    def __init__(self, **kwargs):
        horizon_type = kwargs.get('horizon_type') or kwargs.get('horizon', 'SCALPING')
        if horizon_type.lower() not in ["scalping", "swing", "microscalping"]:
            # Fallback instead of crash for some legacy configs
            horizon_type = "SCALPING"
        self.horizon_type = horizon_type.upper()
        self.horizon = self.horizon_type
        
        # Universal dependencies injected by UniversalStrategyAdapter
        self.data_provider = kwargs.get('data_provider')
        self.events_queue = kwargs.get('events_queue')
        self.portfolio = kwargs.get('portfolio')
        self.risk_manager = kwargs.get('risk_manager')
        self.executor = kwargs.get('executor')
        self.sentiment_loader = kwargs.get('sentiment_loader')
        self.symbol = kwargs.get('symbol', 'ALL')
        self.priority = kwargs.get('priority', 1)
        
        self.sophia = None
        self.requires_training = False # Default for non-ML strategies

    def get_active_pos(self, symbol: str):
        """
        Safely retrieves the active position for this strategy's horizon.
        Prevents Cross-Horizon Cannibalization (Ghost Bug).
        """
        portfolio = getattr(self, 'portfolio', None)
        horizon = getattr(self, 'horizon', 'SCALPING')
        
        if portfolio is not None:
            if hasattr(portfolio, 'get_horizon_position'):
                horizon_pos = portfolio.get_horizon_position(symbol, horizon)
                if horizon_pos is not None:
                    return horizon_pos
            # Fallback physical position
            return portfolio.positions.get(symbol)
            
        dp = getattr(self, 'data_provider', None)
        if dp is not None and hasattr(dp, 'get_active_positions'):
            return dp.get_active_positions().get(symbol)
        return None

    @abstractmethod
    def calculate_signals(self, event, *args, **kwargs):
        """
        Calculate signals based on market data.
        Must return a list of SignalEvents or a single SignalEvent.
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
