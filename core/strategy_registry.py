import os
import sys
import importlib
import inspect
import logging
from typing import Dict, Type, Any, List

logger = logging.getLogger("StrategyRegistry")

class UniversalStrategyAdapter:
    """
    Adapter that guarantees any strategy can be initialized and executed in
    Live, Paper Trading, or Backtesting modes.
    It inspects the strategy constructor and provides the necessary dependencies.
    """
    def __init__(self, strategy_class: Type, **dependencies):
        self.strategy_class = strategy_class
        self.dependencies = dependencies
        self.instance = None

    def initialize(self) -> Any:
        """Instantiates the strategy resolving its constructor arguments."""
        try:
            sig = inspect.signature(self.strategy_class.__init__)
            params = sig.parameters
            
            init_kwargs = {}
            for name, param in params.items():
                if name == 'self':
                    continue
                if name in self.dependencies:
                    init_kwargs[name] = self.dependencies[name]
                elif param.default != inspect.Parameter.empty:
                    # Use default value
                    pass
                else:
                    # If required but not in dependencies, we might have an issue.
                    # We pass None and hope for the best, or log a warning.
                    logger.warning(f"Missing required dependency '{name}' for {self.strategy_class.__name__}")
                    init_kwargs[name] = None
                    
            self.instance = self.strategy_class(**init_kwargs)
            return self.instance
        except Exception as e:
            logger.error(f"Failed to initialize {self.strategy_class.__name__}: {e}")
            return None


class UniversalStrategyRegistry:
    """
    Singleton registry that auto-discovers all strategies in the `strategies/` directory.
    """
    _instance = None
    _strategies: Dict[str, Type] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UniversalStrategyRegistry, cls).__new__(cls)
            cls._instance._discover_strategies()
        return cls._instance

    def _discover_strategies(self):
        """Scans the strategies directory and loads all Strategy subclasses."""
        strategies_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'strategies')
        if not os.path.exists(strategies_dir):
            logger.error(f"Strategies directory not found at {strategies_dir}")
            return

        sys.path.insert(0, os.path.dirname(strategies_dir))
        
        # Load the base strategy to compare
        try:
            from strategies.strategy import Strategy
        except ImportError:
            logger.error("Could not import base Strategy class.")
            return

        for filename in os.listdir(strategies_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = f"strategies.{filename[:-3]}"
                try:
                    module = importlib.import_module(module_name)
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        # Ensure it's a subclass of Strategy, not Strategy itself, and defined in this module
                        if issubclass(obj, Strategy) and obj is not Strategy and obj.__module__ == module_name:
                            self._strategies[name] = obj
                except Exception as e:
                    logger.debug(f"Could not load module {module_name}: {e}")

        logger.info(f"UniversalStrategyRegistry loaded {len(self._strategies)} strategies.")

    @classmethod
    def get_all_strategies(cls) -> Dict[str, Type]:
        registry = cls()
        return registry._strategies

    @classmethod
    def create_all(cls, **dependencies) -> List[Any]:
        """
        Creates instances of all registered strategies using the UniversalStrategyAdapter.
        If a strategy requires 'symbol' and it's not provided globally, it will create
        an instance for each symbol in Config.TRADING_PAIRS.
        """
        registry = cls()
        instances = []
        
        try:
            from config import Config
            trading_pairs = Config.TRADING_PAIRS
        except ImportError:
            trading_pairs = ["BTC/USDT"] # Fallback

        for name, strat_class in registry._strategies.items():
            sig = inspect.signature(strat_class.__init__)
            params = sig.parameters
            
            # Si la estrategia necesita un 'symbol' específico y no se pasó uno global
            if 'symbol' in params and 'symbol' not in dependencies:
                for symbol in trading_pairs:
                    local_deps = dependencies.copy()
                    local_deps['symbol'] = symbol
                    
                    # Special logic for MLStrategy lookback and risk_manager
                    if name == "MLStrategy":
                        local_deps['lookback'] = getattr(Config.Strategies, 'ML_LOOKBACK_BARS', 500)
                        if 'BTC' not in symbol:
                            local_deps['risk_manager'] = None # Only leader gets risk manager by default
                            
                    adapter = UniversalStrategyAdapter(strat_class, **local_deps)
                    instance = adapter.initialize()
                    if instance:
                        # Append symbol to ID to avoid collisions
                        if hasattr(instance, 'strategy_id'):
                            instance.strategy_id += f"_{symbol.replace('/', '')}"
                        instances.append(instance)
            else:
                adapter = UniversalStrategyAdapter(strat_class, **dependencies)
                instance = adapter.initialize()
                if instance:
                    instances.append(instance)
                    
        return instances

    @classmethod
    def get_all_genes(cls) -> Dict[str, Any]:
        """
        FASE 30: Conexión al Evolver
        Extrae todos los parámetros por defecto de las estrategias cargadas dinámicamente.
        Esto alimenta el super-genoma del Evolver.
        """
        registry = cls()
        super_genes = {}
        for name, strat_class in registry._strategies.items():
            sig = inspect.signature(strat_class.__init__)
            for param_name, param in sig.parameters.items():
                if param.default is not inspect.Parameter.empty and isinstance(param.default, (int, float)):
                    # Prefix to avoid collisions
                    gene_key = f"{name}.{param_name}"
                    super_genes[gene_key] = param.default
        return super_genes
