
import logging
from typing import Dict, List, Optional

logger = logging.getLogger("OrderBook")

# 1. Try importing Cython implementation
try:
    from core.c_orderbook import OrderBook as CythonOrderBook
    CYTHON_AVAILABLE = True
    logger.info("🚀 Cython OrderBook loaded successfully.")
except ImportError as e:
    CYTHON_AVAILABLE = False
    logger.warning(f"⚠️ Cython OrderBook not found ({e}). Using Python fallback.")
    CythonOrderBook = object # Dummy for inheritance if needed, or just standard separate class

# 2. Python Fallback Implementation
class PythonOrderBook:
    """
    Standard Python implementation of OrderBook.
    Used when Cython compiled module is missing.
    """
    def __init__(self, max_depth=100):
        self.max_depth = max_depth
        self.bids = {} # Price -> Qty
        self.asks = {} # Price -> Qty
        
    def update_bid(self, price: float, qty: float):
        if qty <= 0:
            if price in self.bids:
                del self.bids[price]
        else:
            self.bids[price] = qty
            
    def update_ask(self, price: float, qty: float):
        if qty <= 0:
            if price in self.asks:
                del self.asks[price]
        else:
            self.asks[price] = qty
            
    def get_snapshot(self) -> Dict[str, List[float]]:
        # Sort and limit
        # Bids: Descending
        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:self.max_depth]
        # Asks: Ascending
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:self.max_depth]
        
        return {
            'bids': sorted_bids,
            'asks': sorted_asks
        }

# 3. Factory / Wrapper
    # 3. Factory / Wrapper
class OrderBook(CythonOrderBook if CYTHON_AVAILABLE else PythonOrderBook):
    """
    Main OrderBook class that inherits from the best available implementation.
    """
    def __init__(self, max_depth=100):
        if CYTHON_AVAILABLE:
            # Cython extension type init
            # __cinit__ has already run with the args passed to constructor
            # object.__init__ takes no args
            pass 
        else:
            # Python class init
            super().__init__(max_depth)
