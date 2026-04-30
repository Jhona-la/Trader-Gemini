
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
        
        # OFI Tracking
        self.prev_best_bid = 0.0
        self.prev_best_ask = 0.0
        self.prev_bid_qty = 0.0
        self.prev_ask_qty = 0.0
        
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
        
    def calculate_spread(self) -> float:
        if not self.bids or not self.asks: return 0.0
        best_bid = max(self.bids.keys())
        best_ask = min(self.asks.keys())
        return best_ask - best_bid
        
    def calculate_microprice(self) -> float:
        if not self.bids or not self.asks: return 0.0
        best_bid = max(self.bids.keys())
        best_ask = min(self.asks.keys())
        bid_qty = self.bids[best_bid]
        ask_qty = self.asks[best_ask]
        imb = bid_qty / (bid_qty + ask_qty + 1e-9)
        return best_ask * imb + best_bid * (1.0 - imb)
        
    def calculate_ofi(self) -> float:
        if not self.bids or not self.asks: return 0.0
        
        curr_best_bid = max(self.bids.keys())
        curr_best_ask = min(self.asks.keys())
        curr_bid_qty = self.bids[curr_best_bid]
        curr_ask_qty = self.asks[curr_best_ask]
        
        delta_w = 0.0
        if curr_best_bid >= self.prev_best_bid:
            if curr_best_bid == self.prev_best_bid:
                delta_w = curr_bid_qty - self.prev_bid_qty
            else:
                delta_w = curr_bid_qty
        else:
            delta_w = -self.prev_bid_qty
            
        delta_v = 0.0
        if curr_best_ask <= self.prev_best_ask:
            if curr_best_ask == self.prev_best_ask:
                delta_v = curr_ask_qty - self.prev_ask_qty
            else:
                delta_v = curr_ask_qty
        else:
            delta_v = -self.prev_ask_qty
            
        self.prev_best_bid = curr_best_bid
        self.prev_best_ask = curr_best_ask
        self.prev_bid_qty = curr_bid_qty
        self.prev_ask_qty = curr_ask_qty
        
        return delta_w - delta_v

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
