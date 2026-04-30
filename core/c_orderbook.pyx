
# cython: language_level=3
# distutils: language = c++

import time
import cython
from libc.stdlib cimport malloc, free
from libc.string cimport memmove

cdef struct OrderLevel:
    double price
    double qty

cdef class OrderBook:
    """
    🔬 PHASE 28: CYTHONIZED ORDERBOOK
    High-performance OrderBook implemented in C-optimized Cython.
    Manages Bids and Asks as sorted arrays for fast insertion/deletion.
    """
    cdef OrderLevel* bids
    cdef OrderLevel* asks
    cdef int max_depth
    cdef public int bid_count
    cdef public int ask_count
    
    # OFI Tracking
    cdef double prev_best_bid
    cdef double prev_best_ask
    cdef double prev_bid_qty
    cdef double prev_ask_qty

    def __cinit__(self, int max_depth=100):
        self.max_depth = max_depth
        self.bid_count = 0
        self.ask_count = 0
        self.bids = <OrderLevel*> malloc(max_depth * sizeof(OrderLevel))
        self.asks = <OrderLevel*> malloc(max_depth * sizeof(OrderLevel))
        
        self.prev_best_bid = 0.0
        self.prev_best_ask = 0.0
        self.prev_bid_qty = 0.0
        self.prev_ask_qty = 0.0
        
        if not self.bids or not self.asks:
             raise MemoryError()

    def __dealloc__(self):
        if self.bids: free(self.bids)
        if self.asks: free(self.asks)

    cdef int _find_bid_idx(self, double price) noexcept:
        cdef int i
        for i in range(self.bid_count):
            if self.bids[i].price == price:
                return i
            elif self.bids[i].price < price:
                # Found insertion point (bids sorted descending)
                return i
        return self.bid_count

    cdef int _find_ask_idx(self, double price) noexcept:
        cdef int i
        for i in range(self.ask_count):
            if self.asks[i].price == price:
                return i
            elif self.asks[i].price > price:
                # Found insertion point (asks sorted ascending)
                return i
        return self.ask_count

    cpdef void update_bid(self, double price, double qty):
        """Update a bid level. If qty is 0, remove."""
        cdef int idx = -1
        cdef int i
        
        # Check if exists
        for i in range(self.bid_count):
            if self.bids[i].price == price:
                idx = i
                break
                
        if qty == 0.0:
            if idx >= 0:
                # Remove
                if idx < self.bid_count - 1:
                    memmove(&self.bids[idx], &self.bids[idx+1], (self.bid_count - 1 - idx) * sizeof(OrderLevel))
                self.bid_count -= 1
        else:
            if idx >= 0:
                self.bids[idx].qty = qty
            else:
                # Insert
                idx = self._find_bid_idx(price)
                if idx < self.max_depth:
                    if idx < self.bid_count:
                        # Shift right
                        shift_count = min(self.bid_count, self.max_depth - 1) - idx
                        memmove(&self.bids[idx+1], &self.bids[idx], shift_count * sizeof(OrderLevel))
                    self.bids[idx].price = price
                    self.bids[idx].qty = qty
                    if self.bid_count < self.max_depth:
                        self.bid_count += 1

    cpdef void update_ask(self, double price, double qty):
        """Update an ask level. If qty is 0, remove."""
        cdef int idx = -1
        cdef int i
        
        for i in range(self.ask_count):
            if self.asks[i].price == price:
                idx = i
                break
                
        if qty == 0.0:
            if idx >= 0:
                if idx < self.ask_count - 1:
                    memmove(&self.asks[idx], &self.asks[idx+1], (self.ask_count - 1 - idx) * sizeof(OrderLevel))
                self.ask_count -= 1
        else:
            if idx >= 0:
                self.asks[idx].qty = qty
            else:
                idx = self._find_ask_idx(price)
                if idx < self.max_depth:
                    if idx < self.ask_count:
                        shift_count = min(self.ask_count, self.max_depth - 1) - idx
                        memmove(&self.asks[idx+1], &self.asks[idx], shift_count * sizeof(OrderLevel))
                    self.asks[idx].price = price
                    self.asks[idx].qty = qty
                    if self.ask_count < self.max_depth:
                        self.ask_count += 1
                        
    cpdef double calculate_spread(self):
        if self.bid_count == 0 or self.ask_count == 0:
            return 0.0
        return self.asks[0].price - self.bids[0].price
        
    cpdef double calculate_microprice(self):
        """Volume-weighted mid price (Microprice)"""
        if self.bid_count == 0 or self.ask_count == 0:
            return 0.0
        cdef double best_bid = self.bids[0].price
        cdef double best_ask = self.asks[0].price
        cdef double bid_qty = self.bids[0].qty
        cdef double ask_qty = self.asks[0].qty
        cdef double imb = bid_qty / (bid_qty + ask_qty + 1e-9)
        return best_ask * imb + best_bid * (1.0 - imb)
        
    cpdef double calculate_ofi(self):
        """Order Flow Imbalance (OFI) - Contovounesios et al."""
        if self.bid_count == 0 or self.ask_count == 0:
            return 0.0
            
        cdef double curr_best_bid = self.bids[0].price
        cdef double curr_best_ask = self.asks[0].price
        cdef double curr_bid_qty = self.bids[0].qty
        cdef double curr_ask_qty = self.asks[0].qty
        
        cdef double delta_W = 0.0
        cdef double delta_V = 0.0
        
        # Bid side (W)
        if curr_best_bid >= self.prev_best_bid:
            if curr_best_bid == self.prev_best_bid:
                delta_W = curr_bid_qty - self.prev_bid_qty
            else:
                delta_W = curr_bid_qty
        else:
            delta_W = -self.prev_bid_qty
            
        # Ask side (V)
        if curr_best_ask <= self.prev_best_ask:
            if curr_best_ask == self.prev_best_ask:
                delta_V = curr_ask_qty - self.prev_ask_qty
            else:
                delta_V = curr_ask_qty
        else:
            delta_V = -self.prev_ask_qty
            
        # Update state for next tick
        self.prev_best_bid = curr_best_bid
        self.prev_best_ask = curr_best_ask
        self.prev_bid_qty = curr_bid_qty
        self.prev_ask_qty = curr_ask_qty
        
        # OFI = ΔW - ΔV (Positive = Buy pressure, Negative = Sell pressure)
        return delta_W - delta_V

    cpdef dict get_snapshot(self):
        """Return Python dict (for compat)"""
        cdef list bids_list = []
        cdef list asks_list = []
        cdef int i
        
        for i in range(self.bid_count):
            bids_list.append([self.bids[i].price, self.bids[i].qty])
            
        for i in range(self.ask_count):
            asks_list.append([self.asks[i].price, self.asks[i].qty])
            
        return {'bids': bids_list, 'asks': asks_list}
