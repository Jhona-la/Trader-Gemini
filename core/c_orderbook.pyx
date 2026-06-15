# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False
"""
⚡ Cython OrderBook Engine
Reemplazo directo para la versión Python para máxima velocidad en la reconstrucción del LOB.
"""

from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as cnp

cdef class OrderBook:
    cdef:
        dict bids
        dict asks
        int max_depth
        double prev_best_bid
        double prev_best_ask
        double prev_bid_qty
        double prev_ask_qty

    def __init__(self, int max_depth=100):
        self.max_depth = max_depth
        self.bids = {}
        self.asks = {}
        self.prev_best_bid = 0.0
        self.prev_best_ask = 0.0
        self.prev_bid_qty = 0.0
        self.prev_ask_qty = 0.0

    cpdef void update_bid(self, double price, double qty):
        if qty <= 0:
            if price in self.bids:
                del self.bids[price]
        else:
            self.bids[price] = qty

    cpdef void update_ask(self, double price, double qty):
        if qty <= 0:
            if price in self.asks:
                del self.asks[price]
        else:
            self.asks[price] = qty

    def get_snapshot(self):
        # Avoid lambda to bypass Cython closure error in cpdef
        cdef list bids_items = list(self.bids.items())
        bids_items.sort(key=lambda x: x[0], reverse=True)
        cdef list sorted_bids = bids_items[:self.max_depth]
        
        cdef list asks_items = list(self.asks.items())
        asks_items.sort(key=lambda x: x[0])
        cdef list sorted_asks = asks_items[:self.max_depth]
        
        return {'bids': sorted_bids, 'asks': sorted_asks}

    cpdef double calculate_spread(self):
        if not self.bids or not self.asks:
            return 0.0
        cdef double best_bid = max(self.bids.keys())
        cdef double best_ask = min(self.asks.keys())
        return best_ask - best_bid

    cpdef double calculate_microprice(self):
        if not self.bids or not self.asks:
            return 0.0
        cdef double best_bid = max(self.bids.keys())
        cdef double best_ask = min(self.asks.keys())
        cdef double bid_qty = self.bids[best_bid]
        cdef double ask_qty = self.asks[best_ask]
        cdef double imb = bid_qty / (bid_qty + ask_qty + 1e-9)
        return best_ask * imb + best_bid * (1.0 - imb)

    cpdef double calculate_ofi(self):
        if not self.bids or not self.asks:
            return 0.0
            
        cdef double curr_best_bid = max(self.bids.keys())
        cdef double curr_best_ask = min(self.asks.keys())
        cdef double curr_bid_qty = self.bids[curr_best_bid]
        cdef double curr_ask_qty = self.asks[curr_best_ask]
        
        cdef double delta_w = 0.0
        if curr_best_bid >= self.prev_best_bid:
            if curr_best_bid == self.prev_best_bid:
                delta_w = curr_bid_qty - self.prev_bid_qty
            else:
                delta_w = curr_bid_qty
        else:
            delta_w = -self.prev_bid_qty
            
        cdef double delta_v = 0.0
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
