# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False
import cython
from libc.stdlib cimport malloc, free

# --- PORTFOLIO MATH ---

cdef float _max(float a, float b):
    return a if a > b else b

cdef float _min(float a, float b):
    return a if a < b else b

cpdef float calculate_unrealized_pnl_fast(float current_price, float entry_price, float quantity, int direction):
    """
    QUANTUM-NANO: O(1) C-level PnL calculation.
    """
    cdef float pnl = 0.0
    cdef float price_diff = 0.0
    
    if quantity <= 0.0:
        return 0.0
        
    if direction == 1:
        price_diff = current_price - entry_price
    else:
        price_diff = entry_price - current_price
        
    pnl = (price_diff / entry_price) * quantity * entry_price
    return pnl

cpdef float calculate_kelly_fraction(int win_streak, int loss_streak, float winrate, float payoff_ratio, float max_kelly, float stress_score, bint apply_mult):
    """
    QUANTUM-NANO: C-level Kelly Criterion with Stress and Streak Modifiers
    """
    cdef float q = 1.0 - winrate
    cdef float kelly_pct = 0.0
    cdef float multiplier = 1.0
    cdef float divisor = 1.0
    
    if payoff_ratio <= 0.0:
        return 0.01
        
    kelly_pct = (winrate * payoff_ratio - q) / payoff_ratio
    
    if not apply_mult:
        if kelly_pct < 0.0: return 0.01
        return _min(kelly_pct, max_kelly)
        
    if stress_score < 90.0:
        kelly_pct *= 0.125
        
    if win_streak > 0:
        multiplier = 1.0 + (_min(win_streak, 5) * 0.1)
        kelly_pct *= multiplier
    elif loss_streak > 0:
        divisor = 1.0 + (_min(loss_streak, 5) * 0.2)
        kelly_pct /= divisor
        
    if kelly_pct <= 0.0:
        return 0.01
        
    return _min(kelly_pct, max_kelly)

cpdef tuple update_hwm_lwm(float price, float hwm, float lwm):
    cdef float new_hwm = hwm
    cdef float new_lwm = lwm
    
    if price > hwm:
        new_hwm = price
    if lwm == 0.0 or price < lwm:
        new_lwm = price
        
    return (new_hwm, new_lwm)


# --- NANO PRIORITY QUEUE ---

cdef class NanoPriorityQueue:
    """
    Lock-Free C-level Priority Queue for Zero-Latency Event Routing.
    Uses pre-allocated Cython arrays to emulate a zero-allocation Ring Buffer.
    """
    cdef list p0_buffer
    cdef list p1_buffer
    cdef list p2_buffer
    
    cdef int p0_head, p0_tail, p0_count
    cdef int p1_head, p1_tail, p1_count
    cdef int p2_head, p2_tail, p2_count
    
    cdef int capacity
    cdef public int total_items

    def __cinit__(self, int capacity=5000):
        self.capacity = capacity
        # Pre-allocate lists
        self.p0_buffer = [None] * capacity
        self.p1_buffer = [None] * capacity
        self.p2_buffer = [None] * capacity
        
        self.p0_head = 0; self.p0_tail = 0; self.p0_count = 0
        self.p1_head = 0; self.p1_tail = 0; self.p1_count = 0
        self.p2_head = 0; self.p2_tail = 0; self.p2_count = 0
        self.total_items = 0

    cpdef bint put(self, object item, int priority):
        """
        Puts an item into the ring buffer at the specified priority level (0, 1, 2).
        Returns False if the buffer is full.
        """
        if priority == 0:
            if self.p0_count >= self.capacity:
                return False
            self.p0_buffer[self.p0_tail] = item
            self.p0_tail = (self.p0_tail + 1) % self.capacity
            self.p0_count += 1
        elif priority == 1:
            if self.p1_count >= self.capacity:
                return False
            self.p1_buffer[self.p1_tail] = item
            self.p1_tail = (self.p1_tail + 1) % self.capacity
            self.p1_count += 1
        else:
            if self.p2_count >= self.capacity:
                return False
            self.p2_buffer[self.p2_tail] = item
            self.p2_tail = (self.p2_tail + 1) % self.capacity
            self.p2_count += 1
            
        self.total_items += 1
        return True

    cpdef object get(self):
        """
        Gets the highest priority item from the buffer.
        Returns None if empty.
        """
        cdef object item = None
        
        if self.p0_count > 0:
            item = self.p0_buffer[self.p0_head]
            self.p0_buffer[self.p0_head] = None # Clear ref
            self.p0_head = (self.p0_head + 1) % self.capacity
            self.p0_count -= 1
        elif self.p1_count > 0:
            item = self.p1_buffer[self.p1_head]
            self.p1_buffer[self.p1_head] = None # Clear ref
            self.p1_head = (self.p1_head + 1) % self.capacity
            self.p1_count -= 1
        elif self.p2_count > 0:
            item = self.p2_buffer[self.p2_head]
            self.p2_buffer[self.p2_head] = None # Clear ref
            self.p2_head = (self.p2_head + 1) % self.capacity
            self.p2_count -= 1
        else:
            return None
            
        self.total_items -= 1
        return item

    cpdef int qsize(self):
        return self.total_items


# --- NANO ORDER BOOK ---

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
