# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False
import cython
from libc.stdlib cimport malloc, free

cdef extern from "math.h":
    double fmax(double x, double y) nogil
    double fmin(double x, double y) nogil

cpdef inline tuple c_update_hwm_lwm(double price, double current_hwm, double current_lwm):
    cdef double new_hwm = fmax(price, current_hwm)
    cdef double new_lwm = price if current_lwm == 0.0 else fmin(price, current_lwm)
    return new_hwm, new_lwm

cdef class NanoVirtualLedger:
    """
    O(1) Virtual Ledger Syncer (Zero-Latency)
    Replaces dict.items() and .startswith() loop in Python.
    """
    cdef dict virtual_ledger
    cdef dict positions
    cdef dict symbol_to_vkeys
    
    def __init__(self, dict virtual_ledger, dict positions):
        self.virtual_ledger = virtual_ledger
        self.positions = positions
        self.symbol_to_vkeys = {}
        
    cpdef void register_vkey(self, str symbol, str v_key):
        if symbol not in self.symbol_to_vkeys:
            self.symbol_to_vkeys[symbol] = []
        if v_key not in self.symbol_to_vkeys[symbol]:
            self.symbol_to_vkeys[symbol].append(v_key)
            
    cpdef void unregister_vkey(self, str symbol, str v_key):
        if symbol in self.symbol_to_vkeys:
            if v_key in self.symbol_to_vkeys[symbol]:
                self.symbol_to_vkeys[symbol].remove(v_key)
                
    cpdef list update_market_price(self, str symbol, double price, bint is_ghost_tick):
        """
        Updates the HWM and LWM for all positions associated with the symbol.
        Returns a list of active v_keys that were updated.
        """
        cdef list active_v_keys = []
        cdef list v_keys
        cdef dict vpos
        cdef dict ppos
        cdef double hwm, lwm
        
        if symbol not in self.symbol_to_vkeys:
            return active_v_keys
            
        v_keys = self.symbol_to_vkeys[symbol]
        
        for v_key in v_keys:
            if v_key in self.virtual_ledger:
                vpos = self.virtual_ledger[v_key]
                vpos['current_price'] = price
                
                hwm = vpos.get('high_water_mark', price)
                lwm = vpos.get('low_water_mark', price)
                
                if not is_ghost_tick:
                    hwm = fmax(price, hwm)
                    lwm = price if lwm == 0.0 else fmin(price, lwm)
                    vpos['high_water_mark'] = hwm
                    vpos['low_water_mark'] = lwm
                    
                if v_key in self.positions:
                    ppos = self.positions[v_key]
                    ppos['current_price'] = price
                    ppos['high_water_mark'] = hwm
                    ppos['low_water_mark'] = lwm
                    
                active_v_keys.append(v_key)
                
        return active_v_keys
