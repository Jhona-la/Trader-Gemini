
import sys

# Cache for commonly used strings to force interning
# Explicit reference keeping to prevent GC collection of interned strings
_INTERN_CACHE = set()

def intern_string(s: str) -> str:
    """
    ⚡ PHASE 21: STRING INTERNING
    Forces the string 's' to be interned (cached in Python's internal table).
    Returns the canonical interned string reference.
    
    Usage:
        symbol = intern_string(msg['s'])
        if symbol is "BTCUSDT": ... # FAST pointer comparison
    """
    if not isinstance(s, str):
        return s
        
    s_interned = sys.intern(s)
    _INTERN_CACHE.add(s_interned) # Keep reference alive
    return s_interned

# Pre-intern common constants
SIDE_BUY = intern_string("BUY")
SIDE_SELL = intern_string("SELL")
TYPE_LIMIT = intern_string("LIMIT")
TYPE_MARKET = intern_string("MARKET")
STATUS_NEW = intern_string("NEW")
STATUS_FILLED = intern_string("FILLED")
STATUS_CANCELED = intern_string("CANCELED")
