
import socket
import logging
import time
from functools import wraps

logger = logging.getLogger("DNSCache")

_original_getaddrinfo = socket.getaddrinfo
_dns_cache = {}
_dns_ttl = 300  # 5 minutes TTL for Binance IPs

def cache_dns_lookups():
    """
    🔬 PHASE 34: LOCAL DNS CACHING
    Monkey-patches socket.getaddrinfo to cache DNS results.
    Reduces 20-100ms latency per connection on some systems.
    """
    
    @wraps(_original_getaddrinfo)
    def cached_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        # Only cache Binance domains to be safe, or cache everything?
        # Caching everything is risky if user connects to dynamic services.
        # But for this bot, we mostly hit specific endpoints.
        
        # Key for cache
        key = (host, port, family, type, proto, flags)
        now = time.time()
        
        if key in _dns_cache:
            entry = _dns_cache[key]
            if now - entry['timestamp'] < _dns_ttl:
                # Cache Hit
                return entry['data']
            else:
                # Expired
                del _dns_cache[key]
        
        # Perform actual lookup
        try:
            res = _original_getaddrinfo(host, port, family, type, proto, flags)
            
            # Store in cache
            _dns_cache[key] = {
                'timestamp': now,
                'data': res
            }
            # logger.debug(f"🌍 DNS Cache: Miss -> Cached {host}")
            return res
        except Exception as e:
            # If lookup fails, ignore (let caller handle it)
            raise e

    socket.getaddrinfo = cached_getaddrinfo
    logger.info("🔬 [PHASE 34] Local DNS Cache active (TTL: 300s).")
