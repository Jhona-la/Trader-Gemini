
import requests
import logging
from requests.adapters import HTTPAdapter
# from requests.packages.urllib3.util.retry import Retry 

logger = logging.getLogger("KeepAlive")

def tune_requests_session(session: requests.Session):
    """
    🔬 PHASE 38: KEEP-ALIVE TUNING (Synchronous)
    Optimizes a requests.Session for HFT:
    1. Increases Connection Pool Size (default 10 -> 50).
    2. Enforces Keep-Alive headers.
    3. Adds automatic retries for connection errors.
    """
    try:
        # Define Retry Strategy (Fast Fail)
        # We don't want exponential backoff here, strict HFT.
        # fast_retry = Retry(total=3, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
        
        # Create Adapter with optimized Pool
        adapter = HTTPAdapter(
            pool_connections=50,    # Number of cached connections
            pool_maxsize=50,        # Max threads accessing pool
            max_retries=3           # Simple retry count
        )
        
        # Mount adapter to https:// and http://
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        # Force default headers
        session.headers.update({
            "Connection": "keep-alive",
            "Keep-Alive": "timeout=60, max=1000"
        })
        
        logger.info("🔬 [PHASE 38] HTTP Session Tuned (Pool: 50, Keep-Alive: 60s)")
        
    except Exception as e:
        logger.error(f"Failed to tune session: {e}")

def tune_ccxt_exchange(exchange):
    """
    Optimizes CCXT exchange instance.
    """
    try:
        if hasattr(exchange, 'session'):
             tune_requests_session(exchange.session)
        
        # Set internal CCXT properties if applicable
        exchange.enableRateLimit = False # We handle it via Predictive Rate Limiter (Phase 14)
        exchange.timeout = 10000 # 10s timeout sincronizado con executor async
        
        # [WSS TCP ZOMBIE SHIELD] Forzar Ping-Pong sub-10s para mantener OS Socket Vivo
        if not hasattr(exchange, 'options'):
            exchange.options = {}
        if 'ws' not in exchange.options:
            exchange.options['ws'] = {}
            
        exchange.options['ws']['pingInterval'] = 4000  # Enviar PING a Binance cada 4s
        exchange.options['ws']['pingTimeout'] = 8000   # Dropear y reintentar si no hay PONG en 8s
        
        logger.info("🔬 [PHASE 38] CCXT Exchange Tuned (WS Ping: 4s/8s | Timeout: 10s).")
    except Exception as e:
        logger.error(f"Failed to tune CCXT: {e}")
