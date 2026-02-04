"""
🌐 BINANCE API MANAGER
======================

PROFESSOR METHOD:
- QUÉ: Gestor de conexiones a APIs de Binance con failover y rate limiting.
- POR QUÉ: Para evitar bans de IP y garantizar disponibilidad del dashboard.
- PARA QUÉ: Sincronización robusta del dashboard con estado de cuenta en tiempo real.
- CÓMO: Decoradores de failover, lectura de headers de rate limit, guardado atómico.
- CUÁNDO: En cada llamada del dashboard a Binance.
- DÓNDE: Usado por dashboard/app.py para obtener datos de cuenta.
- QUIÉN: Dashboard y cualquier componente que necesite datos de Binance.
"""

import os
import json
import time
import threading
from datetime import datetime, timezone
from functools import wraps
from typing import Optional, Dict, Any, Callable
import requests

from utils.logger import logger
from utils.data_sync import atomic_write_json, touch_timestamp
from config import Config


class APIManager:
    """
    Gestor centralizado de conexiones a Binance con:
    - Rate limit awareness
    - Failover a datos locales
    - Guardado atómico
    - Heartbeat worker
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern para único gestor de API."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Rate Limit Tracking
        self.rate_limit_used = 0
        self.rate_limit_max = 1200  # Binance default
        self.rate_limit_threshold = 0.8  # Throttle at 80%
        
        # Connection Status
        self.last_api_call = None
        self.last_latency_ms = 0
        self.is_using_cache = False
        self.connection_status = "UNKNOWN"  # LIVE, CACHED, OFFLINE
        
        # Cache paths
        self.data_dir = "dashboard/data/futures"
        self.status_file = os.path.join(self.data_dir, "live_status.json")
        
        # Heartbeat Worker
        self._worker_running = False
        self._worker_thread = None
        
        # API Config
        self.timeout_prod = 5
        self.timeout_demo = 10
        
        # Drift Detection
        self.last_balance = 0.0
        self.drift_threshold = 0.01  # 1% change triggers broadcast
        
        self._initialized = True
    
    # =========================================================================
    # RATE LIMIT MANAGEMENT
    # =========================================================================
    
    def update_rate_limit(self, response_headers: dict):
        """
        Update rate limit tracking from Binance response headers.
        Header: X-MBX-USED-WEIGHT-1M
        """
        try:
            weight = response_headers.get('X-MBX-USED-WEIGHT-1M')
            if weight:
                self.rate_limit_used = int(weight)
                logger.debug(f"Rate limit: {self.rate_limit_used}/{self.rate_limit_max}")
        except:
            pass
    
    def should_throttle(self) -> bool:
        """Check if we should throttle requests to avoid ban."""
        usage_ratio = self.rate_limit_used / self.rate_limit_max
        return usage_ratio >= self.rate_limit_threshold
    
    def throttle_if_needed(self):
        """Sleep if approaching rate limit."""
        if self.should_throttle():
            wait_time = 5  # Wait 5 seconds
            logger.warning(f"⚠️ Rate limit at {self.rate_limit_used}/{self.rate_limit_max}. Throttling {wait_time}s...")
            time.sleep(wait_time)
    
    # =========================================================================
    # API CALLS WITH FAILOVER
    # =========================================================================
    
    def api_call_with_failover(
        self,
        api_func: Callable,
        fallback_key: str = None,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute API call with automatic failover to cached data.
        
        Args:
            api_func: Function that makes the actual API call
            fallback_key: Key to extract from cached data if API fails
            
        Returns:
            API response or cached data
        """
        # Check rate limit first
        self.throttle_if_needed()
        
        start_time = time.time()
        
        try:
            # Make API call
            result = api_func(*args, **kwargs)
            
            # Calculate latency
            self.last_latency_ms = int((time.time() - start_time) * 1000)
            self.last_api_call = datetime.now(timezone.utc)
            self.is_using_cache = False
            
            # Update status
            if self.last_latency_ms < 200:
                self.connection_status = "LIVE"
            else:
                self.connection_status = "SLOW"
            
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ API call failed: {e}. Using cached data.")
            self.is_using_cache = True
            self.connection_status = "CACHED"
            
            # Load from cache
            return self.load_cached_status(fallback_key)
    
    def load_cached_status(self, key: str = None) -> Dict[str, Any]:
        """Load last valid status from cache file."""
        try:
            if os.path.exists(self.status_file):
                with open(self.status_file, 'r') as f:
                    data = json.load(f)
                if key:
                    return data.get(key, {})
                return data
        except Exception as e:
            logger.error(f"❌ Failed to load cached status: {e}")
        
        return {}
    
    # =========================================================================
    # ATOMIC SAVE
    # =========================================================================
    
    # =========================================================================
    # ATOMIC SAVE
    # =========================================================================
    
    def save_status_atomic(self, data: Dict[str, Any], filepath: str = None):
        """
        Save status using atomic write utility.
        """
        filepath = filepath or self.status_file
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Use utility
        if atomic_write_json(data, filepath):
            # Update last_update signal
            touch_timestamp(os.path.join(os.path.dirname(filepath), "last_update.txt"))
            return True
        return False
    
    # =========================================================================
    # BINANCE ACCOUNT DATA
    # =========================================================================
    
    def get_account_balance(self, is_prod: bool = True) -> Dict[str, Any]:
        """
        Get account balance from Binance.
        Uses different endpoints for Spot vs Futures.
        """
        timeout = self.timeout_prod if is_prod else self.timeout_demo
        
        try:
            # Use existing binance client from config
            from binance.client import Client
            
            # Correct API key selection based on mode (Phase 6 Fix)
            if is_prod:
                api_key = Config.BINANCE_API_KEY
                api_secret = Config.BINANCE_SECRET_KEY
                use_testnet = False
            else:
                if Config.BINANCE_USE_FUTURES:
                    api_key = Config.BINANCE_DEMO_API_KEY
                    api_secret = Config.BINANCE_DEMO_SECRET_KEY
                else:
                    api_key = Config.BINANCE_TESTNET_API_KEY
                    api_secret = Config.BINANCE_TESTNET_SECRET_KEY
                use_testnet = True
            
            if not api_key or not api_secret:
                # logger.warning(f"API keys not configured for {'Production' if is_prod else 'Demo/Testnet'}")
                return self.load_cached_status()
            
            client = Client(
                api_key=api_key,
                api_secret=api_secret,
                testnet=use_testnet
            )
            
            start_time = time.time()
            
            if Config.BINANCE_USE_FUTURES:
                # Futures account
                account = client.futures_account()
                self.last_latency_ms = int((time.time() - start_time) * 1000)
                
                result = {
                    'total_equity': float(account.get('totalWalletBalance', 0)) + float(account.get('totalUnrealizedProfit', 0)),
                    'wallet_balance': float(account.get('totalWalletBalance', 0)),
                    'available_balance': float(account.get('availableBalance', 0)),
                    'used_margin': float(account.get('totalPositionInitialMargin', 0)),
                    'maint_margin': float(account.get('totalMaintMargin', 0)), # Risk Metric
                    'margin_balance': float(account.get('totalMarginBalance', 0)), # Risk Metric
                    'unrealized_pnl': float(account.get('totalUnrealizedProfit', 0)),
                    'positions': self._parse_futures_positions(account.get('positions', [])),
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'source': 'LIVE_API',
                    'latency_ms': self.last_latency_ms
                }
            else:
                # Spot account
                account = client.get_account()
                self.last_latency_ms = int((time.time() - start_time) * 1000)
                
                # Calculate total in USDT equivalent
                balances = {b['asset']: float(b['free']) + float(b['locked']) 
                           for b in account.get('balances', []) 
                           if float(b['free']) + float(b['locked']) > 0}
                
                result = {
                    'total_equity': balances.get('USDT', 0),
                    'available_balance': balances.get('USDT', 0),
                    'balances': balances,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'source': 'LIVE_API',
                    'latency_ms': self.last_latency_ms
                }
            
            # Update connection status
            self.connection_status = "LIVE" if self.last_latency_ms < 200 else "SLOW"
            self.is_using_cache = False
            
            # Phase 6 Fix: Only save atomically IF we have a successful LIVE API fetch
            # This prevents a failing dashboard from wiping the bot's real positions
            if result and result.get('source') == 'LIVE_API' and result.get('total_equity', 0) > 0:
                self.save_status_atomic(result)
            
            return result
            
        except Exception as e:
            # 429 Rate Limit Handling
            if "429" in str(e) or "Way Too Many Requests" in str(e):
                logger.critical("⛔ BINANCE RATE LIMIT HIT! Sleeping 60s...")
                time.sleep(60)
            
            logger.warning(f"⚠️ Binance API error: {e}")
            self.connection_status = "CACHED"
            self.is_using_cache = True
            return self.load_cached_status()
    
    def _parse_futures_positions(self, positions: list) -> Dict[str, Any]:
        """Parse futures positions into simpler format."""
        result = {}
        for pos in positions:
            amt = float(pos.get('positionAmt', 0))
            if amt != 0:
                symbol = pos.get('symbol', '')
                result[symbol] = {
                    'quantity': amt,
                    'avg_price': float(pos.get('entryPrice', 0)),
                    'current_price': float(pos.get('markPrice', 0)),
                    'unrealized_pnl': float(pos.get('unrealizedProfit', 0)),
                    'leverage': int(pos.get('leverage', 1))
                }
        return result
    
    # =========================================================================
    # HEARTBEAT WORKER
    # =========================================================================
    
    def start_heartbeat_worker(self, interval: int = 5, is_prod: bool = True):
        """
        Start background thread that updates status every N seconds.
        Independent of Streamlit UI refresh.
        """
        if self._worker_running:
            return
        
        self._worker_running = True
        
        def worker():
            logger.info("💓 Heartbeat worker started")
            while self._worker_running:
                try:
                    self.get_account_balance(is_prod=is_prod)
                except Exception as e:
                    logger.debug(f"Heartbeat error: {e}")
                time.sleep(interval)
            logger.info("💔 Heartbeat worker stopped")
        
        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()
    
    def stop_heartbeat_worker(self):
        """Stop the heartbeat worker thread."""
        self._worker_running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=2)
    
    # =========================================================================
    # STATUS GETTERS
    # =========================================================================
    
    def get_connection_badge(self) -> tuple:
        """
        Get connection status badge for dashboard.
        Returns: (emoji, text, streamlit_type)
        """
        if self.connection_status == "LIVE":
            return ("🟢", "Live API", "success")
        elif self.connection_status == "SLOW":
            return ("🟡", f"Slow ({self.last_latency_ms}ms)", "warning")
        elif self.connection_status == "CACHED":
            return ("🟡", "Cached Data", "warning")
        else:
            return ("🔴", "Offline", "error")
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of API manager status."""
        return {
            'connection_status': self.connection_status,
            'is_using_cache': self.is_using_cache,
            'last_latency_ms': self.last_latency_ms,
            'rate_limit_used': self.rate_limit_used,
            'rate_limit_max': self.rate_limit_max,
            'worker_running': self._worker_running
        }


# ===========================================================================
# DECORATOR FOR FAILOVER
# ===========================================================================

def with_failover(fallback_key: str = None):
    """
    Decorator that wraps API calls with automatic failover.
    
    Usage:
        @with_failover('account')
        def get_binance_account():
            return client.get_account()
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = APIManager()
            return manager.api_call_with_failover(func, fallback_key, *args, **kwargs)
        return wrapper
    return decorator


# ===========================================================================
# SINGLETON GETTER
# ===========================================================================

def get_api_manager() -> APIManager:
    """Get the singleton API Manager instance."""
    return APIManager()
