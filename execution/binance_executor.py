import ccxt
from datetime import datetime, timezone
from config import Config
from core.events import FillEvent
from utils.logger import logger
from utils.metrics_exporter import metrics
from utils.error_handler import retry_on_api_error, handle_balance_error, handle_order_error
from utils.debug_tracer import trace_execution
from .liquidity_guardian import LiquidityGuardian
from utils.latency_monitor import latency_monitor
import time
import asyncio
import aiohttp
import numpy as np
import ccxt.async_support as ccxt_async
import uuid

# [FASE 28] Nano-Executor SIMD Parsing
try:
    import orjson
    import json
    
    # Monkey-patch CCXT's default json methods if possible
    if hasattr(ccxt, 'json'):
        ccxt.json = orjson.loads
        
    # Safe monkey-patch for standard json that supports fallback for kwargs
    _orig_loads = json.loads
    _orig_dumps = json.dumps
    
    def safe_loads(obj, **kwargs):
        if kwargs:
            return _orig_loads(obj, **kwargs)
        return orjson.loads(obj)
        
    json.loads = safe_loads
    json.dumps = lambda obj, **kwargs: orjson.dumps(obj).decode('utf-8')
except ImportError:
    pass

class BinanceExecutor:
    """
    Handles execution of orders on Binance via CCXT.
    Supports both Spot and Testnet.
    Integrated with UserDataStream for real-time fills.
    """
    def __init__(self, events_queue, portfolio=None, data_provider=None, micro_awareness=None):
        self.events_queue = events_queue
        self.portfolio = portfolio  # Reference for cash release on failure (fallback)
        self.data_provider = data_provider # Source for VWAP and market data
        self.micro_awareness = micro_awareness # Conciencia de cuenta micro
        self.order_manager = None   # Set during engine initialization
        
        # Configure Exchange
        options = {
            'adjustForTimeDifference': True,
            'fetchBalance': False,  # Disable auto balance fetch to prevent Spot endpoint calls
            'fetchMyTrades': False,  # Disable auto trade fetch
            'fetchCurrencies': False,  # CRITICAL: Disable currency fetch to avoid sapi endpoints in Testnet
            'recvWindow': 60000, # CRITICAL: Tolerate up to 60s clock drift
            'margin': False, # PREVENT: AuthenticationError on sapi endpoints
        }
        
        if Config.BINANCE_USE_FUTURES:
            options['defaultType'] = 'future'
        else:
            options['defaultType'] = 'spot'
        
        # Determinar qué API keys usar según el modo
        # BUG #34 FIX: Separate logic for Spot vs Futures
        # - Spot Testnet uses BINANCE_TESTNET_API_KEY
        # - Futures Demo uses BINANCE_DEMO_API_KEY  
        if Config.BINANCE_USE_FUTURES and hasattr(Config, 'BINANCE_USE_DEMO') and Config.BINANCE_USE_DEMO:
            # FUTURES DEMO MODE
            api_key = Config.BINANCE_DEMO_API_KEY
            secret_key = Config.BINANCE_DEMO_SECRET_KEY
            mode_description = "DEMO TRADING (Futures con capital virtual)"
        elif Config.BINANCE_USE_TESTNET:
            # SPOT TESTNET MODE
            api_key = Config.BINANCE_TESTNET_API_KEY
            secret_key = Config.BINANCE_TESTNET_SECRET_KEY
            mode_description = "TESTNET (Spot)"
        else:
            # LIVE PRODUCTION MODE
            api_key = Config.BINANCE_API_KEY
            secret_key = Config.BINANCE_SECRET_KEY
            mode_description = "LIVE"
            
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret_key,
            'enableRateLimit': True,
            'adjustForTimeDifference': True, # PHASE 1: Auto-sync OS drift
            'timeout': 10000, # CRITICAL: 10s timeout
            'options': options
        })
        
        # Habilitar el modo correspondiente (Standardized Phase 6)
        is_demo = hasattr(Config, 'BINANCE_USE_DEMO') and Config.BINANCE_USE_DEMO
        is_demo_futures = Config.BINANCE_USE_FUTURES and is_demo
        if is_demo_futures and hasattr(self.exchange, 'enable_demo_trading'):
            self.exchange.enable_demo_trading(True)
            logger.info(f"🚀 Binance Executor: Running in {mode_description} (Demo Trading Active)")
        elif Config.BINANCE_USE_TESTNET:
            self.exchange.set_sandbox_mode(True)
            logger.info(f"🚀 Binance Executor: Running in {mode_description} (Sandbox Mode Active)")
        elif is_demo_futures:
            logger.warning("⚠️ Binance Executor: Demo Trading requested but CCXT lacks enable_demo_trading(). Falling back to Live endpoints.")
            logger.info(f"🚀 Binance Executor: Running in {mode_description} (Live Mode ACTIVE)")
        else:
            logger.info(f"🚀 Binance Executor: Running in {mode_description} (Live Mode ACTIVE)")

        # Phase 7: Guardián de Liquidez
        self.guardian = LiquidityGuardian(self.exchange)
        self.latency_violations = 0 # Sovereign-Deploy Telemetry

            
        # Phase 15: Leverage Cache & Management
        self._leverage_cache = {} # symbol -> leverage
        self._margin_cache = {}   # symbol -> bool (is_isolated)
        if Config.BINANCE_USE_FUTURES:
            logger.info("⚡ Binance Executor: FUTURES MODE ENABLED (Programmatic Leverage Active)")
        
        # ===================================================================
        # Create permanent Spot exchange instance for balance queries
        # ===================================================================
        # Spot Testnet uses different URLs from Futures Testnet
        # We maintain a separate exchange for Spot queries
        if (hasattr(Config, 'BINANCE_USE_DEMO') and Config.BINANCE_USE_DEMO) or Config.BINANCE_USE_TESTNET:
            self.spot_exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': secret_key,
                'enableRateLimit': True,
                'adjustForTimeDifference': True,
                'timeout': 10000, # CRITICAL: 10s timeout
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                    'margin': False,
                }
            })
            if is_demo_futures and hasattr(self.spot_exchange, 'enable_demo_trading'):
                self.spot_exchange.enable_demo_trading(True)
                logger.info("  → Spot exchange initialized for Demo Trading")
            elif Config.BINANCE_USE_TESTNET:
                self.spot_exchange.set_sandbox_mode(True)
                logger.info("  → Spot exchange initialized for Testnet")
            
        # Phase 14: Rate Limiter
        from core.rate_limiter import PredictiveRateLimiter
        self.rate_limiter = PredictiveRateLimiter()
        
        # 🚀 AITS P4: Cython Execution Intelligence
        self.fast_signer = None
        self.http_session = None  # Lazy initialization in async context
        try:
            from core.rust_execution_bridge import RustBinanceSigner
            self.fast_signer = RustBinanceSigner(api_key, secret_key)
            logger.info("⚡ [EXEC] Rust RustBinanceSigner activated.")
        except ImportError:
            logger.warning("⚠️ [EXEC] c_executor not compiled. Using pure CCXT.")

        # 🏎️ FASE 6: C++ Native TCP/TLS Executor
        self.cpp_socket = None
        if not getattr(Config, 'BINANCE_USE_DEMO', False) and not Config.BINANCE_USE_TESTNET:
            try:
                from execution.cpp_executor_wrapper import CppBinanceExecutor
                self.cpp_socket = CppBinanceExecutor(api_key, secret_key)
                self.cpp_socket.connect()
                logger.info("⚡ [EXEC-CPP] C++ Native TCP/TLS Socket Activated. Nano-latency engaged.")
            except ImportError as e:
                logger.warning(f"⚠️ [EXEC-CPP] cpp_executor_wrapper not available: {e}")

        # ⚡ FASE 12.5: ZERO-LATENCY WEBSOCKET EXECUTOR
        self.ws_executor = None
        try:
            from execution.ws_order_executor import WSOrderExecutor
            self.ws_executor = WSOrderExecutor(
                api_key, 
                secret_key, 
                is_testnet=Config.BINANCE_USE_TESTNET, 
                is_futures=Config.BINANCE_USE_FUTURES
            )
            logger.info("⚡ [WS-EXEC] WSOrderExecutor instance created.")
        except Exception as e:
            logger.error(f"❌ [WS-EXEC] Failed to initialize WSOrderExecutor: {e}")

        # ===================================================================
        # Async Exchange Initialization
        # ===================================================================
        # 🚀 FASE 21: Neural Matrix Execution (HFT Optimization)
        # Refinando keep-alive limits de aiohttp para HFT. Reducimos latencia reconexión.
        try:
            connector = aiohttp.TCPConnector(limit=100, keepalive_timeout=60, ttl_dns_cache=300)
            session = aiohttp.ClientSession(connector=connector)
        except Exception:
            import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
            session = None

        async_options = {
            'apiKey': api_key,
            'secret': secret_key,
            'enableRateLimit': True,
            'adjustForTimeDifference': True,
            'timeout': 10000, 
            'options': options
        }
        if session:
            async_options['session'] = session

        self.async_exchange = ccxt_async.binance(async_options)
        
        # Inyectar async_exchange en LiquidityGuardian para llamadas de liquidez no bloqueantes
        if hasattr(self, 'guardian') and self.guardian:
            self.guardian.async_exchange = self.async_exchange
        if is_demo_futures and hasattr(self.async_exchange, 'enable_demo_trading'):
            self.async_exchange.enable_demo_trading(True)
            logger.info("  ⚡ Async CCXT initialized with Demo Trading")
        elif Config.BINANCE_USE_TESTNET:
            self.async_exchange.set_sandbox_mode(True)
            logger.info("  ⚡ Async CCXT initialized with Sandbox Mode")

        # CRITICAL FIX: Load markets immediately to prevent "markets not loaded" error
        # 1. Set Position Mode to Hedge Mode (Dual Side Position = True)
        if Config.BINANCE_USE_FUTURES and not Config.BINANCE_USE_DEMO:
             # This requires an async call or a one-time sync call.
             # We will attempt to set it via the sync exchange during init.
             try:
                 self.exchange.fapiPrivatePostPositionSideDual({'dualSidePosition': 'true'})
                 logger.info("🛡️ [HEDGE MODE] Enforced Dual Side Position (True)")
             except Exception as e:
                 import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                 logger.warning(f"⚠️ [HEDGE MODE] Could not enforce Hedge Mode (Already set or API error): {e}")

        # CRITICAL: Monkey patch 'request' method to intercept ALL sapi calls AND track Rate Limit
        # This is more robust than patching individual methods
        original_request = self.exchange.request
        
        def intercepted_request(path, api='public', method='GET', params=None, headers=None, body=None, config=None):
            if params is None: params = {}
            if config is None: config = {}
            # 1. TESTNET SAPI BLOCKER (BUG #17)
            if api == 'sapi' and ((hasattr(Config, 'BINANCE_USE_DEMO') and Config.BINANCE_USE_DEMO) or Config.BINANCE_USE_TESTNET):
                return []
            
            # 2. PHASE 14: PREDICTIVE RATE LIMIT CHECK
            is_safe, wait_time = self.rate_limiter.check_limit(weight_cost=1)
            if not is_safe:
                if wait_time > 0.5:
                    logger.warning(f"⚠️ [RATE LIMIT] Aborting request to {path} to avoid ThreadPool deadlock ({wait_time}s)")
                    raise ccxt.RateLimitExceeded(f"Rate limit wait too long: {wait_time}s")
                try:
                    import asyncio
                    loop = asyncio.get_running_loop()
                    if loop.is_running():
                        logger.error(f"🚨 [FATAL] Synchronous time.sleep({wait_time}) called inside AsyncIO Event Loop! Raising to avoid starvation.")
                        raise ccxt.RateLimitExceeded(f"Cannot sleep {wait_time}s inside asyncio loop. Path: {path}")
                    else:
                        time.sleep(wait_time)
                except RuntimeError:
                    time.sleep(wait_time)
            
            # 3. EXECUTE REQUEST
            response = original_request(path, api, method, params, headers, body, config)
            
            # 4. CAPTURE HEADERS (Server Truth)
            # CCXT stores last response headers in exchange object
            if hasattr(self.exchange, 'last_response_headers'):
                self.rate_limiter.update_from_headers(self.exchange.last_response_headers)
                
            return response
        
        self.exchange.request = intercepted_request
        
        # Apply to Spot Exchange instance as well if it exists
        if hasattr(self, 'spot_exchange') and self.spot_exchange:
            original_spot_request = self.spot_exchange.request
            def intercepted_spot_request(path, api='public', method='GET', params=None, headers=None, body=None, config=None):
                if params is None: params = {}
                if config is None: config = {}
                if api == 'sapi':
                    return []
                    
                is_safe, wait_time = self.rate_limiter.check_limit(weight_cost=1)
                if not is_safe:
                    try:
                        import asyncio
                        loop = asyncio.get_running_loop()
                        if loop.is_running():
                            raise ccxt.RateLimitExceeded(f"Cannot sleep {wait_time}s inside asyncio loop.")
                        else:
                            time.sleep(wait_time)
                    except RuntimeError:
                        time.sleep(wait_time)
                    
                response = original_spot_request(path, api, method, params, headers, body, config)
                
                if hasattr(self.spot_exchange, 'last_response_headers'):
                    self.rate_limiter.update_from_headers(self.spot_exchange.last_response_headers)
                    
                return response
            self.spot_exchange.request = intercepted_spot_request
            
        logger.info("  🔧 Testnet: Intercepting 'sapi' & 🛡️ Active Rate Limiting engaged")

        # Force time synchronization
        # BUG #24 FIX: load_time_difference() doesn't exist in current CCXT version
        # We rely on 'adjustForTimeDifference': True in options instead
        try:
            exchange_time = self.exchange.fetch_time()
            local_time = int(time.time() * 1000)
            diff = exchange_time - local_time
            drift = abs(diff)
            
            if drift > 5000: # [SOVEREIGN-DEPLOY] Local Network Adaptation (max 5000ms for Dev Dry-Run)
                 logger.critical(f"🛑 [TIME-DRIFT] Atomic Clock Sync Failed! Drift is {drift}ms (Max: 5000ms). Aborting.")
                 # En windows el NTP por defecto a veces falla, forzamos salida
                 raise RuntimeError(f"Time drift too high: {drift}ms > 5000ms limit")
            else:
                 logger.info(f"⏱️ [TIME-SYNC] Atomic Clock Aligned. Drift: {drift}ms. Forcing CCXT offset: {diff}ms")
                 
            # PROACTIVE CALIBRATION (Windows 500ms drift fix)
            self.exchange.options['timeDifference'] = diff
            if hasattr(self, 'spot_exchange') and self.spot_exchange:
                self.spot_exchange.options['timeDifference'] = diff

            # Just verify the exchange is reachable
            self.exchange.check_required_credentials()
            logger.info(f"  ✅ Exchange credentials verified")
        except ccxt.AuthenticationError as e:
            logger.error(f"  ❌ Authentication failed: Invalid API keys or permissions")
            logger.error(f"     Error: {e}")
            raise  # Fail fast on auth errors
        except ccxt.NetworkError as e:
            logger.warning(f"  ⚠️ Network error during credential check: {e}")
            logger.warning(f"     The bot will continue, but connectivity may be unstable")
        except ccxt.ExchangeError as e:
            logger.warning(f"  ⚠️ Exchange error during credential verification: {e}")

        # Phase 38: Keep-Alive Tuning
        try:
            from utils.keep_alive import tune_ccxt_exchange
            tune_ccxt_exchange(self.exchange)
            if hasattr(self, 'spot_exchange') and self.spot_exchange:
                tune_ccxt_exchange(self.spot_exchange)
        except Exception as e:
            import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
            logger.warning(f"Could not tune CCXT keep-alive: {e}")

        self._initialize_futures_settings()
        logger.info("✅ [Executor] Executor Ready (User Data Stream managed externally)")

    def set_order_manager(self, order_manager):
        """Injects OrderManager into the Executor and its components"""
        self.order_manager = order_manager
        logger.info("🔗 [Executor] OrderManager successfully injected into Executor.")

    def _initialize_futures_settings(self):
        """
        Enforces Hedge Mode (Dual Side Position) and Margin Type (Isolated) to match bot logic.
        """
        logger.info("Binance Executor: Initializing Futures Account Settings...")
        try:
            # 1. Set Position Mode to Hedge Mode (Dual Side Position = True)
            try:
                # 'true' = Hedge Mode, 'false' = One-Way Mode
                self.exchange.fapiPrivatePostPositionSideDual({'dualSidePosition': 'true'})
                logger.info("  ✅ Position Mode set to HEDGE (Multi-Horizon enabled)")
            except ccxt.ExchangeError as e:
                error_msg = str(e)
                if "No need to change" in error_msg or "-4059" in error_msg:
                    logger.info("  ✅ Position Mode already HEDGE")
                else:
                    logger.warning(f"  ⚠️ Could not set Position Mode: {error_msg}")
            except ccxt.NetworkError as e:
                logger.error(f"  ❌ Network error setting Position Mode: {e}")
                logger.warning("     Continuing without Position Mode change...")

            # 2. Set Margin Type for all pairs
            logger.info(f"  ⏳ Setting Margin Type to {Config.BINANCE_MARGIN_TYPE} for {len(Config.TRADING_PAIRS)} pairs...")
            
            # CRITICAL FIX for "binance markets not loaded" error
            try:
                self.exchange.load_markets()
            except Exception as e:
                import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                logger.debug(f"  Could not load markets implicitly: {e}")
                
            for symbol in Config.TRADING_PAIRS:
                try:
                    market = self.exchange.market(symbol)
                    symbol_id = market['id']
                    self.exchange.fapiPrivatePostMarginType({
                        'symbol': symbol_id,
                        'marginType': Config.BINANCE_MARGIN_TYPE.upper()
                    })
                except Exception as e:
                    import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                    if "No need to change" not in str(e) and "-4046" not in str(e):
                         logger.debug(f"  Could not set Margin Type for {symbol}: {e}")

            # 3. Verify Trading Permissions (Phase 14 Proactive Check)
            try:
                account_status = self.exchange.fapiPrivateV2GetAccount()
                can_trade = account_status.get('canTrade', False)
                if not can_trade:
                    logger.error("❌ CRITICAL: API key does NOT have trading permissions enabled!")
                else:
                    logger.info("  ✅ API Trading permissions verified")
            except Exception as e:
                import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                err_str = str(e)
                if "testnet/sandbox mode is not supported for futures anymore" in err_str:
                    logger.warning(f"  ⚠️ Ignorando Permission Check (Binance Vision Limits en Demo Trading)")
                else:
                    logger.warning(f"  ⚠️ Could not verify trading permissions: {err_str}")

            logger.info("  ✅ Margin Types Configured")
            
            # 4. [SOVEREIGN-DEPLOY] Dynamic Fee Awareness
            try:
                logger.info("  ⏳ Fetching Dynamic Commission Rates from Binance...")
                # Fetching fee for a major pair as baseline (Binance usually has global fees per tier)
                fee_info = self.exchange.fapiPrivateGetCommissionRate({'symbol': 'BTCUSDT'})
                taker_fee = float(fee_info.get('takerCommissionRate', 0.0004))
                maker_fee = float(fee_info.get('makerCommissionRate', 0.0002))
                
                # Update global FeeCalculator in RiskManager
                from risk.risk_manager import FeeCalculator
                FeeCalculator.update_dynamic_fees(maker_fee, taker_fee)
            except Exception as e:
                import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                err_str = str(e)
                if "testnet/sandbox mode is not supported for futures anymore" in err_str:
                    logger.warning(f"  ⚠️ Fetch de comisiones dinámicas omitido por Binance Demo API Limits. Usando Config defaults.")
                else:
                    logger.warning(f"  ⚠️ Could not fetch dynamic fees: {err_str}. Using Config defaults.")
            
        except ccxt.NetworkError as e:
            logger.error(f"❌ Network error initializing Futures settings: {e}")
            logger.error("   Bot will continue but manual verification recommended")
        except ccxt.ExchangeError as e:
            logger.error(f"❌ Exchange error initializing Futures settings: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected error initializing Futures settings: {e}")
            logger.error("   Please report this error with the full traceback")

    async def _ensure_leverage(self, symbol_id: str, target_leverage: int):
        """
        ⚡ NANO-EXECUTION: Programmatically sets leverage if it differs from cache.
        """
        current = self._leverage_cache.get(symbol_id)
        if current == target_leverage:
            return
            
        try:
            logger.info(f"⚡ [EXEC] Adjusting Leverage to {target_leverage}x for {symbol_id}")
            # Use Async Exchange for non-blocking execution
            await self.async_exchange.fapiPrivatePostLeverage({
                'symbol': symbol_id,
                'leverage': int(target_leverage)
            })
            self._leverage_cache[symbol_id] = target_leverage
            if hasattr(metrics, 'increment'):
                metrics.increment("leverage_adjustments")
            elif hasattr(metrics, 'inc'):
                metrics.inc("leverage_adjustments")
        except Exception as e:
            import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
            err_msg = str(e)
            if "No need to change" in err_msg:
                self._leverage_cache[symbol_id] = target_leverage
            else:
                logger.warning(f"⚠️ [LEVERAGE-FAIL] Could not set leverage to {target_leverage}x for {symbol_id}: {e}")

    async def _ensure_margin_isolated(self, symbol_id: str):
        """
        ⚡ NANO-EXECUTION: Programmatically sets margin type to ISOLATED to protect capital.
        """
        if self._margin_cache.get(symbol_id):
            return
            
        try:
            logger.info(f"⚡ [EXEC] Enforcing ISOLATED margin for {symbol_id}")
            await self.async_exchange.fapiPrivatePostMarginType({
                'symbol': symbol_id,
                'marginType': 'ISOLATED'
            })
            self._margin_cache[symbol_id] = True
        except Exception as e:
            import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
            err_msg = str(e)
            if "No need to change" in err_msg:
                self._margin_cache[symbol_id] = True
            else:
                logger.warning(f"⚠️ [MARGIN-FAIL] Could not set ISOLATED margin for {symbol_id}: {e}")

    async def _ghost_maker_execution(self, event, symbol_id, symbol_ccxt, side, pos_side, qty_precision, max_retries=5, fallback_timeout=3.0):
        """
        👻 MUTACIÓN 24: GHOST-MAKER (Rebate Arbitrage)
        Despliega órdenes LIMIT POST-ONLY que persiguen dinámicamente el BBO (Best Bid/Offer).
        Si el precio se escapa, ajusta. Si expira el fallback_timeout, tira MARKET panic.
        """
        import time
        start_time = time.time()
        order_type = 'LIMIT'
        
        for attempt in range(max_retries):
            # Check timeout
            if time.time() - start_time > fallback_timeout:
                logger.warning(f"👻⏰ [GHOST-MAKER] Timeout exceeded ({fallback_timeout}s) for {symbol_id}. Triggering Fallback to MARKET.")
                break
                
            try:
                # To exit a LONG (side=SELL), we want to post at ASK (to be Maker)
                # To exit a SHORT (side=BUY), we want to post at BID (to be Maker)
                px_tup = await self.guardian.get_fast_bid_ask(symbol_ccxt)
                target_px = px_tup[1] if side.upper() == 'SELL' else px_tup[0]
                
                # Safety fallback si orderbook is zero
                if target_px <= 0:
                     logger.warning(f"👻⚠️ [GHOST-MAKER] Orderbook returned 0. Falling back to trigger price.")
                     target_px = event.price if event.price else px_tup[0]
                     
                px_str = self.exchange.price_to_precision(symbol_ccxt, target_px)
                
                params = {
                    'symbol': symbol_id,
                    'side': side.upper(),
                    'type': order_type,
                    'quantity': qty_precision,
                    'positionSide': pos_side,
                    'newOrderRespType': 'RESULT',
                    'price': px_str,
                    'timeInForce': 'GTX', # POST-ONLY!
                    'reduceOnly': 'true'
                }
                
                logger.info(f"👻 [GHOST-MAKER] Deploying Post-Only Limit at {px_str} (Attempt {attempt+1}/{max_retries})")
                res = await self.async_exchange.fapiPrivatePostOrder(params)
                order_id = res.get('orderId')
                
                # Wait briefly for fill (Dynamic micro-delay)
                await asyncio.sleep(0.35)
                
                # Check status
                status_res = await self.async_exchange.fapiPrivateGetOrder({
                    'symbol': symbol_id,
                    'orderId': order_id
                })
                
                status = status_res.get('status')
                if status in ['FILLED', 'REJECTED']:
                    logger.info(f"👻✅ [GHOST-MAKER] Order {order_id} reached final status: {status}")
                    return status_res
                
                if status == 'CANCELED':
                    # Podría haber sido rechazada por ser Maker taker-crossing (GTX)
                    logger.warning(f"👻⚠️ [GHOST-MAKER] Order was canceled (GTX collision?). Retrying...")
                    continue
                
                # If NEW or PARTIALLY_FILLED, the market might have moved away
                logger.debug(f"👻🏃 [GHOST-MAKER] Market moved away (status: {status}). Cancelling {order_id} to chase...")
                await self.async_exchange.fapiPrivateDeleteOrder({
                    'symbol': symbol_id,
                    'orderId': order_id
                })
                
            except Exception as e:
                import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                err_str = str(e)
                if 'Margin is insufficient' in err_str:
                    logger.warning(f"👻⚠️ [GHOST-MAKER] Insufficient margin. Aborting Ghost-Maker.")
                    break
                if 'Post only order will be rejected' in err_str or 'Order would immediately match' in err_str:
                    logger.warning(f"👻⚠️ [GHOST-MAKER] GTX collision. Chasing...")
                    continue
                logger.warning(f"👻⚠️ [GHOST-MAKER] Error on attempt {attempt}: {e}")
                
        # FALLBACK A MARKET
        logger.warning(f"🚨 [GHOST-MAKER FALLBACK] Executing Taker MARKET for {symbol_id} to guarantee execution.")
        fallback_params = {
            'symbol': symbol_id,
            'side': side.upper(),
            'type': 'MARKET',
            'quantity': qty_precision,
            'positionSide': pos_side,
            'newOrderRespType': 'RESULT',
            'reduceOnly': 'true'
        }
        return await self.async_exchange.fapiPrivatePostOrder(fallback_params)

    def direct_fast_execute(self, symbol_id: str, side: str, order_type: str, qty: float, price: float = 0.0, timeInForce: str = "GTC", reduceOnly: bool = False, pos_side: str = "BOTH"):
        """
        🚀 ZERO-LATENCY BYPASS: Executes directly via Cython and HTTP/socket 
        without hitting the asyncio queue or engine async loop. (< 1us execution)
        """
        # 🏎️ FASE 6: C++ Native Bypass Injection
        if getattr(self, 'cpp_socket', None):
            try:
                res = self.cpp_socket.send_order_fast(
                    symbol_id,
                    side.upper(),
                    order_type.upper(),
                    float(qty),
                    float(price) if order_type.lower() == 'limit' else 0.0
                )
                import logging
                logging.getLogger('execution').info(f"⚡ [C++ EXEC] Order sent natively: {res}")
                return True
            except Exception as e:
                import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                import logging
                logging.getLogger('execution').warning(f"⚠️ [C++ EXEC] Fast C++ order failed: {e}. Falling back...")

        if getattr(self, 'fast_signer', None) is None or getattr(Config, 'BINANCE_USE_DEMO', False) or getattr(Config, 'BINANCE_USE_TESTNET', False):
            return False # Cython C-Executor not available or not in live mode
            
        endpoint, query, headers = self.fast_signer.build_fapi_order(
            symbol_id, side.upper(), order_type.upper(), float(qty), 
            float(price) if order_type.lower() == 'limit' else 0.0,
            timeInForce,
            bool(reduceOnly),
            pos_side
        )
        url = f"https://fapi.binance.com{endpoint}?{query}"
        
        if self.http_session is None or self.http_session.closed:
            import aiohttp
            connector = aiohttp.TCPConnector(keepalive_timeout=60, limit=100)
            self.http_session = aiohttp.ClientSession(connector=connector)
            
        # Fire-and-forget C-Socket Bypass (Non-blocking)
        async def _fire_and_forget():
            try:
                async with self.http_session.post(url, headers=headers) as resp:
                    pass # Execution confirmation handled by WebSocket User Data stream
            except Exception as e:
                import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                import logging
                logging.getLogger(__name__).error(f"Silent exception caught: {e}", exc_info=True)
                
        import asyncio
        asyncio.create_task(_fire_and_forget())
        return True

    async def execute_order(self, event):
        """
        🚀 SUPREMO-V3: ULTRA-LOW LATENCY EXECUTION
        QUÉ: Envía órdenes al exchange con precisión quirúrgica y mínima latencia.
        """
        if event.type != 'ORDER': return
        
        metadata = dict(event.metadata) if event.metadata else {}
        
        # ⚡ ZERO-QUEUE BYPASS GUARD
        if metadata.get('bypass_executed', False):
            logger.debug(f"⚡ [BYPASS GUARD] Order {event.symbol} already fired via Zero-Latency Bypass. Skipping async execution.")
            return

        # 🪐 [OMEGA REWRITE] Rust Execution Intercept
        # FORENSIC FIX: The Rust FFI `ffi_execute_order_bridge` currently hardcodes the Binance SPOT API
        # (/api/v3/order) and does not support Futures (fapi) or Hedge Mode (positionSide).
        # When BINANCE_USE_FUTURES is enabled, we MUST bypass the Rust intercept and use the
        # Python executor (which leverages Rust for signature generation anyway).
        if not getattr(Config, "BINANCE_USE_FUTURES", False):
            try:
                from core.rust_execution_bridge import ffi_execute_order_bridge
                api_key = Config.API_KEY
                secret_key = Config.API_SECRET
                success = ffi_execute_order_bridge(
                    api_key=api_key,
                    secret_key=secret_key,
                    symbol=event.symbol.replace("/", ""),
                    side=event.direction.value, # Enum to string
                    order_type='MARKET' if event.order_type.value == 'MARKET' else 'LIMIT',
                    qty=event.quantity,
                    price=event.price if event.order_type.value == 'LIMIT' else 0.0
                )
                if success:
                    logger.info(f"⚡ [RUST FFI] Fast Execution successful for {event.direction.value} {event.quantity} {event.symbol}")
                    return
            except Exception as e:
                logger.error(f"Rust FFI Execution intercept failed: {e}. Falling back to Python Executor.")        # 👻 [FASE III] MODO SOMBRA CUÁNTICO (SHADOW DEPLOYMENT)
        # QUÉ: Intercepta la orden justo antes de ir a Binance, simulando el fill
        #   y enviando el evento al Portfolio, pero guardando en un Flight Recorder local.
        if getattr(Config, 'SHADOW_MODE', False) or getattr(event, 'is_shadow', False):
            logger.info(f"👻 [SHADOW MODE] VIRTUAL EXECUTION: {event.direction.value} {event.quantity} {event.symbol} @ {event.price or 'MKT'}")
            
            # 🌌 [FASE III] INYECCIÓN DE PROFUNDIDAD DE MERCADO
            # Asumimos volumen base estático en primer nivel (1000 USDT para alts, 15000 USDT para majors)
            is_major = event.symbol in ['BTC/USDT', 'ETH/USDT', 'BTCUSDT', 'ETHUSDT']
            liquidity_threshold = 15000.0 if is_major else 1000.0
            
            fill_cost_base = event.price * event.quantity
            
            # Dinamic Slippage Proporcional
            base_slip_pct = 0.00015
            liquidity_impact = min(fill_cost_base / liquidity_threshold, 1.5) # Cap at 150%
            slip_pct = base_slip_pct + (0.0005 * liquidity_impact)
            
            # Rechazo por Liquidez si el slippage supera el 0.3%
            if slip_pct > 0.003:
                logger.error(f"📉 [SHADOW REJECT] Liquidity Vacuum. Orden de ${fill_cost_base:.2f} generaría slippage de {slip_pct*100:.2f}% (Máx 0.3%). Descartada.")
                return
                
            executed_price = event.price * (1 + slip_pct) if event.direction.value == 'BUY' else event.price * (1 - slip_pct)
            fill_cost = executed_price * event.quantity
            commission = fill_cost * 0.0005 # 0.05% Taker Fee
            
            pass
            pass
            fill_event = FillEvent(
                timeindex=datetime.now(timezone.utc),
                symbol=event.symbol,
                exchange="SHADOW_BINANCE",
                quantity=event.quantity,
                direction=event.direction,
                fill_cost=fill_cost,
                commission=commission,
                strategy_id=getattr(event, 'strategy_id', 'UNKNOWN'),
                fill_price=executed_price,
                order_id=f"SHADOW_{int(time.time()*1000)}",
                sl_pct=event.sl_pct,
                tp_pct=event.tp_pct,
                horizon=getattr(event, 'horizon', 'SCALPING'),
                leverage=getattr(event, 'leverage', 10),
                metadata=metadata,
                trade_id=getattr(event, 'trade_id', None),
                setup_type=getattr(event, 'setup_type', None),
                exit_reason=getattr(event, 'exit_reason', None)
            )
            
            # Log to Flight Recorder
            try:
                import json
                import os
                os.makedirs("results", exist_ok=True)
                with open("results/shadow_flight_recorder.jsonl", "a") as f:
                    rec = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "symbol": event.symbol,
                        "direction": event.direction.value,
                        "quantity": event.quantity,
                        "price": executed_price,
                        "notional": fill_cost,
                        "fee": commission,
                        "strategy": getattr(event, 'strategy_id', 'UNKNOWN')
                    }
                    f.write(json.dumps(rec) + "\n")
            except Exception as e:
                logger.error(f"Failed to write Shadow Recorder: {e}")
                
            if hasattr(self, 'events_queue') and self.events_queue:
                self.events_queue.put(fill_event)
            return

        
        # 🔫 [FASE 20] GRID BURST GENERATOR (Ametralladora L2)
        # QUÉ: Si la orden es marcada como "Grid Burst", la dividimos en 3 micro-órdenes.
        if metadata.get('is_grid_burst', False):
            logger.critical(f"🔫 [GRID BURST] Desplegando Ametralladora L2 para {event.symbol}!")
            import copy
            
            # Remove flag to prevent infinite recursion
            base_metadata = copy.deepcopy(metadata)
            base_metadata['is_grid_burst'] = False
            base_metadata['sniper_mode'] = True # Force aggressive pricing
            
            qty_per_bullet = event.quantity / 3.0
            price_offset = 0.0005 # 0.05% separation
            
            tasks = []
            for i in range(3):
                bullet_event = copy.deepcopy(event)
                bullet_event.quantity = qty_per_bullet
                bullet_event.metadata = copy.deepcopy(base_metadata)
                
                # Offset price based on step
                if bullet_event.price and bullet_event.price > 0:
                    offset_multiplier = 1 + (price_offset * i) if event.direction.value == 'BUY' else 1 - (price_offset * i)
                    bullet_event.price = bullet_event.price * offset_multiplier
                
                # Fire bullet concurrently
                tasks.append(asyncio.create_task(self.execute_order(bullet_event)))
            
            # Wait for all bullets to execute
            await asyncio.gather(*tasks, return_exceptions=True)
            return

        start_exec = time.perf_counter_ns()
        
        # 🌌 [CROSS-EXCHANGE] LAST-MICROSECOND VETO
        # QUÉ: Verificación final de O(1) en la memoria global antes de tocar Binance.
        # POR QUÉ: Si la señal tomó 5ms en procesarse, Coinbase pudo haber caído en ese tiempo.
        # CÓMO: Veto inmediato si el PDC va fuerte en contra.
        if not getattr(event, 'is_exit', False):
            try:
                from core.global_state import global_state
                if hasattr(global_state, 'cross_exchange_metrics'):
                    metrics = global_state.cross_exchange_metrics.get(event.symbol, {})
                    pdc = metrics.get('pdc_signal', 0.0)
                    side_str = event.direction.value.upper()
                    
                    if side_str == 'LONG' and pdc < -0.6:
                        logger.critical(f"🛑 [LATE-VETO] Ejecución LONG abortada en {event.symbol} por colapso repentino en Coinbase/Deribit (PDC: {pdc:.2f})")
                        return
                    elif side_str == 'SHORT' and pdc > 0.6:
                        logger.critical(f"🛑 [LATE-VETO] Ejecución SHORT abortada en {event.symbol} por pump repentino en Coinbase/Deribit (PDC: {pdc:.2f})")
                        return
            except Exception as e:
                import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                import logging
                logging.getLogger(__name__).error(f"Silent exception caught: {e}", exc_info=True)
        
        # 🧬 [Phase 19] SHADOW MODE INTERCEPTION
        # If this is a Shadow Order, we DO NOT send it to Binance.
        # We log it and send a Telegram notification so the user can track Prospect symbols.
        if getattr(event, 'is_shadow', False):
            logger.info(f"👻 [SHADOW] VIRTUAL EXECUTION: {event.direction.value} {event.quantity} {event.symbol} @ {event.price or 'MKT'}")
            try:
                from utils.notifier import Notifier
                horizon = getattr(event, 'horizon', 'UNKNOWN')
                confidence = getattr(event, 'ml_confidence', 0.0)
                msg = f"👻 *SHADOW PROSPECT SIGNAL* 👻\n"
                msg += f"Par: `{event.symbol}`\n"
                msg += f"Dirección: `{event.direction.value}`\n"
                msg += f"Horizonte: `{horizon}`\n"
                msg += f"Confianza: `{confidence*100:.1f}%`\n"
                msg += f"Estrategia: `{getattr(event, 'strategy_id', 'UNKNOWN')}`\n"
                msg += f"_(Esta operación es en papel para medir viabilidad)_"
                Notifier.send_telegram(msg, "INFO")
            except Exception as e:
                logger.error(f"Error sending shadow notification: {e}")
            return

        symbol = event.symbol
        symbol_ccxt = symbol.replace('USDT', '/USDT')
        
        # 🏎️ [NANO-SPEEDS] Initialize local mutable state from frozen event
        side = event.direction.value.lower()
        order_type = event.order_type.value.lower()
        price = event.price
        metadata = dict(event.metadata) if event.metadata else {}
        
        try:
            # 1. MARKETS LOADED CHECK
            if not self.async_exchange.markets:
                await self.async_exchange.load_markets()
            
            market = self.exchange.market(symbol_ccxt)
            symbol_id = market['id']
            
            # --- 🛡️ PHASE 15: PROGRAMMATIC LEVERAGE CHECK ---
            # Ensures the exchange is set to the correct leverage FOR THIS SPECIFIC HORIZON
            target_leverage = getattr(event, 'leverage', None)
            if target_leverage and Config.BINANCE_USE_FUTURES:
                # ⚡ FASE 10: Zero-Latency Fire-and-Forget
                asyncio.create_task(self._ensure_margin_isolated(symbol_id))
                asyncio.create_task(self._ensure_leverage(symbol_id, target_leverage))
            # 🚀 FASE 22: Nano-Timeframe Validation (Order Book Imbalance)
            # Solo para MICROSCALPING: Si el Order Book está demasiado en contra, abortamos antes de entrar
            pos_horizon = getattr(event, 'horizon', '')
            if pos_horizon == 'MICROSCALPING' and not getattr(event, 'is_exit', False):
                is_balanced, imbalance_reason = self.guardian.check_order_book_imbalance(symbol_ccxt, side, depth=10, threshold_ratio=2.5)
                if not is_balanced:
                    logger.warning(f"🛑 [NANO-VALIDATION] MICROSCALPING Aborted for {symbol}: {imbalance_reason}")
                    return
                    
            # 🛡️ PHASE II: ANTI-SLIPPAGE (Order Book Depth Check)
            # If MARKET order and liquidity is thin, downgrade to LIMIT or abort.
            if order_type == 'market':
                try:
                    bid, ask = await self.guardian.get_fast_bid_ask(symbol_ccxt)
                    
                    if bid > 0 and ask > 0:
                        spread_pct = (ask - bid) / bid
                        # Si el spread es > 0.1% (Muy alto para HFT), forzar LIMIT
                        if spread_pct > 0.001:
                            logger.warning(f"⚠️ High Spread ({spread_pct*100:.3f}%) detected for {symbol}. Downgrading to LIMIT.")
                            order_type = 'limit'
                            # Post at Best Bid/Ask
                            price = bid if side == 'sell' else ask
                except Exception as e:
                    import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                    logger.warning(f"⚠️ Liquidity Check Failed: {e}. Proceeding carefully.")
            
            # ✅ PHASE II: ATOMIC BALANCE VALIDATION
            # EXCELSIOR-TITAN: Prevent "Insufficient Funds" by pre-checking API balance.
            
            # 💸 PHASE 23: COST OPTIMIZATION (The Leak Preventer)
            from execution.cost_guard import CostGuard
            if not CostGuard.check_funding_leak(self.exchange, symbol_ccxt, side):
                logger.warning(f"🛑 [The Leak Preventer] Trade Aborted due to Toxic Funding.")
                return

            # 🧠 PHASE 23: SMART ORDER ROUTING (SOR)
            # Adapt Order Type based on Regime/Urgency (If not already forced to LIMIT by Anti-Slippage)
            # Accessing regime via portfolio reference if available
            current_regime = 'UNKNOWN'
            if self.portfolio and hasattr(self.portfolio, 'market_regime'):
                 current_regime = self.portfolio.market_regime or 'UNKNOWN'

            if order_type == 'market':
                # IF RANGING (Low Urgency) -> Try LIMIT (Maker)
                if current_regime == 'RANGING' or current_regime == 'CHOPPY':
                     logger.info(f"🧠 [SOR] Regime {current_regime} detected. Switching MARKET -> LIMIT (Maker Priority).")
                     order_type = 'limit'
                     # Post at Order Book Top
                     bid, ask = await self.guardian.get_fast_bid_ask(symbol_ccxt)
                     price = bid if side == 'buy' else ask
            
            if side == 'buy': # Only check for BUYS (Entry/Cover)
                try:
                    quote_currency = market['quote']
                    last_px = await self.guardian.get_fast_bid_ask(symbol_ccxt)
                    last_price = last_px[0] if last_px[0] > 0 else 1.0 # fallback bid
                    cost_est = event.quantity * (price if price else last_price)
                    
                    # ═══════════════════════════════════════════════════════════════
                    # FIX: LEVERAGE AWARENESS FOR $13 MICRO-ACCOUNT
                    # Divide cost_est by leverage to check required margin, not notional.
                    # ═══════════════════════════════════════════════════════════════
                    target_leverage = getattr(event, 'leverage', Config.BINANCE_LEVERAGE)
                    if Config.BINANCE_USE_FUTURES and target_leverage and target_leverage > 0:
                        required_margin = cost_est / target_leverage
                    else:
                        required_margin = cost_est
                    
                    # 🚀 ZERO-LATENCY BALANCE CHECK (Using local Portfolio)
                    if self.portfolio is not None:
                        available = getattr(self.portfolio, 'available_cash', 0.0)
                    else:
                        # Fallback for tests/standalone
                        params = {}
                        if Config.BINANCE_USE_FUTURES: params['type'] = 'future'
                        balance = await self.async_exchange.fetch_free_balance(params=params)
                        available = balance.get(quote_currency, 0.0)
                    
                    if available < required_margin:
                         logger.error(f"🚫 [ATOMIC] INSUFFICIENT FUNDS! Need Margin: {required_margin:.2f} {quote_currency} (Notional: {cost_est:.2f}), Avail: {available:.2f}")
                         if self.events_queue:
                             from core.events import ExecutionFailedEvent
                             asyncio.create_task(self.zmq_push.push(ExecutionFailedEvent(
                                 symbol=symbol, quantity=event.quantity, price=cost_est, direction=event.direction,
                                 reason="INSUFFICIENT_FUNDS", strategy_id=getattr(event, 'strategy_id', None), trade_id=getattr(event, 'trade_id', None)
                             )))
                         elif self.portfolio: 
                             client_id = event.metadata.get('client_order_id') if getattr(event, 'metadata', None) else None
                             self.portfolio.release_order_margin(amount=required_margin, order_id=client_id)
                         return
                         
                except Exception as e:
                    import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                    logger.warning(f"⚠️ Balance Check Skipped: {e}")

            # ⚡ ZERO-LATENCY WEBSOCKET CONNECTION MANAGER
            if self.ws_executor and not self.ws_executor.running:
                logger.info("⚡ [WS-EXEC] Lazy starting WS connection...")
                await self.ws_executor.start()
                for _ in range(5):
                    if self.ws_executor.is_ready(): break
                    await asyncio.sleep(0.1)


            # 2. EXIT PRIORITY (Rule 2.1) - Skip Guardian if EXIT
            is_exit = getattr(event, 'is_exit', False) or (getattr(event, 'strategy_id', '') == 'EMERGENCY_EXIT')
            
            if not is_exit:
                # Normal Signal: Run Guardian check
                
                # 🏎️ [L-001] Guardian Bypass Logic: Speed up Scalping Maker limits
                is_post_only = metadata.get('timeInForce') == 'GTX'
                is_scalp_limit = (order_type == 'limit' and getattr(event, 'horizon', '') == 'SCALPING')
                should_bypass = is_post_only or is_scalp_limit
                
                # 🌊 FASE 13: Microstructure Veto
                if self.data_provider and hasattr(self.data_provider, 'microstructure'):
                    internal_sym = symbol_ccxt.replace('/', '')
                    if internal_sym in self.data_provider.microstructure:
                        micro = self.data_provider.microstructure[internal_sym].get_metrics()
                        
                        # Veto si hay Spoofing en nuestra contra, o toxicidad extrema
                        if micro.get('is_toxic', False) or micro.get('is_spoofing', False) or micro.get('gamma_expansion_risk', False):
                            logger.warning(f"🛑 [MICRO-VETO] Entry blocked for {symbol_ccxt} due to Toxic Order Flow / Spoofing / Gamma Risk.")
                            if self.events_queue:
                                from core.events import ExecutionFailedEvent
                                asyncio.create_task(self.zmq_push.push(ExecutionFailedEvent(
                                    symbol=symbol, quantity=event.quantity, price=price or 0.0, direction=event.direction,
                                    reason="MICROSTRUCTURE_TOXIC_FLOW", strategy_id=getattr(event, 'strategy_id', None), trade_id=getattr(event, 'trade_id', None)
                                )))
                            elif self.portfolio: 
                                amt = event.quantity * (price if price else 0)
                                client_id = event.metadata.get('client_order_id') if getattr(event, 'metadata', None) else None
                                self.portfolio.release_order_margin(amount=amt, order_id=client_id)
                            return

                liquidity = await self.guardian.analyze_liquidity(symbol, event.quantity, event.direction.name, should_bypass)
                if not liquidity['is_safe']:
                    logger.warning(f"🛡️ [GUARDIAN] Order Blocked: {liquidity['reason']}")
                    if self.events_queue:
                        from core.events import ExecutionFailedEvent
                        asyncio.create_task(self.zmq_push.push(ExecutionFailedEvent(
                            symbol=symbol, quantity=event.quantity, price=price or 0.0, direction=event.direction,
                            reason=liquidity['reason'], strategy_id=getattr(event, 'strategy_id', None), trade_id=getattr(event, 'trade_id', None)
                        )))
                    elif self.portfolio: 
                        amt = event.quantity * (price if price else 0)
                        client_id = event.metadata.get('client_order_id') if getattr(event, 'metadata', None) else None
                        self.portfolio.release_order_margin(amount=amt, order_id=client_id)
                    return
                
                # ✅ PHASE II.6: VWAP-RELATIVE EXECUTION (Smart Execution)
                # If buying significantly above VWAP, switch to LIMIT to avoid chasing tops.
                try:
                    if self.data_provider:
                        bars = self.data_provider.get_latest_bars(symbol, n=15) # 15m VWAP
                    else:
                        bars = None # Fallback
                        
                    if bars is not None and len(bars) > 5:
                        # VWAP = Sum(Close * Vol) / Sum(Vol)
                        # Structured array: 'close', 'volume'
                        vsum = np.sum(bars['volume'])
                        if vsum > 0:
                            vwap_val = np.sum(bars['close'] * bars['volume']) / vsum
                            last_px = await asyncio.to_thread(self.guardian.get_fast_bid_ask, symbol_ccxt)
                            current_price = last_px[1] if side == 'buy' else last_px[0]
                            
                            # Logic: If BUY and Price > VWAP + 0.3% -> Switch MARKET to LIMIT
                            if side == 'buy' and current_price > vwap_val * 1.003:
                                if order_type == 'market' and not getattr(event, 'urgent', False):
                                    logger.info(f"📉 [VWAP] Price {current_price:.2f} > VWAP {vwap_val:.2f} (+0.3%). Switching to LIMIT/PASSIVE.")
                                    order_type = 'limit'
                                    price = current_price * 0.9995 # Bid side
                                    # Modified local order_type instead of frozen event
                except Exception as e:
                    import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                    import logging
                    logging.getLogger(__name__).error(f"Silent exception caught: {e}", exc_info=True) # Non-critical
                
                # --- PHASE 14: SMART-ORDER ROUTING (SOR) ---
                # Decide order type based on urgency and rebate priority
                is_urgent = getattr(event, 'urgent', False)
                rebate_priority = getattr(self.portfolio, 'rebate_priority', True)
                
                # FORENSIC FIX: Force Maker-Only for Scalping Entries
                is_scalping_entry = (getattr(event, 'horizon', '') == 'SCALPING' or getattr(event, 'horizon', '') == 'MICROSCALPING') and not getattr(event, 'is_exit', False) and not getattr(event, 'is_close', False)
                
                # [QUANTUM EVOLUTION] LOB Imbalance Sniping (Nanosecond Injection)
                imbalance = 0.0
                if self.data_provider and hasattr(self.data_provider, 'lob_imbalance') and symbol in self.data_provider.lob_imbalance:
                    imbalance = self.data_provider.lob_imbalance[symbol].get('imbalance', 0.0)
                    
                if is_scalping_entry:
                    logger.info(f"🛡️ [FORENSIC-SOR] SCALPING Entry Detected for {symbol}: Forcing ABSOLUTE LIMIT (GTX/Post-Only) to stop Fee Bleed on 13 USD capital.")
                    order_type = 'limit'
                    metadata['timeInForce'] = 'GTX'
                    
                    # Ensure price is perfectly pegged to the safe maker side
                    if self.data_provider and hasattr(self.data_provider, 'lob_imbalance') and symbol in self.data_provider.lob_imbalance:
                        if side == 'buy':
                            price = self.data_provider.lob_imbalance[symbol].get('bid_price', price)
                        else:
                            price = self.data_provider.lob_imbalance[symbol].get('ask_price', price)
                elif order_type == 'limit' and is_urgent:
                    logger.info("⚡ [SOR] Urgency detected: Switching LIMIT to MARKET to ensure entry.")
                    order_type = 'market'
                elif order_type == 'market' and rebate_priority and not is_urgent:
                    logger.info("💰 [SOR] Rebate Priority Active: Switching MARKET to LIMIT (Post-Only).")
                    order_type = 'limit'
                    # Post-Only flag for Binance (using local metadata instead of frozen event)
                    metadata['timeInForce'] = 'GTX' # GTX = Post Only
                
                smart_price = liquidity.get('avg_fill_price', price)
            else:
                # EXIT: High Priority
                strategy_name = getattr(event, 'strategy_id', '')
                is_emergency = strategy_name in ("EMERGENCY_EXIT", "KILL_SWITCH", "HARD_SL", "TIME_STOP")
                is_trailing = "TRAIL" in strategy_name
                is_active_exit = is_emergency or is_trailing or ("EXIT" in strategy_name) or ("PREDICTION" in strategy_name) or ("TURBO" in strategy_name) or ("CLOSE" in strategy_name)
                is_resting_tp = metadata.get('is_tp_limit', False) if metadata else False

                if is_emergency:
                    logger.critical(f"🚨 [EMERGENCY EXIT] {strategy_name} for {symbol}. Forcing MARKET panic fill.")
                    order_type = 'market'
                    smart_price = price
                elif Config.Execution.USE_LIMIT_BBO_EXITS and is_active_exit and not is_resting_tp:
                    logger.info(f"🛡️ [WEALTH-PHASE CHASE] Dynamic exit {strategy_name} for {symbol}. Forcing LIMIT chasing logic to save fees.")
                    order_type = 'limit'
                    metadata['timeInForce'] = 'GTX'
                    try:
                        bid, ask = await asyncio.to_thread(self.guardian.get_fast_bid_ask, symbol_ccxt)
                        smart_price = ask if side.lower() == 'sell' else bid
                    except Exception as e:
                        import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                        logger.warning(f"⚠️ [BBO-EXIT] Fallback to price: {e}")
                        smart_price = price
                elif is_active_exit and not is_resting_tp:
                    logger.info(f"⚡ [TAKER EXIT] Dynamic exit detected for {symbol} ({strategy_name}). Forcing MARKET to guarantee fill.")
                    order_type = 'market'
                    smart_price = price
                elif order_type == 'limit':
                    logger.info(f"🛡️ [FORENSIC-SOR] BBO Limit Exit for {symbol} ({strategy_name}). Positioning at Maker side.")
                    try:
                        bid, ask = await asyncio.to_thread(self.guardian.get_fast_bid_ask, symbol_ccxt)
                        # To exit LONG (SELL), we want to be at the ASK (to be a maker)
                        # To exit SHORT (BUY), we want to be at the BID (to be a maker)
                        smart_price = ask if side.lower() == 'sell' else bid
                    except Exception as e:
                        import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                        logger.warning(f"⚠️ [BBO-EXIT] Could not fetch orderbook: {e}. Fallback to trigger price.")
                        smart_price = price
                else:
                    smart_price = price

            # 3. SURGICAL PRECISION (Roundings)
            qty_precision = self.exchange.amount_to_precision(symbol_ccxt, event.quantity)
            final_qty = float(qty_precision)
            
            # Sniper Logic for LIMIT orders (Aggressive pricing to capture liquidity)
            # PHASE 13: Enhanced V3 Sniper
            # Si es una entrada agresiva (Imbalance > 3.0), empujamos el precio para asegurar el fill.
            if order_type == 'limit':
                spread_adj = 0.0001 # Default 0.01% bias
                
                # Check for Sniper Condition in metadata
                if getattr(event, 'metadata', None) and event.metadata.get('sniper_mode'):
                    spread_adj = 0.0003 # 0.03% more aggressive
                    logger.info(f"🎯 [SNIPER_V3] Aggressive Entry engaged for {symbol}")

                # 🧟 ZOMBIE FEATURE INTEGRATION: Dynamic Limit Offset from PredictionTracker
                if metadata and 'limit_offset_pct' in metadata:
                    spread_adj = metadata['limit_offset_pct']
                    logger.info(f"🎯 [PREDICTIVE LIMIT] Using dynamic limit offset {spread_adj:.4%} for {symbol}")

                # [PHASE 5] Scalping Optimization: No aggressive pushing for MUST-MAKER (GTX) orders
                is_post_only = metadata.get('timeInForce') == 'GTX'
                if is_post_only:
                    spread_adj = 0.0 # No aggressive pushing for Maker orders
                    logger.info(f"💰 [MAKER_V5] Post-Only engaged. Pricing safely to avoid crossing spread.")
                    
                    # Ensure price is at the optimal side of the order book to avoid GTX rejection
                    try:
                        bid, ask = await asyncio.to_thread(self.guardian.get_fast_bid_ask, symbol_ccxt)
                        if side == 'buy':
                            smart_price = min(smart_price, bid) if smart_price is not None else bid
                        else:
                            smart_price = max(smart_price, ask) if smart_price is not None else ask
                    except Exception as e:
                        import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                        logger.warning(f"⚠️ Could not fetch orderbook for Maker pricing: {e}")

                if smart_price is None:
                    try:
                        bid, ask = await asyncio.to_thread(self.guardian.get_fast_bid_ask, symbol_ccxt)
                        smart_price = bid if side == 'buy' else ask
                    except Exception as e:
                        import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                        logger.warning(f"⚠️ Fallback pricing failed: {e}")
                        return

                if side == 'buy': smart_price *= (1 + spread_adj)
                else: smart_price *= (1 - spread_adj)
                
            # 🏎️ [MUTACIÓN 36] QUANTUM PING-DRIFT COMPENSATOR
            # Detectamos si el data_provider tiene latencia WebSocket
            # Si latency > 30ms y somos 'limit', empujamos un poco más agresivo 
            # para no quedar colgados por el drift temporal.
            if order_type == 'limit' and self.data_provider and hasattr(self.data_provider, 'get_ws_latency'):
                ws_latency = self.data_provider.get_ws_latency()
                if ws_latency > 30.0:
                    drift_compensator = 0.0001 * min(ws_latency / 100.0, 5.0) # Max 0.05% compensation
                    logger.info(f"⏱️ [PING-DRIFT] Latency {ws_latency:.1f}ms. Applying +{drift_compensator:.4%} drift compensation to {symbol} (Anticipatory Slippage).")
                    if side == 'buy': smart_price *= (1 + drift_compensator)
                    else: smart_price *= (1 - drift_compensator)
            
            price_precision = self.exchange.price_to_precision(symbol_ccxt, smart_price)
            final_price = float(price_precision)

            # 3.5 [DF-C9] FAT FINGER PROTECTION — Price Sanity Check
            # QUÉ: Bloquea órdenes con precio que se desvía >5% del mercado.
            # POR QUÉ: Un bug en la señal o datos corruptos podría enviar
            #   price=0.0 o price=last*100, causando pérdida catastrófica.
            # CÓMO: Compara final_price vs último precio conocido del portfolio
            #   o del Guardian. Si la desviación >FAT_FINGER_THRESHOLD, bloquea.
            FAT_FINGER_THRESHOLD = Config.RISK_FAT_FINGER_THRESHOLD if hasattr(Config, 'RISK_FAT_FINGER_THRESHOLD') else 0.05
            reference_price = None
            try:
                # Try portfolio's last known price first (fastest, no API call)
                if self.portfolio and symbol in self.portfolio.positions:
                    reference_price = self.portfolio.positions[symbol].get('current_price')
                # Fallback: Guardian's order book mid-price
                if not reference_price or reference_price <= 0:
                    px_tup = await asyncio.to_thread(self.guardian.get_fast_bid_ask, symbol)
                    reference_price = px_tup[0] if side == 'sell' else px_tup[1]
                # Fallback: ticker
                if not reference_price or reference_price <= 0:
                    ticker = await self.async_exchange.fetch_ticker(symbol_ccxt)
                    reference_price = float(ticker.get('last', 0))
            except Exception as e:
                import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                logger.warning(f"⚠️ [FAT FINGER] Could not get reference price for {symbol}: {e}")
                reference_price = None

            if reference_price and reference_price > 0 and final_price > 0:
                deviation = abs(final_price - reference_price) / reference_price
                if deviation > FAT_FINGER_THRESHOLD:
                    logger.critical(
                        f"🚨 [DF-C9] FAT FINGER BLOCKED: {symbol} "
                        f"order_price={final_price:.6f} vs market={reference_price:.6f} "
                        f"(deviation={deviation:.2%} > {FAT_FINGER_THRESHOLD:.0%})"
                    )
                    if side == 'buy':
                        if self.events_queue:
                            from core.events import ExecutionFailedEvent
                            asyncio.create_task(self.zmq_push.push(ExecutionFailedEvent(
                                symbol=symbol, quantity=event.quantity, price=event.price or 0.0, direction=event.direction,
                                reason="FAT_FINGER", strategy_id=getattr(event, 'strategy_id', None), trade_id=getattr(event, 'trade_id', None)
                            )))
                        elif self.portfolio:
                            amt = event.quantity * (event.price or 0)
                            client_id = event.metadata.get('client_order_id') if getattr(event, 'metadata', None) else None
                            self.portfolio.release_order_margin(amount=amt, order_id=client_id)
                    return
                elif deviation > FAT_FINGER_THRESHOLD * 0.5:  # Warn at 2.5%
                    logger.warning(
                        f"⚠️ [FAT FINGER] Elevated deviation: {symbol} "
                        f"price={final_price:.6f} vs market={reference_price:.6f} "
                        f"({deviation:.2%})"
                    )

            # 3.8. CANCEL TP LIMIT PRIOR TO PANIC EXIT
            _cancel_tp = metadata.get("cancel_tp_first", False)
            if _cancel_tp and (getattr(event, 'is_exit', False) or getattr(event, 'is_close', False)):
                logger.info(f"🗑️ [PREDICTIVE LIMIT] Cancelling resting TP limit for {symbol_ccxt} before Market exit...")
                try:
                    # ccxt soporta cancel_all_orders para casi todos
                    await asyncio.wait_for(self.async_exchange.cancel_all_orders(symbol_ccxt), timeout=4.0)
                    logger.info(f"✅ [PREDICTIVE LIMIT] Pending TP orders cancelled for {symbol_ccxt}.")
                except Exception as e:
                    import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                    logger.warning(f"⚠️ Could not cancel previous TP Limit: {e}")

            # 4. SEND ORDER
            logger.info(f"⚡ [EXEC] {order_type.upper()} {side.upper()} {symbol} | Qty: {final_qty} | P: {final_price}")
            
            # 👻 MUTACIÓN 24: GHOST-MAKER INTERCEPTOR
            is_ghost_maker = metadata.get('use_ghost_maker', False) if metadata else False
            if is_ghost_maker and Config.BINANCE_USE_FUTURES:
                pos_side = 'LONG' if side.upper() == 'SELL' else 'SHORT' # Exits invert direction
                try:
                    order_raw = await self._ghost_maker_execution(event, symbol_id, symbol_ccxt, side, pos_side, qty_precision)
                    if order_raw and 'orderId' in order_raw:
                        logger.info(f"👻🏁 [GHOST-MAKER] Successful execution for {symbol}. OrderID: {order_raw['orderId']}")
                        # Publicar evento de orden exitosa (simulando flow)
                        if self.events_queue:
                            from core.events import OrderEvent
                            asyncio.create_task(self.zmq_push.push(OrderEvent(
                                symbol=symbol, order_type='LIMIT', quantity=final_qty, direction=event.direction,
                                price=float(order_raw.get('avgPrice', order_raw.get('price', 0))), 
                                status='FILLED' if order_raw.get('status') == 'FILLED' else 'SUBMITTED',
                                trade_id=getattr(event, 'trade_id', None),
                                strategy_id="GHOST_MAKER_EXIT"
                            )))
                    return
                except Exception as e:
                    logger.error(f"🚨 [GHOST-MAKER CRASH] {e}. Falling back to normal execution.")
            
            if Config.BINANCE_USE_FUTURES:
                # Use Raw API for minimum latency
                # 🛡️ PHOENIX V3: HEDGE MODE ENFORCEMENT
                is_exit_signal = getattr(event, 'is_exit', False) or (getattr(event, 'strategy_id', '') == "EMERGENCY_EXIT")
                
                pos_side = 'LONG' if side.upper() == 'BUY' else 'SHORT'
                if getattr(event, 'is_close', False) or is_exit_signal:
                    # Closing order. If we sell to close, it's a LONG position. If we buy to close, it's a SHORT position.
                    pos_side = 'LONG' if side.upper() == 'SELL' else 'SHORT'
                else:
                    # OPEN order. If we buy, we enter LONG. If we sell, we enter SHORT.
                    pos_side = 'LONG' if side.upper() == 'BUY' else 'SHORT'
                    
                params = {
                    'symbol': symbol_id,
                    'side': side.upper(),
                    'type': order_type.upper(),
                    'quantity': qty_precision,
                    'positionSide': pos_side,
                    'newOrderRespType': 'RESULT',
                    'recvWindow': 60000
                }
                
                # FORENSIC FIX: Inject Horizon into clientOrderId for perfect WebSocket tracking
                pos_horizon = getattr(event, 'horizon', 'SCALPING')
                uid = uuid.uuid4().hex[:8]
                metadata_client_id = metadata.get('client_order_id') if metadata else None
                if metadata_client_id:
                    params['newClientOrderId'] = metadata_client_id
                else:
                    # Tagging specifically for TG_{HORIZON}_{UUID} routing
                    prefix = "TG_SCL_" if pos_horizon == 'SCALPING' else "TG_SWG_" if pos_horizon == 'SWING' else "TG_MIC_"
                    params['newClientOrderId'] = f"{prefix}{uid}"
                
                if order_type == 'limit':
                    params['price'] = price_precision
                    # Phase 5: Enforce Post-Only if intended
                    is_post_only = metadata.get('timeInForce') == 'GTX'
                    params['timeInForce'] = 'GTX' if is_post_only else 'GTC'
                
                if getattr(event, 'is_close', False) or is_exit_signal:
                    params['reduceOnly'] = 'true'
                
                # Para la orden PREDICTIVE_TP explícita
                if metadata.get("is_tp_limit", False):
                    params['reduceOnly'] = 'true'
                
                # ccxt already imported at top
                
                # --- 18. MICRO-GRID SWARM INJECTION ---
                is_swarm = (pos_horizon == 'MICROSCALPING' and order_type == 'limit' and not is_exit_signal and not getattr(event, 'is_close', False))
                order_raw = None
                
                if is_swarm:
                    logger.info(f"🐝 [MUTACION 18] Micro-Grid Swarm Initiated for {symbol}. Fragmenting {qty_precision} into 3 nodes.")
                    tick_vol = 0.0002
                    if metadata and 'metrics' in metadata:
                        tick_vol = metadata.get('metrics', {}).get('order_flow', {}).get('tick_volatility', 0.0002)
                    
                    grid_weights = [0.4, 0.3, 0.3]
                    multipliers = [0, 1.5, 3.0] # Spaced by 1.5x tick_vol
                    tasks = []
                    
                    for i, w in enumerate(grid_weights):
                        sub_qty = self.exchange.amount_to_precision(symbol_ccxt, event.quantity * w)
                        if side.upper() == 'BUY':
                            sub_px = final_price * (1 - (tick_vol * multipliers[i]))
                        else:
                            sub_px = final_price * (1 + (tick_vol * multipliers[i]))
                            
                        sub_px_str = self.exchange.price_to_precision(symbol_ccxt, sub_px)
                        sub_params = params.copy()
                        sub_params['quantity'] = sub_qty
                        sub_params['price'] = sub_px_str
                        sub_params['newClientOrderId'] = f"{params['newClientOrderId']}_{i}"
                        tasks.append(self.async_exchange.fapiPrivatePostOrder(sub_params))
                    
                    try:
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        for res in results:
                            if isinstance(res, dict) and 'orderId' in res:
                                if order_raw is None: order_raw = res
                        if not order_raw:
                            raise RuntimeError(f"All Grid orders failed: {results}")
                    except Exception as e:
                        logger.error(f"⚠️ [SWARM ERROR]: {e}")
                        raise

                # ⚡ FASE 12.5: ZERO-LATENCY WEBSOCKET EXECUTION
                elif getattr(self, 'ws_executor', None) and self.ws_executor.is_ready():
                    logger.info("⚡ [WS-EXEC] Routing order through Zero-Latency WebSocket...")
                    max_bbo_retries = 3
                    bbo_attempts = 0
                    
                    while bbo_attempts < max_bbo_retries:
                        try:
                            order_raw = await self.ws_executor.place_order(params)
                            break
                        except Exception as e:
                            import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                            err_msg = str(e)
                            if "immediately match" in err_msg.lower() and params.get('timeInForce') == 'GTX':
                                bbo_attempts += 1
                                if bbo_attempts >= max_bbo_retries:
                                    logger.warning(f"⚠️ [BBO MAKER REJECTED] WS order {side} {symbol_id} would cross spread. Fallback to GTC.")
                                    params['timeInForce'] = 'GTC'
                                    continue
                                
                                logger.info(f"🔄 [BBO CHASE] Repricing {symbol_id} GTX WS-order (Attempt {bbo_attempts}/{max_bbo_retries})...")
                                px_tup = await asyncio.to_thread(self.guardian.get_fast_bid_ask, symbol_ccxt)
                                new_price = px_tup[0] if side.upper() == 'BUY' else px_tup[1]
                                params['price'] = self.exchange.price_to_precision(symbol_ccxt, new_price)
                                await asyncio.sleep(0.05)
                            else:
                                logger.warning(f"⚠️ [WS-EXEC] WS Order Failed ({e}). Falling back to Cython/REST.")
                                order_raw = None
                                break

                    # If WS failed for non-GTX reasons, we fall back to CCXT
                    if not order_raw:
                        logger.info("📡 [WS-FALLBACK] Executing via CCXT REST...")
                        order_raw = await self.async_exchange.fapiPrivatePostOrder(params)

                # ⚡ NATIVE C-LEVEL EXECUTION (AITS Phase 4)
                elif getattr(self, 'fast_signer', None) and not getattr(Config, 'BINANCE_USE_DEMO', False) and not Config.BINANCE_USE_TESTNET:
                    if self.http_session is None or self.http_session.closed:
                        # Re-use connection pool for sub-millisecond execution
                        connector = aiohttp.TCPConnector(keepalive_timeout=60, limit=100)
                        self.http_session = aiohttp.ClientSession(connector=connector)
                        
                    endpoint, query, headers = self.fast_signer.build_fapi_order(
                        symbol_id, side.upper(), order_type.upper(), float(qty_precision), 
                        float(price_precision) if order_type == 'limit' else 0.0,
                        params.get('timeInForce', 'GTC'),
                        bool(params.get('reduceOnly', False)),
                        pos_side
                    )
                    url = f"https://fapi.binance.com{endpoint}?{query}"
                    
                    max_bbo_retries = 3
                    
                    try:
                        bbo_attempts = 0
                        while bbo_attempts < max_bbo_retries:
                            async with self.http_session.post(url, headers=headers) as resp:
                                order_raw = await resp.json()
                                if 'code' in order_raw and int(order_raw['code']) < 0:
                                    err_msg = order_raw.get('msg', '')
                                    if "immediately match" in err_msg.lower() and params.get('timeInForce') == 'GTX':
                                        bbo_attempts += 1
                                        if bbo_attempts >= max_bbo_retries:
                                            logger.warning(f"⚠️ [BBO MAKER REJECTED] Futures order {side} {symbol_id} would cross spread. Max retries reached. Fallback to GTC to secure fill.")
                                            endpoint, query, headers = self.fast_signer.build_fapi_order(
                                                symbol_id, side.upper(), order_type.upper(), float(qty_precision), 
                                                float(price_precision) if order_type == 'limit' else 0.0,
                                                "GTC",
                                                bool(params.get('reduceOnly', False)),
                                                pos_side
                                            )
                                            url = f"https://fapi.binance.com{endpoint}?{query}"
                                            continue # Try GTC
                                        
                                        # Reprice and retry BBO
                                        logger.info(f"🔄 [BBO CHASE] Repricing {symbol_id} GTX order (Attempt {bbo_attempts}/{max_bbo_retries})...")
                                        px_tup = await asyncio.to_thread(self.guardian.get_fast_bid_ask, symbol_ccxt)
                                        new_price = px_tup[0] if side.upper() == 'BUY' else px_tup[1]
                                        price_precision = self.exchange.price_to_precision(symbol_ccxt, new_price)
                                        
                                        endpoint, query, headers = self.fast_signer.build_fapi_order(
                                            symbol_id, side.upper(), order_type.upper(), float(qty_precision), 
                                            float(price_precision),
                                            "GTX",
                                            bool(params.get('reduceOnly', False)),
                                            pos_side
                                        )
                                        url = f"https://fapi.binance.com{endpoint}?{query}"
                                        await asyncio.sleep(0.05)
                                    else:
                                        logger.error(f"Binance API Error: {order_raw}")
                                        raise ccxt.ExchangeError(f"Binance API Error: {err_msg}")
                                else:
                                    break # Order succeeded
                    except asyncio.TimeoutError:
                        logger.critical(f"🛑 [TIMEOUT] Futures order {side} {symbol_id} hung >9s! OS Network Blocked.")
                        raise RuntimeError("API Timeout / Disconnect in execution")
                else:
                    max_bbo_retries = 3
                    bbo_attempts = 0
                    while bbo_attempts < max_bbo_retries:
                        try:
                            order_raw = await asyncio.wait_for(self.async_exchange.fapiPrivatePostOrder(params), timeout=9.0)
                            break # Success
                        except asyncio.TimeoutError:
                            logger.critical(f"🛑 [TIMEOUT] Futures order {side} {symbol_id} hung >9s! OS Network Blocked.")
                            raise RuntimeError("API Timeout / Disconnect in execution")
                        except ccxt.ExchangeError as e:
                            if "immediately match" in str(e).lower() and params.get('timeInForce') == 'GTX':
                                bbo_attempts += 1
                                if bbo_attempts >= max_bbo_retries:
                                    if pos_horizon in ('MICROSCALPING', 'SCALPING'):
                                        logger.warning(f"⚠️ [BBO MAKER REJECTED] {side} {symbol_id} crossed spread. STRICT POST_ONLY for {pos_horizon}. ABORTING to prevent taker fee.")
                                        raise RuntimeError(f"Strict POST_ONLY rule violated for {pos_horizon}. Order aborted to prevent Taker fees.")
                                    else:
                                        logger.warning(f"⚠️ [BBO MAKER REJECTED] Futures order {side} {symbol_id} would cross spread. Max retries. Fallback to GTC to secure fill.")
                                        params['timeInForce'] = 'GTC'
                                        continue
                                
                                logger.info(f"🔄 [BBO CHASE] Repricing {symbol_id} GTX order (Attempt {bbo_attempts}/{max_bbo_retries})...")
                                px_tup = await asyncio.to_thread(self.guardian.get_fast_bid_ask, symbol_ccxt)
                                new_price = px_tup[0] if side.upper() == 'BUY' else px_tup[1]
                                params['price'] = self.exchange.price_to_precision(symbol_ccxt, new_price)
                                await asyncio.sleep(0.05)
                            else:
                                raise e
                order = order_raw # Simplified mapping
            else:
                # SPOT
                pass
                try:
                    spot_params = {}
                    if order_type == 'limit':
                        is_post_only = metadata.get('timeInForce') == 'GTX'
                        spot_params['timeInForce'] = 'GTX' if is_post_only else 'GTC'
                        
                    order = await asyncio.wait_for(self.async_exchange.create_order(
                        symbol=symbol_ccxt,
                        type=order_type,
                        side=side,
                        amount=final_qty,
                        price=final_price if order_type == 'limit' else None,
                        params=spot_params
                    ), timeout=9.0)
                except asyncio.TimeoutError:
                    logger.critical(f"🛑 [TIMEOUT] Spot order {side} {symbol_id} hung >9s! OS Network Blocked.")
                    raise RuntimeError("API Timeout / Disconnect in execution")
                except ccxt.OrderImmediatelyFillable:
                    logger.warning(f"⚠️ [BBO MAKER REJECTED] Order {side} {symbol_id} would cross the spread immediately. GTX blocked execution. (Chase Order needed)")
                    # The pending cash remains tied unless cancelled properly via Portfolio, or OrderManager handles the omission.
                    return  # Silent return to avoid crashing the event loop, let Portfolio handle orphan resolution
                if 'info' in order: order = order['info']

            # 5. PROCESS RESPONSE & EMIT FILL
            end_exec = time.perf_counter_ns()
            exec_latency_ns = end_exec - start_exec
            exec_latency = exec_latency_ns / 1_000_000.0
            latency_monitor.track_hotpath(exec_latency_ns)
            
            # [SOVEREIGN-DEPLOY] Latency Guard (Average > 150ms blocks MICROSCALPING)
            if not hasattr(self, '_latency_samples'):
                self._latency_samples = []
            
            self._latency_samples.append(exec_latency)
            if len(self._latency_samples) > 5:
                self._latency_samples.pop(0)
                
            avg_latency = sum(self._latency_samples) / len(self._latency_samples)
            
            if avg_latency > 150.0:
                self.latency_violations = getattr(self, 'latency_violations', 0) + 1
                logger.warning(f"⚠️ [LATENCY] Avg latency {avg_latency:.2f}ms (>150ms). Violation {self.latency_violations}/3. MICROSCALPING disabled temporarily.")
                if self.portfolio:
                    # Temporarily instruct portfolio/risk to block scalping via a flag
                    self.portfolio.microscalping_disabled_until = time.time() + 60.0 # block for 60s
                
                if self.latency_violations >= 3:
                    logger.critical(f"🚨 [PANIC] 3 Consecutive High-Latency averages! Engaging PASSIVE MODE/LOCK.")
                    try:
                        with open("EMERGENCY_KILL_SWITCH.lock", "w") as f:
                            f.write(f"LATENCY_PANIC: {avg_latency:.2f}ms")
                    except Exception as e:
                        logger.error(f"Failed to write lock: {e}")
            else:
                self.latency_violations = 0 # Reset on healthy execution
            
            fill_price = float(order.get('avgPrice', final_price if order_type == 'limit' else 0.0))
            filled_qty = float(order.get('executedQty', final_qty))
            order_id = str(order.get('orderId', ''))
            is_fully_filled = (filled_qty >= event.quantity * 0.9999)  # Tolerance for floating point
            
            logger.info(f"✅ Order OK: {order_id} | Filled: {filled_qty} @ {fill_price} in {exec_latency:.2f}ms")
            
            # 🔥 START PHOENIX CHASE LOGIC BACKGROUND TASK (If LIMIT && Maker)
            if order_type == 'limit' and not getattr(Config, 'BINANCE_USE_DEMO', False):
                # Don't chase offline or simulated executions
                asyncio.create_task(
                    self._chase_order_loop(
                        symbol_ccxt=symbol_ccxt,
                        symbol_id=symbol_id,
                        side=side,
                        qty_precision=qty_precision,
                        params=params if Config.BINANCE_USE_FUTURES else spot_params,
                        original_order=order,
                        event=event,
                        max_chases=5
                    )
                )
                
                # 🌊 FASE 12: ADVERSE SELECTION VETO MONITOR (Only for entries)
                is_exit_or_close = getattr(event, 'is_exit', False) or getattr(event, 'is_close', False)
                if not is_exit_or_close:
                    asyncio.create_task(
                        self._adverse_selection_monitor(
                            symbol_ccxt=symbol_ccxt,
                            symbol_id=symbol_id,
                            side=side,
                            order_id=order_id,
                            original_price=final_price,
                            event=event
                        )
                    )
            
            # [DF-C7] PARTIAL FILL DETECTION & WARNING
            if not is_fully_filled and filled_qty > 0:
                fill_ratio = filled_qty / event.quantity if event.quantity > 0 else 0
                logger.warning(
                    f"⚠️ [DF-C7] PARTIAL FILL: {symbol} filled {filled_qty}/{event.quantity} "
                    f"({fill_ratio:.1%}). SL/TP will use ACTUAL filled qty."
                )
            elif filled_qty <= 0:
                logger.warning(f"⚠️ [DF-C7] ZERO FILL: {symbol} order {order_id} returned 0 qty. Skipping fill event.")
                return

            # Create Fill Event
            # ═══════════════════════════════════════════════════════════════
            # FORENSIC FIX #2: Propagate metadata from OrderEvent → FillEvent
            # QUÉ: Copia metadata (dollar_size, entry_mode, order_type) al fill.
            # POR QUÉ: Sin esto, Portfolio.update_fill() no puede:
            #   1) Liberar pending_cash con el monto exacto reservado (dollar_size)
            #   2) Aplicar el fee rate correcto (Maker vs Taker)
            # PARA QUÉ: Eliminar margin leak acumulativo y phantom fee drain.
            # ═══════════════════════════════════════════════════════════════
            _order_metadata = dict(getattr(event, 'metadata', {}) or {})
            # ═══════════════════════════════════════════════════════════════
            # FORENSIC-V35: GRANULAR ORDER TYPE ENRICHMENT
            # QUÉ: Enriquece 'actual_order_type' con sub-tipo descriptivo.
            # POR QUÉ: 'limit' o 'market' es insuficiente para auditar
            #   decisiones de ejecución. El usuario necesita saber si fue
            #   Post-Only (GTX), BBO, SOR-downgrade, Emergency, TP/SL, etc.
            # PARA QUÉ: Telegram muestra el tipo exacto de ejecución.
            # ═══════════════════════════════════════════════════════════════
            is_gtx = metadata.get('timeInForce') == 'GTX'
            is_tp_limit = metadata.get('is_tp_limit', False)
            is_reduce = getattr(event, 'is_close', False) or getattr(event, 'is_exit', False)
            
            if order_type == 'limit':
                if is_tp_limit:
                    enriched_type = 'LIMIT_TP'
                elif is_gtx:
                    enriched_type = 'LIMIT_POST_ONLY'
                elif is_reduce:
                    enriched_type = 'LIMIT_BBO_EXIT'
                else:
                    enriched_type = 'LIMIT_BBO'
            elif order_type == 'market':
                if is_reduce:
                    enriched_type = 'MARKET_EXIT'
                elif getattr(event, 'strategy_id', '') == 'EMERGENCY_EXIT':
                    enriched_type = 'MARKET_EMERGENCY'
                else:
                    enriched_type = 'MARKET_SOR'
            else:
                enriched_type = order_type.upper()
            
            _order_metadata['actual_order_type'] = order_type  # raw for fee calc
            _order_metadata['enriched_order_type'] = enriched_type  # display
            _order_metadata['is_exit'] = getattr(event, 'is_exit', False)
            _order_metadata['is_close'] = getattr(event, 'is_close', False)
            
            fill_event = FillEvent(
                timeindex=datetime.now(timezone.utc),
                symbol=symbol,
                exchange='BINANCE',
                quantity=filled_qty,
                direction=event.direction, # Using Typed Enum
                fill_cost=filled_qty * fill_price,
                fill_price=fill_price,
                order_id=order_id,
                commission=None,
                strategy_id=getattr(event, 'strategy_id', 'Unknown'),
                horizon=getattr(event, 'horizon', 'SCALPING'),
                sl_pct=getattr(event, 'sl_pct', None),
                tp_pct=getattr(event, 'tp_pct', None),
                # Phase 31: Partial Fill Logic
                is_closed=is_fully_filled,
                # ML Telemetry
                ml_confidence=getattr(event, 'ml_confidence', None),
                predicted_duration=getattr(event, 'predicted_duration', None),
                # Forensic Details
                setup_type=getattr(event, 'setup_type', None),
                exit_reason=getattr(event, 'exit_reason', None),
                order_type=getattr(event, 'order_type', None),
                trade_id=getattr(event, 'trade_id', None),
                thought_id=getattr(event, 'thought_id', None),
                strategy_version=getattr(event, 'strategy_version', '1.0.0'),
                # FORENSIC FIX #2: Carry metadata for fee attribution + margin release
                metadata=_order_metadata,
            )
            
            if self.events_queue:
                self.events_queue.put(fill_event)
            else:
                await self.events_queue.put(fill_event)
            
            # 6. TRACKING & PROTECTIVE ORDERS
            # ═══════════════════════════════════════════════════════════════
            # BBO ARCHITECTURE: Track ALL limit orders (entries AND exits)
            # QUÉ: Registra órdenes LIMIT en OrderManager para chase lifecycle.
            # POR QUÉ: Exits LIMIT BBO necesitan monitoring para fallback MARKET.
            # ═══════════════════════════════════════════════════════════════
            if self.order_manager and order_id and order_type == 'limit':
                _evt_metadata = dict(metadata) if metadata else {}
                _evt_metadata['is_exit_order'] = getattr(event, 'is_exit', False) or getattr(event, 'is_close', False)
                _evt_metadata['horizon'] = getattr(event, 'horizon', 'SCALPING')
                self.order_manager.track_order(
                    order_id, 
                    symbol, 
                    order_type, 
                    side, 
                    final_price, 
                    final_qty, 
                    getattr(event, 'strategy_id', 'Unknown'),
                    ttl=getattr(event, 'ttl', None),
                    metadata=_evt_metadata
                )
            
            # Exchange-Based Protective Orders (Failsafe Layer 3)
            # [DF-C7 FIX] Use filled_qty (actual) NOT final_qty (requested)
            # POR QUÉ: If only 30% filled, placing SL for 100% would attempt
            #   to close a position larger than what we hold → Binance error
            #   or phantom position risk.
            # BBO FIX: Skip protective orders for EXIT fills (they ARE the exit)
            is_exit_fill = getattr(event, 'is_exit', False) or getattr(event, 'is_close', False)
            if Config.BINANCE_USE_FUTURES and filled_qty > 0 and not is_exit_fill:
                try:
                    sl_pct = getattr(event, 'sl_pct', 0.003) or 0.003
                    tp_pct = getattr(event, 'tp_pct', 0.008) or 0.008
                    pos_side = 'LONG' if side.upper() == 'BUY' else 'SHORT'
                    await self._place_protective_orders(symbol_id, side.upper(), filled_qty, fill_price, sl_pct, tp_pct, pos_side)
                except Exception as ex:
                    import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                    logger.warning(f"⚠️ Protective orders failed: {ex}")

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"🚨 [FATAL-EXECUTION] Order {event.direction} for {event.symbol} failed!\nException: {e}\nTraceback:\n{tb}")
            if self.portfolio and side == 'buy':
                # Release pending cash
                amt = event.quantity * (event.price or 0)
                client_id = event.metadata.get('client_order_id') if getattr(event, 'metadata', None) else None
                self.portfolio.release_order_margin(amount=amt, order_id=client_id)
    
    async def _place_protective_orders(self, symbol_id, side, quantity, entry_price, sl_pct, tp_pct, pos_side='LONG'):
        """
        🚀 FORENSIC FIX: VIRTUAL NETTING ENFORCEMENT
        QUÉ: Deshabilitamos el envío de STOP/TP al Exchange Binance.
        POR QUÉ: Binance Hedge mode soporta 1 LONG y 1 SHORT globales. Si Scalping (0.3% SL)
          y Swing (3% SL) abren LONG, Binance mezcla los buckets y el SL del Scalping 
          "machaca" (ejecuta) el saldo del Swing, anulando el multi-horizonte.
        PARA QUÉ: Permitir que Portfolio.py y RiskManager.check_stops() ejecuten 
          cierres milimétricos, aislando lógicamente Scalping y Swing en una cuenta micro.
        """
        logger.info(f"🛡️ [VIRTUAL SL/TP] Delegando Stops al Neural Ledger (Shadow Mode) para evitar cacería en {symbol_id} ({pos_side})")
        
        # 🚀 FIX PARA TEST_HORIZON_ISOLATION: Garantizamos un retorno temprano
        # para que NINGUNA orden protectora (ni el catastrophe stop) sea enviada a Binance,
        # protegiendo el 100% de la arquitectura Multi-Horizonte.
        return
        
        # 👻 MUTACIÓN 29: SHADOW STOPS PURos + CATASTROPHE STOP
        # En lugar de enviar nuestro SL real al Exchange (donde pueden cazarlo), 
        # enviamos un "Catastrophe Stop" a 10% de distancia por si se corta el internet,
        # pero el SL real se queda en memoria RAM.
        catastrophe_sl_pct = 0.10 # 10% de caída máxima tolerada por apagón
        if side.upper() == 'BUY':  # LONG
            stop_side = 'SELL'
            catastrophe_price = entry_price * (1 - catastrophe_sl_pct)
        else:                      # SHORT
            stop_side = 'BUY'
            catastrophe_price = entry_price * (1 + catastrophe_sl_pct)
            
        symbol_ccxt = symbol_id.replace('USDT', '/USDT')
        catastrophe_price_str = self.exchange.price_to_precision(symbol_ccxt, catastrophe_price)
        qty_str = self.exchange.amount_to_precision(symbol_ccxt, quantity)
        
        try:
            stop_params = {
                'symbol': symbol_id,
                'side': stop_side,
                'positionSide': pos_side,
                'type': 'STOP_MARKET',
                'quantity': qty_str,
                'stopPrice': catastrophe_price_str,
                'reduceOnly': 'true',
                'newOrderRespType': 'RESULT'
            }
            stop_order = await self.async_exchange.fapiPrivatePostOrder(stop_params)
            logger.info(f"  🛑 [CATASTROPHE STOP] Red de seguridad colocada en {catastrophe_price_str} (Order ID: {stop_order.get('orderId')})")
        except Exception as e:
            logger.error(f"⚠️ Error colocando Catastrophe Stop: {e}")
            
        stop_price = 0.0; target_price = 0.0  # Linter fix
        return # Salimos para NO enviar el Take Profit ni el Stop Loss reales
        
        # ═══════════════════════════════════════════════════════════════
        # BBO ARCHITECTURE: STOP (Limit) + TAKE_PROFIT (Limit)
        # SL Limit Price: slightly worse than trigger to ensure fill
        # TP Limit Price: at trigger price for pure Maker
        # ═══════════════════════════════════════════════════════════════
        # [FASE 29] SHADOW EXECUTION ENGINE (Multi-Horizon Netting)
        # ═══════════════════════════════════════════════════════════════
        use_native_stops = getattr(Config, 'Execution', None) and getattr(Config.Execution, 'USE_NATIVE_STOPS', False)
        
        if not use_native_stops:
            logger.info(f"  🛡️ [SHADOW EXITS] Native SL/TP Disabled for {symbol_id}. Cython Risk Manager handling Soft-Exits.")
        else:
            use_limit_protective = getattr(Config.Execution, 'USE_LIMIT_PROTECTIVE_ORDERS', True)
            sl_tolerance = getattr(Config.Execution, 'STOP_LIMIT_TOLERANCE_PCT', 0.001) if use_limit_protective else 0
            tp_tolerance = getattr(Config.Execution, 'TP_LIMIT_TOLERANCE_PCT', 0.0) if use_limit_protective else 0
            
            if use_limit_protective:
                # SL Limit: For LONG sells below trigger, for SHORT buys above trigger
                if stop_side == 'SELL':  # Closing LONG: sell limit slightly below trigger
                    stop_limit_price = stop_price * (1 - sl_tolerance)
                else:  # Closing SHORT: buy limit slightly above trigger
                    stop_limit_price = stop_price * (1 + sl_tolerance)
                
                # TP Limit: At exact trigger (will fill AT trigger price = pure Maker)
                if stop_side == 'SELL':  # Closing LONG: sell at/above target
                    tp_limit_price = target_price * (1 + tp_tolerance)
                else:  # Closing SHORT: buy at/below target
                    tp_limit_price = target_price * (1 - tp_tolerance)
            
            # Format prices to exchange precision
            symbol_ccxt = symbol_id.replace('USDT', '/USDT')
            stop_price_str = self.exchange.price_to_precision(symbol_ccxt, stop_price)
            target_price_str = self.exchange.price_to_precision(symbol_ccxt, target_price)
            qty_str = self.exchange.amount_to_precision(symbol_ccxt, quantity)
            
            if use_limit_protective:
                stop_limit_str = self.exchange.price_to_precision(symbol_ccxt, stop_limit_price)
                tp_limit_str = self.exchange.price_to_precision(symbol_ccxt, tp_limit_price)
            
            try:
                # ── STOP LOSS: STOP (Limit) with tolerance ──
                if use_limit_protective:
                    stop_params = {
                        'symbol': symbol_id,
                        'side': stop_side,
                        'positionSide': pos_side,
                        'type': 'STOP',           # ← LIMIT-based (Maker fee)
                        'quantity': qty_str,
                        'stopPrice': stop_price_str,
                        'price': stop_limit_str,   # ← Limit execution price
                        'timeInForce': 'GTC',
                        'reduceOnly': 'true',
                        'newOrderRespType': 'RESULT'
                    }
                    logger.info(f"  💰 [BBO] SL as STOP (Limit): trigger={stop_price_str}, limit={stop_limit_str} (Maker fee)")
                else:
                    stop_params = {
                        'symbol': symbol_id,
                        'side': stop_side,
                        'positionSide': pos_side,
                        'type': 'STOP_MARKET',
                        'quantity': qty_str,
                        'stopPrice': stop_price_str,
                        'reduceOnly': 'true',
                        'newOrderRespType': 'RESULT'
                    }
                
                try:
                    stop_order = await self.async_exchange.fapiPrivatePostOrder(stop_params)
                    logger.info(f"  🛑 Stop-Loss placed at {stop_price_str} (Order ID: {stop_order.get('orderId')})")
                except (ccxt.InvalidOrder, ccxt.ExchangeError) as e:
                    # FALLBACK: If STOP (Limit) is rejected, try STOP_MARKET
                    if use_limit_protective and 'STOP' in stop_params.get('type', ''):
                        logger.warning(f"  ⚠️ [BBO-FALLBACK] STOP Limit rejected: {e}. Falling back to STOP_MARKET.")
                        stop_params['type'] = 'STOP_MARKET'
                        stop_params.pop('price', None)
                        stop_params.pop('timeInForce', None)
                        stop_order = await self.async_exchange.fapiPrivatePostOrder(stop_params)
                        logger.info(f"  🛑 Stop-Loss (MARKET fallback) placed at {stop_price_str}")
                    else:
                        raise
                
                # ── TAKE PROFIT: TAKE_PROFIT (Limit) at trigger ──
                if use_limit_protective:
                    tp_params = {
                        'symbol': symbol_id,
                        'side': stop_side,
                        'positionSide': pos_side,
                        'type': 'TAKE_PROFIT',     # ← LIMIT-based (Maker fee)
                        'quantity': qty_str,
                        'stopPrice': target_price_str,
                        'price': tp_limit_str,      # ← Limit execution price
                        'timeInForce': 'GTC',
                        'reduceOnly': 'true',
                        'newOrderRespType': 'RESULT'
                    }
                    logger.info(f"  💰 [BBO] TP as TAKE_PROFIT (Limit): trigger={target_price_str}, limit={tp_limit_str} (Maker fee)")
                else:
                    tp_params = {
                        'symbol': symbol_id,
                        'side': stop_side,
                        'positionSide': pos_side,
                        'type': 'TAKE_PROFIT_MARKET',
                        'quantity': qty_str,
                        'stopPrice': target_price_str,
                        'reduceOnly': 'true',
                        'newOrderRespType': 'RESULT'
                    }
                
                try:
                    tp_order = await self.async_exchange.fapiPrivatePostOrder(tp_params)
                    logger.info(f"  💰 Take-Profit placed at {target_price_str} (Order ID: {tp_order.get('orderId')})")
                except (ccxt.InvalidOrder, ccxt.ExchangeError) as e:
                    # FALLBACK: If TAKE_PROFIT (Limit) is rejected, try TAKE_PROFIT_MARKET
                    if use_limit_protective and 'TAKE_PROFIT' == tp_params.get('type', ''):
                        logger.warning(f"  ⚠️ [BBO-FALLBACK] TP Limit rejected: {e}. Falling back to TAKE_PROFIT_MARKET.")
                        tp_params['type'] = 'TAKE_PROFIT_MARKET'
                        tp_params.pop('price', None)
                        tp_params.pop('timeInForce', None)
                        tp_order = await self.async_exchange.fapiPrivatePostOrder(tp_params)
                        logger.info(f"  💰 Take-Profit (MARKET fallback) placed at {target_price_str}")
                    else:
                        raise
            
            except ccxt.InsufficientFunds as e:
                logger.warning(f"  ⚠️ Insufficient funds for protective orders: {e}")
            except ccxt.InvalidOrder as e:
                logger.warning(f"  ⚠️ Invalid protective order params: {e}")
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                # Non-critical: bot can still function with Layers 1 & 2
                logger.warning(f"  ⚠️ Protective orders failed: {e}")

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancels an open order on Binance.
        PROFESSOR: Módulo crítico para el 'Anti-Liquidity Sniping'.
        """
        try:
            # Prepare symbol string (e.g., BTCUSDT)
            if not self.exchange.markets:
                self.exchange.load_markets()
            
            market = self.exchange.market(symbol)
            symbol_id = market['id']
            
            if Config.BINANCE_USE_FUTURES:
                # ⚡ FASE 12.5: WS CANCEL
                if getattr(self, 'ws_executor', None) and self.ws_executor.is_ready():
                    logger.info(f"⚡ [WS-EXEC] Cancelling order {order_id} via WS")
                    result = await self.ws_executor.cancel_order(symbol=symbol_id, order_id=order_id)
                else:
                    # FUTURES: fapiPrivateDeleteOrder
                    result = await self.async_exchange.fapiPrivateDeleteOrder({
                        'symbol': symbol_id,
                        'orderId': order_id,
                        'recvWindow': 60000
                    })
                logger.info(f"🗑️ [EXEC] Deleted Futures Order: {order_id} ({symbol})")
            else:
                # SPOT
                result = await self.async_exchange.cancel_order(order_id, symbol)
                logger.info(f"🗑️ [EXEC] Deleted Spot Order: {order_id} ({symbol})")
                
            return True
        except ccxt.OrderNotFound:
            logger.warning(f"⚠️ [EXEC] Order {order_id} not found (already filled or cancelled?)")
            return True # Consider a win
        except Exception as e:
            logger.error(f"❌ [EXEC] Failed to cancel order {order_id}: {e}")
            return False

    def get_all_balances(self):
        """
        Fetches and displays COMPLETE account information from all Binance wallets:
        1. USDT-M Futures (USDT-Ⓜ) - Full Account Info
        2. COIN-M Futures (COIN-Ⓜ) - Full Account Info
        3. Spot (Balance Estimado)
        
        Returns the primary USDT-M total wallet balance for portfolio sync.
        """
        logger.info("=" * 70)
        logger.info("💰 BINANCE ACCOUNT - COMPLETE ANALYSIS")
        logger.info("=" * 70)
        
        primary_balance = None
        total_usdt = 0.0
        total_btc = 0.0
        non_zero_balances = False
        
        # ===================================================================
        # 1. USDT-M Futures - COMPLETE ACCOUNT INFORMATION
        # ===================================================================
        try:
            # SKIP Futures check if we are in SPOT TESTNET mode (Keys are not compatible)
            if Config.BINANCE_USE_TESTNET and not Config.BINANCE_USE_FUTURES:
                raise ccxt.ExchangeError("Skipping Futures check in Spot Testnet (Keys incompatible)")

            # Ensure URLs exist
            if Config.BINANCE_USE_FUTURES and Config.BINANCE_USE_TESTNET:
                if 'fapiPrivateV3' not in self.exchange.urls['api']:
                    v1_url = self.exchange.urls['api'].get('fapiPrivate', 'https://testnet.binancefuture.com/fapi/v1')
                    self.exchange.urls['api']['fapiPrivateV3'] = v1_url.replace('v1', 'v3')
                    if 'test' in self.exchange.urls:
                        self.exchange.urls['test']['fapiPrivateV3'] = self.exchange.urls['api']['fapiPrivateV3']
            
            # GET /fapi/v3/account - COMPLETE ACCOUNT INFORMATION
            account_info = self.exchange.fapiPrivateV3GetAccount()
            
            # Extract critical metrics
            total_wallet = float(account_info.get('totalWalletBalance', 0))
            total_margin = float(account_info.get('totalMarginBalance', 0))
            total_unpnl = float(account_info.get('totalUnrealizedProfit', 0))
            available = float(account_info.get('availableBalance', 0))
            total_init_margin = float(account_info.get('totalInitialMargin', 0))
            total_maint_margin = float(account_info.get('totalMaintMargin', 0))
            total_pos_margin = float(account_info.get('totalPositionInitialMargin', 0))
            total_order_margin = float(account_info.get('totalOpenOrderInitialMargin', 0))
            max_withdraw = float(account_info.get('maxWithdrawAmount', 0))
            
            # Calculate margin ratio (risk indicator)
            margin_ratio = 0
            if total_margin > 0 and total_maint_margin > 0:
                margin_ratio = (total_maint_margin / total_margin) * 100
            
            logger.info("📊 USDT-Ⓜ FUTURES - COMPLETE ACCOUNT STATUS")
            logger.info("-" * 70)
            logger.info(f"  💵 Wallet Balance:       ${total_wallet:>15,.2f}")
            logger.info(f"  📈 Margin Balance:       ${total_margin:>15,.2f}")
            logger.info(f"  💰 Available Balance:    ${available:>15,.2f}")
            logger.info(f"  🎯 Max Withdraw:         ${max_withdraw:>15,.2f}")
            logger.info(f"  {'📉' if total_unpnl < 0 else ' 📈'} Unrealized PnL:       ${total_unpnl:>15,.2f}")
            logger.info(f"\n  ⚠️  MARGIN METRICS:")
            logger.info(f"     Initial Margin:       ${total_init_margin:>15,.2f}")
            logger.info(f"     Maintenance Margin:   ${total_maint_margin:>15,.2f}")
            logger.info(f"     Position Margin:      ${total_pos_margin:>15,.2f}")
            logger.info(f"     Order Margin:         ${total_order_margin:>15,.2f}")
            if total_maint_margin > 0:
                logger.info(f"     🚨 Margin Ratio:      {margin_ratio:>16,.2f}%")
            
            # Show positions if any
            positions = account_info.get('positions', [])
            active_positions = [p for p in positions if float(p.get('positionAmt', 0)) != 0]
            
            if active_positions:
                logger.info(f"\n  📍 OPEN POSITIONS ({len(active_positions)}):")
                for pos in active_positions:
                    symbol = pos['symbol']
                    amt = float(pos['positionAmt'])
                    entry_price = float(pos.get('entryPrice', 0))
                    unpnl = float(pos.get('unrealizedProfit', 0))
                    notional = float(pos.get('notional', 0))
                    leverage = pos.get('leverage', 'N/A')
                    isolated = pos.get('isolated', False)
                    margin_type = "Isolated" if isolated else "Cross"
                    
                    side = "LONG" if amt > 0 else "SHORT"
                    color = "🟢" if unpnl >= 0 else "🔴"
                    
                    logger.info(f"     {color} {symbol:12} {side:5} {abs(amt):>10.4f} @ ${entry_price:<10,.2f}")
                    logger.info(f"        Leverage: {leverage}x | {margin_type} | PnL: ${unpnl:+,.2f} | Notional: ${abs(notional):,.2f}")
            else:
                logger.info(f"\n  📍 No open positions")
            
            primary_balance = total_wallet
            
        except ccxt.AuthenticationError as e:
            logger.error(f"❌ Authentication error fetching balance: {e}")
            logger.error("   Check API key permissions (requires 'Enable Reading')")
            return None
        except ccxt.NetworkError as e:
            logger.warning(f"Could not fetch USDT-M Futures account info: Network error - {e}")
            logger.warning("⚠ USDT-Ⓜ Futures: Network error - retrying...")
        except ccxt.ExchangeError as e:
            logger.warning(f"Could not fetch USDT-M Futures account info: Exchange error - {e}")
            logger.warning("⚠ USDT-Ⓜ Futures: Error fetching data")
            # Fallback to balance-only
            try:
                response = self.exchange.fapiPrivateV2GetBalance()
                for asset in response:
                    if asset['asset'] == 'USDT':
                        primary_balance = float(asset['balance'])
                        logger.info(f"  (Fallback) Balance: ${primary_balance:,.2f}")
                        break
            except Exception as e:
                import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                import logging
                logging.getLogger(__name__).error(f"Silent exception caught: {e}", exc_info=True)
        
        # ===================================================================
        # 2. COIN-M Futures - ACCOUNT INFORMATION
        # ===================================================================
        try:
            # Ensure COIN-M URL exists
            if Config.BINANCE_USE_TESTNET:
                if 'dapiPrivate' not in self.exchange.urls['api']:
                    self.exchange.urls['api']['dapiPrivate'] = 'https://testnet.binancefuture.com/dapi/v1'
                    if 'test' in self.exchange.urls:
                        self.exchange.urls['test']['dapiPrivate'] = self.exchange.urls['api']['dapiPrivate']
            
            # SKIP COIN-M check if we are in SPOT TESTNET mode
            if Config.BINANCE_USE_TESTNET and not Config.BINANCE_USE_FUTURES:
                raise ccxt.ExchangeError("Skipping COIN-M check in Spot Testnet")

            # GET /dapi/v1/balance
            coin_balances = self.exchange.dapiPrivateGetBalance()
            
            has_coin_balance = False
            logger.info(f"📊 COIN-Ⓜ FUTURES")
            logger.info("-" * 70)
            
            for asset_info in coin_balances:
                balance = float(asset_info.get('balance', 0))
                if balance > 0:
                    asset = asset_info['asset']
                    available = float(asset_info.get('availableBalance', 0))
                    cross_unpnl = float(asset_info.get('crossUnPnl', 0))
                    
                    logger.info(f"  💎 {asset:8} Balance: {balance:>12.6f} | Available: {available:>12.6f} | UnPnL: {cross_unpnl:+.6f}")
                    has_coin_balance = True
            
            if not has_coin_balance:
                logger.info(f"  No balances")
                    
        except ccxt.NetworkError as e:
            logger.warning(f"Network error fetching COIN-M Futures info: {e}")
            logger.info(f"📊 COIN-Ⓜ FUTURES")
            logger.info("-" * 70)
            logger.info(f"  Network error fetching data")
        except ccxt.ExchangeError as e:
            logger.warning(f"Exchange error fetching COIN-M Futures info: {e}")
            logger.info(f"📊 COIN-Ⓜ FUTURES")
            logger.info("-" * 70)
            logger.info(f"  Error fetching data")
        
        # ===================================================================
            logger.info("-" * 70)
            
            if total_usdt > 0 or total_btc > 0:
                if total_usdt > 0:
                    logger.info(f"  💵 USDT Total:     ${total_usdt:>15,.2f}")
                if total_btc > 0:
                    logger.info(f"  ₿  BTC Total:      {total_btc:>16.8f}")
           
            # Show top non-zero balances
            if non_zero_balances:
                # Sort by total value (descending)
                sorted_balances = sorted(non_zero_balances, key=lambda x: x['total'], reverse=True)
                
                logger.info(f"\n  💰 Assets ({len(non_zero_balances)} total):")
                for b in sorted_balances[:5]:  # Top 5
                    asset = b['asset']
                    free = b['free']
                    locked = b['locked']
                    total = b['total']
                    logger.info(f"     {asset:8} Free: {free:>12,.4f} | Locked: {locked:>12,.4f} | Total: {total:>12,.4f}")
                
                if len(non_zero_balances) > 5:
                    logger.info(f"     ... and {len(non_zero_balances) - 5} more")
            else:
                logger.info(f"  No balances")
                    
        except Exception as e:
            import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
            # Log the specific error for debugging
            error_msg = str(e)
            if 'testnet.binancefuture.com' in error_msg or '404' in error_msg:
                logger.debug(f"Spot query hit Futures server (expected in Futures-only mode): {e}")
                logger.info(f"📊 SPOT (Balance Estimado)")
                logger.info("-" * 70)
                logger.info(f"  Not available (Futures Testnet - different server)")
            else:
                logger.warning(f"Could not fetch Spot balance: {e}")
                logger.info(f"📊 SPOT (Balance Estimado)")
                logger.info("-" * 70)
                logger.info(f"  Error: {error_msg[:50]}...")
        
        
        logger.info("=" * 70)
        
        return primary_balance
    
    def get_balance(self):
        """
        Legacy method for compatibility.
        Calls get_all_balances and returns primary USDT-M balance.
        """
        return self.get_all_balances()

    async def audit_position_sync(self):
        """
        AUDIT DEPT: Ghost Position Catcher
        Ejecuta cada 5 minutos. Compara las posiciones reales en Binance con el ledger interno del portfolio.
        Si encuentra discrepancias críticas (Posición en Binance que no existe localmente), lanza alerta.
        """
        logger.info("🛡️ [AUDIT] Iniciando tarea de sincronización de Posiciones Fantasma...")
        try:
            while True:
                await asyncio.sleep(300) # Check every 5 minutes
                if not self.portfolio or not Config.BINANCE_USE_FUTURES:
                    continue
                    
                try:
                    positions = await self.async_exchange.fapiPrivateV2GetPositionRisk()
                    for pos in positions:
                        amt = float(pos.get('positionAmt', 0))
                        if abs(amt) > 0:
                            sym = pos.get('symbol', '')
                            # Translate symbol (BTCUSDT -> BTC/USDT)
                            if not '/' in sym and sym.endswith('USDT'):
                                sym = f"{sym[:-4]}/USDT"
                            
                            # Check if we have this tracked
                            local_qty = 0.0
                            if sym in self.portfolio.positions:
                                local_qty = self.portfolio.positions[sym].get('quantity', 0.0)
                                
                            if abs(amt) > 1e-8 and abs(local_qty) < 1e-8:
                                logger.critical(f"🚨 [GHOST POSITION] Binance reporta {amt} {sym} abierto, pero Portfolio local tiene 0. Riesgo de liquidación silenciosa.")
                                # Action: Notify Telegram or Emergency Close
                                from utils.notifier import Notifier
                                Notifier.send_telegram(f"🚨 GHOST POSITION DETECTED: {sym} Amt={amt}. Por favor revisa Binance inmediatamente.", "CRITICAL")
                except Exception as e:
                    logger.error(f"⚠️ [AUDIT] Error verificando posiciones fantasma: {e}")
        except asyncio.CancelledError:
            pass

    async def start_zmq_loop(self):
        """Phase OMNI: Decoupled OS-Level Event Ingestion via ZMQ."""
        if not self.zmq_pull:
            return
            
        # Start Hardened Audits
        asyncio.create_task(self.audit_position_sync())
        self._latency_samples = []
            
        logger.info("🔄 [ZMQ] Starting Executor PULL Loop...")
        from core.events import OrderEvent
        try:
            while True:
                event = await self.zmq_pull.pull()
                if isinstance(event, OrderEvent):
                    await self.execute_order(event)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Executor ZMQ Pull Error: {e}")

    def sync_portfolio_state(self, portfolio):
        """
        Synchronize local portfolio with actual Binance state.
        1. Sync Balance
        2. Sync Open Positions
        """
        logger.info("Syncing Portfolio with Binance State...")
        
        # 1. Sync Balance
        balance = self.get_balance()
        if balance is not None:
            portfolio.current_cash = balance
            portfolio.initial_capital = balance # Reset initial capital to current for session PnL
            logger.info(f"✅ Balance Synced: ${balance:.2f}")
        
        # 2. Sync Positions
        try:
            if Config.BINANCE_USE_FUTURES:
                # Fetch positions from Futures API
                # We use the standard CCXT method which handles endpoints automatically
                try:
                    # Try standard fetch_positions (usually maps to v2/positionRisk)
                    positions = self.exchange.fetch_positions()
                except Exception as e:
                    err_str = str(e)
                    if "testnet/sandbox mode is not supported for futures anymore" in err_str:
                        logger.warning(f"⚠️ Fetch de Posiciones denegado por Demo API (Binance Limits). Asumiendo 0 Posiciones Iniciales.")
                        positions = []
                    else:
                        logger.warning(f"Network error in fetch_positions fallback: {e}. Trying raw v2 endpoint...")
                        # Fallback: Ensure URL exists and try raw
                        if 'fapiPrivateV2' not in self.exchange.urls['api']:
                            self.exchange.urls['api']['fapiPrivateV2'] = self.exchange.urls['api']['fapiPrivate'].replace('v1', 'v2')
                            if 'test' in self.exchange.urls:
                                self.exchange.urls['test']['fapiPrivateV2'] = self.exchange.urls['test']['fapiPrivate'].replace('v1', 'v2')
                        
                        try:
                            positions = self.exchange.fapiPrivateV2GetPositionRisk()
                        except Exception as e2:
                            err_str2 = str(e2)
                            if "testnet/sandbox mode is not supported for futures anymore" in err_str2:
                                logger.warning(f"⚠️ Fetch the Posiciones denegado en V2. Asumiendo 0 Posiciones.")
                                positions = []
                            else:
                                raise e2
                
                synced_count = 0
                for pos in positions:
                    # CCXT fetch_positions returns a unified structure
                    # Raw API returns a different structure. We need to handle both.
                    
                    # Check if it's CCXT structure (has 'info') or Raw
                    is_ccxt_struct = 'info' in pos
                    raw_pos = pos['info'] if is_ccxt_struct else pos
                    
                    symbol = raw_pos['symbol']
                    amt = float(raw_pos['positionAmt'])
                    entry_price = float(raw_pos['entryPrice'])
                    
                    # Only care about non-zero positions
                    if abs(amt) > 0:
                        # Update local portfolio
                        # Convert symbol format if needed (BTCUSDT -> BTC/USDT)
                        internal_symbol = symbol
                        if not '/' in symbol:
                            if symbol.endswith('USDT'):
                                internal_symbol = f"{symbol[:-4]}/USDT"
                            else:
                                # Generic fallback
                                internal_symbol = symbol 
                        
                        # Handle CCXT unified symbol mapping for Futures
                        # CCXT often uses SYMBOL:USDT format
                        if ':' in internal_symbol:
                            base_part = internal_symbol.split(':')[0]
                            internal_symbol = base_part
                        
                        # Accumulate for Hedge Mode (LONG is +, SHORT is -)
                        if internal_symbol not in portfolio.positions:
                            portfolio.positions[internal_symbol] = {
                                'quantity': amt,
                                'avg_price': entry_price,
                                'current_price': entry_price # Will be updated by data feed
                            }
                        else:
                            # If hedge mode has dual positions, compute net average (naive) or keep latest
                            prev_qty = portfolio.positions[internal_symbol]['quantity']
                            portfolio.positions[internal_symbol]['quantity'] = prev_qty + amt
                            # Keep the first entry price for simplicity on cold boot
                        
                        # CRITICAL: Reconstruct Used Margin
                        # Margin = Notional Value / Leverage
                        # We assume Config.BINANCE_LEVERAGE is correct for all pairs
                        position_value = abs(amt * entry_price)
                        margin_required = position_value / Config.BINANCE_LEVERAGE
                        portfolio.used_margin += margin_required
                        
                        synced_count += 1
                        logger.info(f"  -> Found Position: {internal_symbol} {amt} @ ${entry_price:.2f} (Margin: ${margin_required:.2f})")
                
                if synced_count > 0:
                    logger.info(f"Synced {synced_count} open positions. Total Used Margin: ${portfolio.used_margin:.2f}")
                else:
                    logger.info("No open positions found on Binance.")
                    
            else:
                # SPOT Position Sync
                # In Spot, we don't have "positions" like Futures (LONG/SHORT contracts).
                # Instead, we have "balances" of assets (BTC, ETH, etc.)
                # We need to fetch the Spot wallet and convert non-zero balances to "positions"
                
                logger.info("Syncing Spot Balances...")
                try:
                    # Fetch Spot balances
                    # Use spot_exchange if in Testnet, otherwise use main exchange
                    try:
                        if hasattr(self, 'spot_exchange') and self.spot_exchange:
                            balance_data = self.spot_exchange.fetch_balance()
                        else:
                            balance_data = self.exchange.fetch_balance()
                    except Exception as e:
                        err_str = str(e)
                        if "testnet/sandbox mode is not supported for futures anymore" in err_str:
                            logger.warning(f"⚠️ Fetch de Spot Balances denegado en Demo Trading. Asignando 0 por defecto.")
                            balance_data = {}
                        else:
                            raise e
                    
                    synced_count = 0
                    
                    # Iterate through all assets with non-zero balances
                    for asset, details in balance_data.items():
                        if asset == 'info' or asset == 'free' or asset == 'used' or asset == 'total':
                            continue  # Skip metadata
                        
                        # Get the 'total' balance (free + locked)
                        if isinstance(details, dict) and 'total' in details:
                            total_balance = details['total']
                            
                            # Only process non-zero balances
                            # BUG #64 FIX: Show USDT balance too!
                            if total_balance and total_balance > 0:
                                # Convert to internal symbol format (BTC -> BTC/USDT)
                                internal_symbol = f"{asset}/USDT"
                                
                                # Fetch current price for this asset
                                try:
                                    ticker = self.spot_exchange.fetch_ticker(internal_symbol) if hasattr(self, 'spot_exchange') else self.exchange.fetch_ticker(internal_symbol)
                                    current_price = ticker.get('last')
                                    if current_price is None:
                                        current_price = 0.0
                                except Exception:
                                    import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                                    current_price = 0.0  # If we can't get price, set to 0
                                
                                # Add to portfolio as a "position" (quantity of asset held)
                                portfolio.positions[internal_symbol] = {
                                    'quantity': total_balance,  # Amount of BTC/ETH/etc. we own
                                    'avg_price': current_price,  # We don't know entry price in Spot, use current
                                    'current_price': current_price
                                }
                                
                                synced_count += 1
                                logger.info(f"  -> Found Spot Balance: {internal_symbol} {total_balance} @ ${current_price:.2f}")
                    
                    if synced_count > 0:
                        logger.info(f"Synced {synced_count} Spot balances.")
                    else:
                        logger.info("No Spot balances found (only USDT).")
                        
                except Exception as e:
                    logger.error(f"Failed to sync Spot balances: {e}")
                
        except ccxt.AuthenticationError as e:
            logger.error(f"❌ Authentication error syncing positions: {e}")
            logger.error("   Verify API key has required permissions")
        except ccxt.NetworkError as e:
            logger.error(f"Network error syncing positions: {e}")
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error syncing positions: {e}")
        except Exception as e:
            logger.error(f"Unexpected error syncing positions: {e}")

    def get_multi_wallet_overview(self):
        """
        Fetches and displays balances from all 3 wallets:
        - Spot
        - USD-M Futures
        - COIN-M Futures (Delivery)
        
        Production Ready: Works seamlessly in both Testnet and Live modes.
        """
        logger.info("="*70)
        logger.info("📊 BINANCE ACCOUNT OVERVIEW - ALL WALLETS")
        if Config.BINANCE_USE_TESTNET:
            logger.info("⚠️  TESTNET MODE (Note: Spot and Futures wallets are SEPARATE in Testnet)")
        else:
            logger.info("🚀 PRODUCTION MODE (Unified Account)")
        logger.info("="*70)
        
        total_value_usd = 0
        
        # ===================================================================
        # 1. SPOT WALLET
        # ===================================================================
        try:
            logger.info("💰 SPOT WALLET")
            logger.info("-" * 70)
            
            # Create a temporary Spot exchange if needed
            if Config.BINANCE_USE_FUTURES and not hasattr(self, 'spot_exchange'):
                # We're in Futures mode but need Spot data
                spot_exchange = ccxt.binance({
                    'apiKey': self.exchange.apiKey,
                    'secret': self.exchange.secret,
                    'options': {'defaultType': 'spot'},
                    'enableRateLimit': True
                })
                if Config.BINANCE_USE_TESTNET:
                    spot_exchange.set_sandbox_mode(True)
            else:
                spot_exchange = self.spot_exchange if hasattr(self, 'spot_exchange') else self.exchange
            
            spot_balance = spot_exchange.fetch_balance()
            
            # Filter non-zero balances
            spot_assets = []
            for asset, details in spot_balance.items():
                if asset not in ['info', 'free', 'used', 'total', 'datetime', 'timestamp']:
                    if isinstance(details, dict) and details.get('total', 0) > 0:
                        spot_assets.append((asset, details['total']))
            
            if spot_assets:
                # Get prices for valuation
                for asset, amount in sorted(spot_assets, key=lambda x: x[0]):
                    try:
                        if asset == 'USDT':
                            price = 1.0
                            value = amount
                        elif asset in ['BUSDT', 'USDC']: # Stablecoins
                            price = 1.0
                            value = amount
                        else:
                            # Try to get price from ticker
                            try:
                                ticker = spot_exchange.fetch_ticker(f"{asset}/USDT")
                                price = ticker['last']
                            except Exception:
                                import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                                price = 0
                            value = amount * price
                        
                        if price > 0:
                            total_value_usd += value
                            logger.info(f"  {asset:8s}: {amount:>15,.4f}  @ ${price:>10,.2f}  = ${value:>12,.2f}")
                        else:
                            logger.info(f"  {asset:8s}: {amount:>15,.4f}  (Price Unavailable)")
                    except Exception:
                        import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                        logger.info(f"  {asset:8s}: {amount:>15,.4f}  (Error valuing)")
            else:
                logger.info("  No assets found")
                
        except Exception as e:
            logger.error(f"  ❌ Error fetching Spot wallet: {e}")
            if Config.BINANCE_USE_TESTNET:
                logger.info("  ℹ️  This is expected in Testnet mode if using Futures keys (Spot and Futures are separate)")
        
        # ===================================================================
        # 2. USD-M FUTURES WALLET
        # ===================================================================
        # ===================================================================
        # 2. USD-M FUTURES WALLET
        # ===================================================================
        try:
            logger.info("⚡ USD-M FUTURES WALLET")
            logger.info("-" * 70)
            
            # Create temporary Futures exchange if needed
            if not Config.BINANCE_USE_FUTURES:
                futures_exchange = ccxt.binance({
                    'apiKey': self.exchange.apiKey,
                    'secret': self.exchange.secret,
                    'options': {'defaultType': 'future'},
                    'enableRateLimit': True
                })
                if Config.BINANCE_USE_TESTNET:
                    futures_exchange.set_sandbox_mode(True)
            else:
                futures_exchange = self.exchange
            
            futures_balance = futures_exchange.fetch_balance()
            
            # USD-M Futures uses USDT as collateral
            usdt_total = futures_balance.get('USDT', {}).get('total', 0)
            usdt_free = futures_balance.get('USDT', {}).get('free', 0)
            usdt_used = futures_balance.get('USDT', {}).get('used', 0)
            
            if usdt_total > 0:
                logger.info(f"  USDT (Collateral):")
                logger.info(f"    Total:     ${usdt_total:>15,.2f}")
                logger.info(f"    Available: ${usdt_free:>15,.2f}")
                logger.info(f"    In Use:    ${usdt_used:>15,.2f}")
                total_value_usd += usdt_total
            else:
                logger.info("  No USDT balance")
                
        except Exception as e:
            logger.error(f"  ❌ Error fetching USD-M Futures wallet: {e}")
            if Config.BINANCE_USE_TESTNET:
                logger.info("  ℹ️  This is expected in Testnet mode if using Spot keys")
        
        # ===================================================================
        # 3. COIN-M FUTURES (DELIVERY) WALLET
        # ===================================================================
        # ===================================================================
        # 3. COIN-M FUTURES (DELIVERY) WALLET
        # ===================================================================
        try:
            logger.info("🪙  COIN-M FUTURES WALLET (Delivery)")
            logger.info("-" * 70)
            
            # COIN-M uses the DAPI endpoint
            # Need to create a separate exchange instance for delivery futures
            try:
                # Create COIN-M exchange instance
                delivery_exchange = ccxt.binance({
                    'apiKey': self.exchange.apiKey,
                    'secret': self.exchange.secret,
                    'options': {
                        'defaultType': 'delivery',  # COIN-M futures
                        'adjustForTimeDifference': True,
                    },
                    'enableRateLimit': True
                })
                
                # Set URLs for delivery (COIN-M)
                if Config.BINANCE_USE_TESTNET:
                    delivery_exchange.set_sandbox_mode(True)
                    # Override with correct DAPI URLs
                    delivery_testnet_base = 'https://testnet.binancefuture.com'
                    delivery_exchange.urls['api']['dapiPublic'] = f'{delivery_testnet_base}/dapi/v1'
                    delivery_exchange.urls['api']['dapiPrivate'] = f'{delivery_testnet_base}/dapi/v1'
                
                # Fetch COIN-M balance
                delivery_balance = delivery_exchange.fetch_balance()
                
                # COIN-M uses crypto as collateral (BTC, ETH, etc.)
                coin_assets = []
                for asset, details in delivery_balance.items():
                    if asset not in ['info', 'free', 'used', 'total', 'datetime', 'timestamp']:
                        if isinstance(details, dict) and details.get('total', 0) > 0:
                            coin_assets.append((asset, details))
                
                if coin_assets:
                    for asset, details in sorted(coin_assets, key=lambda x: x[0]):
                        total = details.get('total', 0)
                        free = details.get('free', 0)
                        used = details.get('used', 0)
                        
                        try:
                            # Get price in USDT for valuation
                            ticker = self.exchange.fetch_ticker(f"{asset}/USDT")
                            price = ticker['last']
                            value = total * price
                            
                            logger.info(f"  {asset:8s} (Collateral):")
                            logger.info(f"    Total:     {total:>15,.6f}  @ ${price:>10,.2f}  = ${value:>12,.2f}")
                            logger.info(f"    Available: {free:>15,.6f}")
                            logger.info(f"    In Use:    {used:>15,.6f}")
                            
                            total_value_usd += value
                        except Exception:
                            import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                            logger.info(f"  {asset:8s}: {total:>15,.6f}  (price unavailable)")
                else:
                    logger.info("  No assets found")
                    
            except Exception as delivery_error:
                import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                logger.warning(f"  ⚠️  Could not fetch COIN-M wallet: {delivery_error}")
                if Config.BINANCE_USE_TESTNET:
                    logger.info("  ℹ️  This is expected in Testnet mode (limited COIN-M support)")
                
        except Exception as e:
            logger.error(f"  ❌ Error fetching COIN-M wallet: {e}")

        
        # ===================================================================
        # SUMMARY
        # ===================================================================
        logger.info("="*70)
        logger.info(f"💵 TOTAL ESTIMATED VALUE: ${total_value_usd:>,.2f} USD")
        logger.info(f"🎯 CURRENT MODE: {'FUTURES' if Config.BINANCE_USE_FUTURES else 'SPOT'}")
        if Config.BINANCE_USE_TESTNET:
            logger.info("⚠️  TESTNET MODE - Using demo funds")
        else:
            logger.info("🔴 LIVE MODE - Using real funds")
        logger.info("="*70)

    async def _chase_order_loop(self, symbol_ccxt: str, symbol_id: str, side: str, qty_precision: float, params: dict, original_order: dict, event, max_chases: int = 5):
        """
        🚀 PHASE 2: HFT CHASE LOGIC for BBO Limits
        Monitors an unfilled GTX order, cancels it if the spread moves away, and replaces it.
        """
        order_id = original_order.get('id', original_order.get('orderId'))
        if not order_id: return
        
        # BBO-ULTRA: Multi-speed Chasing based on ML Confidence
        ml_conf = getattr(event, 'ml_confidence', 0.5) or 0.5
        chase_interval = 2.0 if ml_conf < 0.8 else 0.5 # 4x faster for high conviction
        max_chases = max_chases if ml_conf < 0.9 else max_chases + 2 # Allow more room for certain trades
        
        chases = 0
        current_order_id = order_id
        
        # Safe fallback for Backtest/Offline Demo
        if getattr(Config, 'BINANCE_USE_DEMO', False): return
        
        pass
        
        while chases < max_chases:
            await asyncio.sleep(chase_interval)
            try:
                # Fetch order status to see if it filled
                if Config.BINANCE_USE_FUTURES:
                    try:
                        status_raw = await asyncio.wait_for(self.async_exchange.fapiPrivateGetOrder({'symbol': symbol_id, 'orderId': current_order_id}), timeout=5.0)
                        status_txt = status_raw.get('status', 'FILLED')
                        if status_txt in ['FILLED', 'CANCELED', 'REJECTED']:
                            logger.info(f"☑️ [CHASE] Order {current_order_id} concluded with status: {status_txt}")
                            return
                    except Exception as e:
                        import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                        logger.warning(f"⚠️ [CHASE] Error fetching futures status {current_order_id}: {e}")
                        pass
                else:
                    try:
                        status_ccxt = await asyncio.wait_for(self.async_exchange.fetch_order(current_order_id, symbol_ccxt), timeout=5.0)
                        if status_ccxt['status'] in ['closed', 'canceled', 'rejected']:
                            logger.info(f"☑️ [CHASE] Order {current_order_id} concluded with status: {status_ccxt['status']}")
                            return
                    except Exception as e:
                        import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                        logger.warning(f"⚠️ [CHASE] Error fetching spot status {current_order_id}: {e}")
                        pass
                
                # Order still OPEN. Moving spread.
                logger.warning(f"🏃 [CHASE] {symbol_id} Limit unfilled. Cancelling {current_order_id} to chase spread...")
                
                # Cancel old order
                if Config.BINANCE_USE_FUTURES:
                    await asyncio.wait_for(self.async_exchange.fapiPrivateDeleteOrder({'symbol': symbol_id, 'orderId': current_order_id}), timeout=5.0)
                else:
                    await asyncio.wait_for(self.async_exchange.cancel_order(current_order_id, symbol_ccxt), timeout=5.0)
                
                # Fetch new BBO to position at absolute top of the book
                px_tup = await asyncio.to_thread(self.guardian.get_fast_bid_ask, event.symbol)
                new_price = px_tup[0] if side.lower() == 'buy' else px_tup[1]
                
                # Format precision
                from core.state_manager import BinanceLoader
                precision_data = BinanceLoader.get_precision_data(event.symbol)
                tick_size = precision_data.get('price_tick_size', 0.01) if precision_data else 0.01
                from utils.math_helpers import round_step
                price_precision = round_step(new_price, tick_size)
                
                # Re-submit
                chases += 1
                params['price'] = price_precision
                
                if Config.BINANCE_USE_FUTURES:
                    new_order = await asyncio.wait_for(self.async_exchange.fapiPrivatePostOrder(params), timeout=9.0)
                    current_order_id = new_order.get('orderId')
                else:
                    spot_params = params.copy()
                    for k in ['symbol', 'side', 'type', 'quantity', 'price']:
                        spot_params.pop(k, None)
                    new_order = await asyncio.wait_for(self.async_exchange.create_order(
                        symbol=symbol_ccxt,
                        type='limit',
                        side=side.lower(),
                        amount=qty_precision,
                        price=price_precision,
                        params=spot_params
                    ), timeout=9.0)
                    current_order_id = new_order.get('id')
                    
                logger.info(f"🎯 [CHASE {chases}/{max_chases}] Relocated Limit BBO -> {price_precision} (ID: {current_order_id})")
                
            except Exception as e:
                logger.error(f"🛑 [CHASE ERROR] Failed chase cycle {chases}: {e}")
                break
                
        # Timeout reached
        if chases >= max_chases:
            logger.critical(f"⚠️ [CHASE FAILED] Max chases ({max_chases}) exceeded for {symbol_id}. Forcing MARKET panic fill.")
            try:
                # Cancel current
                if Config.BINANCE_USE_FUTURES:
                    await asyncio.wait_for(self.async_exchange.fapiPrivateDeleteOrder({'symbol': symbol_id, 'orderId': current_order_id}), timeout=5.0)
                else:
                    await asyncio.wait_for(self.async_exchange.cancel_order(current_order_id, symbol_ccxt), timeout=5.0)
                
                # Force Market
                mkt_params = {
                    'symbol': symbol_id,
                    'side': side.upper(),
                    'type': 'MARKET',
                    'quantity': qty_precision,
                }
                if Config.BINANCE_USE_FUTURES:
                    mkt_params['positionSide'] = params.get('positionSide', 'LONG')
                    mkt_params['reduceOnly'] = params.get('reduceOnly', 'false')
                    await asyncio.wait_for(self.async_exchange.fapiPrivatePostOrder(mkt_params), timeout=7.0)
                else:
                    await asyncio.wait_for(self.async_exchange.create_order(
                        symbol=symbol_ccxt, type='market', side=side.lower(), amount=qty_precision
                    ), timeout=7.0)
                logger.warning(f"✅ [MARKET PANIC] Filled {side.upper()} {qty_precision} {symbol_id} via Market.")
                
            except Exception as e:
                logger.error(f"💥 [FATAL CHASE] Failed to force Market Fill after chasing timeout: {e}")

    async def transfer_to_spot(self, asset: str, amount: float) -> bool:
        """
        FASE 5: Auto-Secuestro de Capital.
        Transfiere fondos desde USD-M Futures hacia Spot para aislar el capital (Risk-Free Mode).
        """
        if not Config.BINANCE_USE_FUTURES:
            logger.warning("⚠️ No se puede transferir a Spot porque no estamos en modo Futures.")
            return False
            
        try:
            logger.critical(f"🏦 [AUTO-SECUESTRO] Iniciando transferencia de {amount} {asset} de Futures a Spot...")
            # type 2: UMfuture_main (Futures to Spot)
            params = {
                'asset': asset,
                'amount': self.exchange.currency_to_precision(asset, amount) if hasattr(self.exchange, 'currency_to_precision') else round(amount, 4),
                'type': 2 
            }
            res = await self.async_exchange.sapiPostFuturesTransfer(params)
            tran_id = res.get('tranId')
            logger.critical(f"🛡️ [RISK-FREE MODE] ¡Éxito! Transacción ID: {tran_id}. Semilla inicial de ${amount} {asset} asegurada en Spot.")
            return True
        except Exception as e:
            logger.error(f"❌ [AUTO-SECUESTRO] Falló la transferencia: {e}")
            return False

    async def _adverse_selection_monitor(self, symbol_ccxt: str, symbol_id: str, side: str, order_id: str, original_price: float, event):
        """
        🌊 FASE 12: ADVERSE SELECTION VETO & SYNTHETIC QUEUE ESTIMATOR
        Monitorea el Order Book Imbalance (OBI) y la posición en la cola de una orden LIMIT.
        Si la liquidez se vuelve tóxica, CANCELA la orden para evitar ser atropellado por un muro institucional.
        """
        if getattr(Config, 'BINANCE_USE_DEMO', False): return
        
        # Permitimos a la orden descansar brevemente en el LOB
        await asyncio.sleep(0.1)
        
        # Evaluamos por un máximo de 10 segundos
        for _ in range(50):
            await asyncio.sleep(0.2)
            
            imbalance = 0.0
            micro = {}
            if self.data_provider and hasattr(self.data_provider, 'microstructure'):
                internal_sym = symbol_ccxt.replace('/', '')
                if internal_sym in self.data_provider.microstructure:
                    micro = self.data_provider.microstructure[internal_sym].get_metrics()
                    imbalance = micro.get('obi', 0.0)
            elif self.data_provider and hasattr(self.data_provider, 'lob_imbalance'):
                internal_sym = symbol_ccxt.replace('/', '')
                if internal_sym in self.data_provider.lob_imbalance:
                    imbalance = self.data_provider.lob_imbalance[internal_sym].get('imbalance', 0.0)
            
            # Criterios Tóxicos (Adverse Selection)
            # LONG Limit: Si OBI se vuelve muy negativo, viene una cascada de ventas
            # SHORT Limit: Si OBI se vuelve muy positivo, viene una estampida de compras
            is_toxic = False
            toxic_reason = ""
            
            # 🌊 FASE 13: Iceberg, VPIN & Spoofing Check
            if side.upper() == 'BUY':
                if imbalance < -0.75:
                    is_toxic, toxic_reason = True, "Cascada de Ventas (OBI < -0.75)"
                elif micro.get('iceberg_score', 0.0) > 0.8 and micro.get('dark_pool_side') == 'SELL':
                    is_toxic, toxic_reason = True, "Iceberg de Venta Activo (Score > 0.8)"
                elif micro.get('vpin', 0.0) > 0.7:
                    is_toxic, toxic_reason = True, "Flujo Tóxico (VPIN > 0.7)"
                elif micro.get('is_spoofing') and micro.get('spoofing_side') == 'BUY':
                    is_toxic, toxic_reason = True, "Mirage: Falsa Pared de Compra (Spoofing)"
            else:
                if imbalance > 0.75:
                    is_toxic, toxic_reason = True, "Estampida de Compras (OBI > 0.75)"
                elif micro.get('iceberg_score', 0.0) > 0.8 and micro.get('dark_pool_side') == 'BUY':
                    is_toxic, toxic_reason = True, "Iceberg de Compra Activo (Score > 0.8)"
                elif micro.get('vpin', 0.0) > 0.7:
                    is_toxic, toxic_reason = True, "Flujo Tóxico (VPIN > 0.7)"
                elif micro.get('is_spoofing') and micro.get('spoofing_side') == 'SELL':
                    is_toxic, toxic_reason = True, "Mirage: Falsa Pared de Venta (Spoofing)"
                
            if is_toxic:
                logger.critical(f"🌊🛑 [ADVERSE SELECTION VETO] {symbol_id} {side} LIMIT Order {order_id} bloqueada en cola. Mercado tóxico: {toxic_reason}. Ejecutando CANCEL de emergencia!")
                try:
                    if Config.BINANCE_USE_FUTURES:
                        await self.async_exchange.fapiPrivateDeleteOrder({'symbol': symbol_id, 'orderId': order_id})
                    else:
                        await self.async_exchange.cancel_order(order_id, symbol_ccxt)
                        
                    # Emitir evento de fallo de ejecución para limpiar el Portfolio margin
                    if self.events_queue:
                        from core.events import ExecutionFailedEvent
                        asyncio.create_task(self.zmq_push.push(ExecutionFailedEvent(
                            symbol=event.symbol, quantity=event.quantity, price=original_price, direction=event.direction,
                            reason=f"ADVERSE_SELECTION_{toxic_reason}", strategy_id=getattr(event, 'strategy_id', None), trade_id=getattr(event, 'trade_id', None)
                        )))
                except Exception as e:
                    import logging; logging.getLogger(__name__).error('Silent exception caught', exc_info=True)
                    if 'Unknown order' not in str(e):
                        logger.warning(f"⚠️ [ADVERSE SELECTION] No se pudo cancelar orden {order_id}: {e}")
                return # Detener monitoreo

    async def stop(self):
        """Graceful shutdown"""
        pass
        if hasattr(self, 'stream_task'):
            self.stream_task.cancel()
        if hasattr(self, 'async_exchange'):
            await self.async_exchange.close()
            logger.info("  ⚡ Closed async CCXT connection")
        logger.info("✅ [Executor] Stopped.")
