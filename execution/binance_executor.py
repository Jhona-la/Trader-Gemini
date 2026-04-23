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
from .user_data_stream import UserDataStream  # [Dept 3 Fix]
import time
import asyncio
import numpy as np
import ccxt.async_support as ccxt_async

class BinanceExecutor:
    """
    Handles execution of orders on Binance via CCXT.
    Supports both Spot and Testnet.
    Integrated with UserDataStream for real-time fills.
    """
    def __init__(self, events_queue, portfolio=None, data_provider=None, micro_awareness=None):
        self.events_queue = events_queue
        self.portfolio = portfolio  # Reference for cash release on failure
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

        # ===================================================================
        # Async Exchange Initialization
        # ===================================================================
        self.async_exchange = ccxt_async.binance({
            'apiKey': api_key,
            'secret': secret_key,
            'enableRateLimit': True,
            'adjustForTimeDifference': True,
            'timeout': 10000, 
            'options': options
        })
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
                 logger.warning(f"⚠️ [HEDGE MODE] Could not enforce Hedge Mode (Already set or API error): {e}")

        # CRITICAL: Monkey patch 'request' method to intercept ALL sapi calls AND track Rate Limit
        # This is more robust than patching individual methods
        original_request = self.exchange.request
        
        def intercepted_request(path, api='public', method='GET', params={}, headers=None, body=None, config={}):
            # 1. TESTNET SAPI BLOCKER (BUG #17)
            if api == 'sapi' and ((hasattr(Config, 'BINANCE_USE_DEMO') and Config.BINANCE_USE_DEMO) or Config.BINANCE_USE_TESTNET):
                return []
            
            # 2. PHASE 14: PREDICTIVE RATE LIMIT CHECK
            # Estimated weight: order=1, others=1. Heavy endpoints handled by buffer.
            is_safe, wait_time = self.rate_limiter.check_limit(weight_cost=1)
            if not is_safe:
                # BLOCKING WAIT (Safety first)
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
            def intercepted_spot_request(path, api='public', method='GET', params={}, headers=None, body=None, config={}):
                if api == 'sapi':
                    return []
                    
                is_safe, wait_time = self.rate_limiter.check_limit(weight_cost=1)
                if not is_safe:
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
            
            if drift > 1500: # [SOVEREIGN-DEPLOY] Local Network Adaptation (max 1500ms for Dev Dry-Run)
                 logger.critical(f"🛑 [TIME-DRIFT] Atomic Clock Sync Failed! Drift is {drift}ms (Max: 1500ms). Aborting.")
                 # En windows el NTP por defecto a veces falla, forzamos salida
                 raise RuntimeError(f"Time drift too high: {drift}ms > 1500ms limit")
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
            logger.warning(f"Could not tune CCXT keep-alive: {e}")

        self._initialize_futures_settings()
        
        # [Dept 3 Fix] Start User Data Stream (Moved to main.py loop)
        self.user_stream = UserDataStream(self.events_queue, self.exchange)
        logger.info("✅ [Executor] User Data Stream Ready (Will start in async loop)")

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
            metrics.increment("leverage_adjustments")
        except Exception as e:
            err_msg = str(e)
            if "No need to change" in err_msg:
                self._leverage_cache[symbol_id] = target_leverage
            else:
                logger.warning(f"⚠️ [LEVERAGE-FAIL] Could not set leverage to {target_leverage}x for {symbol_id}: {e}")

    # Removed @trace_execution to prevent issues with async
    async def execute_order(self, event):
        """
        🚀 SUPREMO-V3: ULTRA-LOW LATENCY EXECUTION
        QUÉ: Envía órdenes al exchange con precisión quirúrgica y mínima latencia.
        """
        if event.type != 'ORDER': return
        
        start_exec = time.perf_counter()
        
        # 🧬 [Phase 19] SHADOW MODE INTERCEPTION
        # If this is a Shadow Order, we DO NOT send it to Binance.
        # We just log it as "Virtual Fill" and return.
        if getattr(event, 'is_shadow', False):
            logger.info(f"👻 [SHADOW] VIRTUAL EXECUTION: {event.direction.value} {event.quantity} {event.symbol} @ {event.price or 'MKT'}")
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
            if not self.exchange.markets:
                self.exchange.load_markets()
            
            market = self.exchange.market(symbol_ccxt)
            symbol_id = market['id']
            
            # --- 🛡️ PHASE 15: PROGRAMMATIC LEVERAGE CHECK ---
            # Ensures the exchange is set to the correct leverage FOR THIS SPECIFIC HORIZON
            target_leverage = getattr(event, 'leverage', None)
            if target_leverage and Config.BINANCE_USE_FUTURES:
                await self._ensure_leverage(symbol_id, target_leverage)
            
            # 🛡️ PHASE II: ANTI-SLIPPAGE (Order Book Depth Check)
            # If MARKET order and liquidity is thin, downgrade to LIMIT or abort.
            if order_type == 'market':
                try:
                    bid, ask = await asyncio.to_thread(self.guardian.get_fast_bid_ask, symbol_ccxt)
                    
                    if bid > 0 and ask > 0:
                        spread_pct = (ask - bid) / bid
                        # Si el spread es > 0.1% (Muy alto para HFT), forzar LIMIT
                        if spread_pct > 0.001:
                            logger.warning(f"⚠️ High Spread ({spread_pct*100:.3f}%) detected for {symbol}. Downgrading to LIMIT.")
                            order_type = 'limit'
                            # Post at Best Bid/Ask
                            price = bid if side == 'sell' else ask
                except Exception as e:
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
                     bid, ask = await asyncio.to_thread(self.guardian.get_fast_bid_ask, symbol_ccxt)
                     price = bid if side == 'buy' else ask
            
            if side == 'buy': # Only check for BUYS (Entry/Cover)
                try:
                    quote_currency = market['quote']
                    last_px = await asyncio.to_thread(self.guardian.get_fast_bid_ask, symbol_ccxt)
                    last_price = last_px[0] if last_px[0] > 0 else 1.0 # fallback bid
                    cost_est = event.quantity * (price if price else last_price)
                    
                    # Fetch Balance (Optimized: fetch only what's needed if supported, else full)
                    # CCXT fetch_balance is cached by default in some modes, but we want FRESH.
                    # params={'type': 'future'} if futures
                    params = {}
                    if Config.BINANCE_USE_FUTURES: params['type'] = 'future'
                    
                    # We use a specialized light check or full fetch
                    # Note: frequent fetching hits rate limits. We rely on 'User Data Stream' mostly,
                    # but for "Atomic" check we might double check if we are close to edge.
                    # Optimization: Only check if local estimate is within 10% of total equity?
                    # No, user wants invalidation.
                    
                    # Rate Limit Protection: Only fetch if > 1s since last fetch?
                    # Start with standard fetch.
                    balance = await self.async_exchange.fetch_free_balance(params=params)
                    available = balance.get(quote_currency, 0.0)
                    
                    if available < cost_est:
                         logger.error(f"🚫 [ATOMIC] INSUFFICIENT FUNDS! Need: {cost_est:.2f} {quote_currency}, Avail: {available:.2f}")
                         if self.portfolio: self.portfolio.release_cash(cost_est)
                         return
                         
                except Exception as e:
                    logger.warning(f"⚠️ Balance Check Skipped: {e}")


            # 2. EXIT PRIORITY (Rule 2.1) - Skip Guardian if EXIT
            is_exit = getattr(event, 'is_exit', False) or (getattr(event, 'strategy_id', '') == 'EMERGENCY_EXIT')
            
            if not is_exit:
                # Normal Signal: Run Guardian check
                
                # 🏎️ [L-001] Guardian Bypass Logic: Speed up Scalping Maker limits
                is_post_only = metadata.get('timeInForce') == 'GTX'
                is_scalp_limit = (order_type == 'limit' and getattr(event, 'horizon', '') == 'SCALPING')
                should_bypass = is_post_only or is_scalp_limit
                
                liquidity = await asyncio.to_thread(self.guardian.analyze_liquidity, symbol, event.quantity, event.direction.name, should_bypass)
                if not liquidity['is_safe']:
                    logger.warning(f"🛡️ [GUARDIAN] Order Blocked: {liquidity['reason']}")
                    if self.portfolio: self.portfolio.release_cash(event.quantity * price if price else 0)
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
                    pass # Non-critical
                
                # --- PHASE 14: SMART-ORDER ROUTING (SOR) ---
                # Decide order type based on urgency and rebate priority
                is_urgent = getattr(event, 'urgent', False)
                rebate_priority = getattr(self.portfolio, 'rebate_priority', True)
                
                # FORENSIC FIX: Force Maker-Only for Scalping Entries
                is_scalping_entry = (getattr(event, 'horizon', '') == 'SCALPING') and not getattr(event, 'is_exit', False) and not getattr(event, 'is_close', False)
                
                if is_scalping_entry:
                    logger.info(f"🛡️ [FORENSIC-SOR] SCALPING Entry Detected for {symbol}: Forcing LIMIT (GTX/Post-Only) to stop Fee Bleed.")
                    order_type = 'limit'
                    metadata['timeInForce'] = 'GTX'
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
                # EXIT: High Priority - Use current market price directly
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
                            smart_price = bid # Top of book bid
                        else:
                            smart_price = ask # Bottom of book ask
                    except Exception as e:
                        logger.warning(f"⚠️ Could not fetch orderbook for Maker pricing: {e}")

                if side == 'buy': smart_price *= (1 + spread_adj)
                else: smart_price *= (1 - spread_adj)
            
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
                    ticker = self.exchange.fetch_ticker(symbol_ccxt)
                    reference_price = float(ticker.get('last', 0))
            except Exception as e:
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
                    if self.portfolio and side == 'buy':
                        self.portfolio.release_cash(event.quantity * (event.price or 0))
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
                    import asyncio
                    # ccxt soporta cancel_all_orders para casi todos
                    await asyncio.wait_for(self.async_exchange.cancel_all_orders(symbol_ccxt), timeout=4.0)
                    logger.info(f"✅ [PREDICTIVE LIMIT] Pending TP orders cancelled for {symbol_ccxt}.")
                except Exception as e:
                    logger.warning(f"⚠️ Could not cancel previous TP Limit: {e}")

            # 4. SEND ORDER
            logger.info(f"⚡ [EXEC] {order_type.upper()} {side.upper()} {symbol} | Qty: {final_qty} | P: {final_price}")
            
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
                
                import asyncio
                try:
                    order_raw = await asyncio.wait_for(self.async_exchange.fapiPrivatePostOrder(params), timeout=9.0)
                except asyncio.TimeoutError:
                    logger.critical(f"🛑 [TIMEOUT] Futures order {side} {symbol_id} hung >9s! OS Network Blocked.")
                    raise RuntimeError("API Timeout / Disconnect in execution")
                order = order_raw # Simplified mapping
            else:
                # SPOT
                import asyncio
                import ccxt
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
            end_exec = time.perf_counter()
            exec_latency = (end_exec - start_exec) * 1000
            latency_monitor.track('order_to_send', exec_latency)
            
            # [SOVEREIGN-DEPLOY] Latency Guard (< 200ms threshold)
            if exec_latency > 200.0:
                self.latency_violations += 1
                logger.warning(f"⚠️ [LATENCY] Order execution took {exec_latency:.2f}ms (>200ms). Violation {self.latency_violations}/3.")
                if self.latency_violations >= 3:
                    logger.critical(f"🚨 [PANIC] 3 Consecutive High-Latency Orders! Engaging PASSIVE MODE/LOCK.")
                    try:
                        with open("EMERGENCY_KILL_SWITCH.lock", "w") as f:
                            f.write(f"LATENCY_PANIC: {exec_latency:.2f}ms")
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
            _order_metadata['actual_order_type'] = order_type  # 'limit' or 'market'
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
                strategy_version=getattr(event, 'strategy_version', '1.0.0'),
                # FORENSIC FIX #2: Carry metadata for fee attribution + margin release
                metadata=_order_metadata,
            )
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
                    logger.warning(f"⚠️ Protective orders failed: {ex}")

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"🚨 [FATAL-EXECUTION] Order {event.direction} for {event.symbol} failed!\nException: {e}\nTraceback:\n{tb}")
            if self.portfolio and side == 'buy':
                # Release pending cash
                self.portfolio.release_cash(event.quantity * (event.price or 0))
    
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
        logger.info(f"🛡️ [VIRTUAL SL/TP] Delegando Stops al Neural Ledger para evitar pisada de pies en {symbol_id} ({pos_side})")
        return
        
        # ═══════════════════════════════════════════════════════════════
        # BBO ARCHITECTURE: STOP (Limit) + TAKE_PROFIT (Limit)
        # SL Limit Price: slightly worse than trigger to ensure fill
        # TP Limit Price: at trigger price for pure Maker
        # ═══════════════════════════════════════════════════════════════
        use_limit_protective = getattr(Config, 'Execution', None) and getattr(Config.Execution, 'USE_LIMIT_PROTECTIVE_ORDERS', True)
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
            except:
                pass
        
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
                                except:
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
                            except:
                                price = 0
                            value = amount * price
                        
                        if price > 0:
                            total_value_usd += value
                            logger.info(f"  {asset:8s}: {amount:>15,.4f}  @ ${price:>10,.2f}  = ${value:>12,.2f}")
                        else:
                            logger.info(f"  {asset:8s}: {amount:>15,.4f}  (Price Unavailable)")
                    except:
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
                        except:
                            logger.info(f"  {asset:8s}: {total:>15,.6f}  (price unavailable)")
                else:
                    logger.info("  No assets found")
                    
            except Exception as delivery_error:
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
        
        import asyncio
        import ccxt
        
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
                        logger.warning(f"⚠️ [CHASE] Error fetching futures status {current_order_id}: {e}")
                        pass
                else:
                    try:
                        status_ccxt = await asyncio.wait_for(self.async_exchange.fetch_order(current_order_id, symbol_ccxt), timeout=5.0)
                        if status_ccxt['status'] in ['closed', 'canceled', 'rejected']:
                            logger.info(f"☑️ [CHASE] Order {current_order_id} concluded with status: {status_ccxt['status']}")
                            return
                    except Exception as e:
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

    async def stop(self):
        """Graceful shutdown"""
        if hasattr(self, 'user_stream'):
            await self.user_stream.stop()
        if hasattr(self, 'stream_task'):
            self.stream_task.cancel()
        if hasattr(self, 'async_exchange'):
            await self.async_exchange.close()
            logger.info("  ⚡ Closed async CCXT connection")
        logger.info("✅ [Executor] Stopped.")
