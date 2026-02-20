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

class BinanceExecutor:
    """
    Handles execution of orders on Binance via CCXT.
    Supports both Spot and Testnet.
    Integrated with UserDataStream for real-time fills.
    """
    def __init__(self, events_queue, portfolio=None):
        self.events_queue = events_queue
        self.portfolio = portfolio  # Reference for cash release on failure
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
            'timeout': 10000, # CRITICAL: 10s timeout
            'options': options
        })
        
        # Habilitar el modo correspondiente (Standardized Phase 6)
        if (hasattr(Config, 'BINANCE_USE_DEMO') and Config.BINANCE_USE_DEMO) or Config.BINANCE_USE_TESTNET:
            self.exchange.set_sandbox_mode(True)
            logger.info(f"🚀 Binance Executor: Running in {mode_description} (Sandbox Mode Active)")
        else:
            logger.info(f"🚀 Binance Executor: Running in {mode_description} (Live Mode ACTIVE)")

        # Phase 7: Guardián de Liquidez
        self.guardian = LiquidityGuardian(self.exchange)

            
        # Set Leverage for Futures
        # NOTE: Leverage endpoint not available on Testnet
        # Configure leverage manually in Binance Demo UI
        if Config.BINANCE_USE_FUTURES:
            logger.info("Binance Executor: FUTURES MODE ENABLED")
            logger.info("  → Using server-side leverage (configure in Binance Demo UI)")
        
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
                'timeout': 10000, # CRITICAL: 10s timeout
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                }
            })
            # Set Spot Testnet URLs (testnet.binance.vision)
            self.spot_exchange.set_sandbox_mode(True)
            logger.info("  → Spot exchange initialized for Testnet")
            
        # Phase 14: Rate Limiter
        from core.rate_limiter import PredictiveRateLimiter
        self.rate_limiter = PredictiveRateLimiter()

        # CRITICAL FIX: Load markets immediately to prevent "markets not loaded" error
        # 1. Set Position Mode to One-Way (Dual Side Position = False)
        # We use One-Way mode because our RiskManager assumes simple Buy/Sell
        # If account is in Hedge Mode, create_order fails without positionSide
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
        
        # [Dept 3 Fix] Start User Data Stream
        self.user_stream = UserDataStream(self.events_queue, self.exchange)
        self.stream_task = asyncio.create_task(self.user_stream.start())
        logger.info("✅ [Executor] User Data Stream Background Task Started")

    def _initialize_futures_settings(self):
        """
        Enforces One-Way Mode and Margin Type (Isolated) to match bot logic.
        """
        logger.info("Binance Executor: Initializing Futures Account Settings...")
        try:
            # 1. Set Position Mode to One-Way (Dual Side Position = False)
            # We use One-Way mode because our RiskManager assumes simple Buy/Sell
            # If account is in Hedge Mode, create_order fails without positionSide
            try:
                # 'true' = Hedge Mode, 'false' = One-Way Mode
                self.exchange.fapiPrivatePostPositionSideDual({'dualSidePosition': 'false'})
                logger.info("  ✅ Position Mode set to ONE-WAY")
            except ccxt.ExchangeError as e:
                error_msg = str(e)
                if "No need to change" in error_msg or "-4059" in error_msg:
                    logger.info("  ✅ Position Mode already ONE-WAY")
                else:
                    logger.warning(f"  ⚠️ Could not set Position Mode: {error_msg}")
            except ccxt.NetworkError as e:
                logger.error(f"  ❌ Network error setting Position Mode: {e}")
                logger.warning("     Continuing without Position Mode change...")

            # 2. Set Margin Type for all pairs
            logger.info(f"  ⏳ Setting Margin Type to {Config.BINANCE_MARGIN_TYPE} for {len(Config.TRADING_PAIRS)} pairs...")
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
                logger.warning(f"  ⚠️ Could not verify trading permissions: {e}")

            logger.info("  ✅ Margin Types Configured")
            
        except ccxt.NetworkError as e:
            logger.error(f"❌ Network error initializing Futures settings: {e}")
            logger.error("   Bot will continue but manual verification recommended")
        except ccxt.ExchangeError as e:
            logger.error(f"❌ Exchange error initializing Futures settings: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected error initializing Futures settings: {e}")
            logger.error("   Please report this error with the full traceback")

    @trace_execution
    def execute_order(self, event):
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
            logger.info(f"👻 [SHADOW] VIRTUAL EXECUTION: {event.side} {event.quantity} {event.symbol} @ {event.price or 'MKT'}")
            # TODO: Generate a FAKE FillEvent for Portfolio to track Shadow PnL?
            # For now, just blocking is enough to prevent real money loss.
            return

        symbol = event.symbol
        symbol_ccxt = symbol.replace('USDT', '/USDT')
        side = event.side.lower()
        order_type = event.order_type.lower()
        
        try:
            # 1. MARKETS LOADED CHECK
            if not self.exchange.markets:
                self.exchange.load_markets()
            
            market = self.exchange.market(symbol_ccxt)
            symbol_id = market['id']
            
            # 🛡️ PHASE II: ANTI-SLIPPAGE (Order Book Depth Check)
            # If MARKET order and liquidity is thin, downgrade to LIMIT or abort.
            if order_type == 'market':
                try:
                    orderbook = self.exchange.fetch_order_book(symbol_ccxt, limit=5)
                    bid = orderbook['bids'][0][0] if orderbook['bids'] else 0
                    ask = orderbook['asks'][0][0] if orderbook['asks'] else 0
                    
                    if bid > 0 and ask > 0:
                        spread_pct = (ask - bid) / bid
                        # Si el spread es > 0.1% (Muy alto para HFT), forzar LIMIT
                        if spread_pct > 0.001:
                            logger.warning(f"⚠️ High Spread ({spread_pct*100:.3f}%) detected for {symbol}. Downgrading to LIMIT.")
                            order_type = 'limit'
                            # Post at Best Bid/Ask
                            event.price = bid if side == 'sell' else ask
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
                     orderbook = self.exchange.fetch_order_book(symbol_ccxt, limit=5)
                     bid = orderbook['bids'][0][0]
                     ask = orderbook['asks'][0][0]
                     event.price = bid if side == 'buy' else ask
            
            if OrderSide.BUY in side: # Only check for BUYS (Entry/Cover)
                try:
                    quote_currency = market['quote']
                    cost_est = event.quantity * (event.price if event.price else self.guardian.get_last_price(symbol_ccxt))
                    
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
                    balance = self.exchange.fetch_free_balance(params=params)
                    available = balance.get(quote_currency, 0.0)
                    
                    if available < cost_est:
                         logger.error(f"🚫 [ATOMIC] INSUFFICIENT FUNDS! Need: {cost_est:.2f} {quote_currency}, Avail: {available:.2f}")
                         if self.portfolio: self.portfolio.release_cash(cost_est)
                         return
                         
                except Exception as e:
                    logger.warning(f"⚠️ Balance Check Skipped: {e}")

                except Exception as e:
                    logger.warning(f"⚠️ Balance Check Skipped: {e}")

            # 2. EXIT PRIORITY (Rule 2.1) - Skip Guardian if EXIT
            is_exit = getattr(event, 'is_exit', False) or (hasattr(event, 'side') and event.side == 'EXIT')
            
            if not is_exit:
                # Normal Signal: Run Guardian check
                liquidity = self.guardian.analyze_liquidity(symbol, event.quantity, event.side)
                if not liquidity['is_safe']:
                    logger.warning(f"🛡️ [GUARDIAN] Order Blocked: {liquidity['reason']}")
                    if self.portfolio: self.portfolio.release_cash(event.quantity * event.price if event.price else 0)
                    return
                
                # ✅ PHASE II.6: VWAP-RELATIVE EXECUTION (Smart Execution)
                # If buying significantly above VWAP, switch to LIMIT to avoid chasing tops.
                try:
                    from data.data_provider import get_data_provider
                    dp = get_data_provider()
                    bars = dp.get_latest_bars(symbol, n=15) # 15m VWAP
                    if bars is not None and len(bars) > 5:
                        # VWAP = Sum(Close * Vol) / Sum(Vol)
                        # Structured array: 'close', 'volume'
                        vsum = np.sum(bars['volume'])
                        if vsum > 0:
                            vwap_val = np.sum(bars['close'] * bars['volume']) / vsum
                            current_price = self.guardian.get_last_price(symbol_ccxt)
                            
                            # Logic: If BUY and Price > VWAP + 0.3% -> Switch MARKET to LIMIT
                            if OrderSide.BUY in side and current_price > vwap_val * 1.003:
                                if order_type == 'market' and not getattr(event, 'urgent', False):
                                    logger.info(f"📉 [VWAP] Price {current_price:.2f} > VWAP {vwap_val:.2f} (+0.3%). Switching to LIMIT/PASSIVE.")
                                    order_type = 'limit'
                                    event.price = current_price * 0.9995 # Bid side
                                    event.order_type = 'LIMIT'
                except Exception as e:
                    pass # Non-critical
                
                # --- PHASE 14: SMART-ORDER ROUTING (SOR) ---
                # Decide order type based on urgency and rebate priority
                is_urgent = getattr(event, 'urgent', False)
                rebate_priority = getattr(self.portfolio, 'rebate_priority', True)
                
                if order_type == 'limit' and is_urgent:
                    logger.info("⚡ [SOR] Urgency detected: Switching LIMIT to MARKET to ensure entry.")
                    order_type = 'market'
                elif order_type == 'market' and rebate_priority and not is_urgent:
                    logger.info("💰 [SOR] Rebate Priority Active: Switching MARKET to LIMIT (Post-Only).")
                    order_type = 'limit'
                    # Post-Only flag for Binance
                    event.metadata = event.metadata or {}
                    event.metadata['timeInForce'] = 'GTX' # GTX = Post Only
                
                smart_price = liquidity.get('avg_fill_price', event.price)
            else:
                # EXIT: High Priority - Use current market price directly
                smart_price = event.price

            # 3. SURGICAL PRECISION (Roundings)
            qty_precision = self.exchange.amount_to_precision(symbol_ccxt, event.quantity)
            final_qty = float(qty_precision)
            
            # Sniper Logic for LIMIT orders (Aggressive pricing to capture liquidity)
            # PHASE 13: Enhanced V3 Sniper
            # Si es una entrada agresiva (Imbalance > 3.0), empujamos el precio para asegurar el fill.
            if order_type == 'limit':
                spread_adj = 0.0001 # Default 0.01% bias
                
                # Check for Sniper Condition in metadata
                if event.metadata and event.metadata.get('sniper_mode'):
                    spread_adj = 0.0003 # 0.03% more aggressive
                    logger.info(f"🎯 [SNIPER_V3] Aggressive Entry engaged for {symbol}")

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
                    reference_price = self.guardian.get_last_price(symbol) if hasattr(self.guardian, 'get_last_price') else None
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

            # 4. SEND ORDER
            logger.info(f"⚡ [EXEC] {order_type.upper()} {side.upper()} {symbol} | Qty: {final_qty} | P: {final_price}")
            
            if Config.BINANCE_USE_FUTURES:
                # Use Raw API for minimum latency
                params = {
                    'symbol': symbol_id,
                    'side': side.upper(),
                    'type': order_type.upper(),
                    'quantity': qty_precision,
                    'newOrderRespType': 'RESULT',
                    'recvWindow': 60000
                }
                if order_type == 'limit':
                    params['price'] = price_precision
                    params['timeInForce'] = 'GTC' # Good Till Cancel
                
                order_raw = self.exchange.fapiPrivatePostOrder(params)
                order = order_raw # Simplified mapping
            else:
                # SPOT
                order = self.exchange.create_order(
                    symbol=symbol_ccxt,
                    type=order_type,
                    side=side,
                    amount=final_qty,
                    price=final_price if order_type == 'limit' else None
                )
                if 'info' in order: order = order['info']

            # 5. PROCESS RESPONSE & EMIT FILL
            end_exec = time.perf_counter()
            exec_latency = (end_exec - start_exec) * 1000
            latency_monitor.track('order_to_send', exec_latency)
            
            fill_price = float(order.get('avgPrice', final_price if order_type == 'limit' else 0.0))
            filled_qty = float(order.get('executedQty', final_qty))
            order_id = str(order.get('orderId', ''))
            is_fully_filled = (filled_qty >= event.quantity * 0.9999)  # Tolerance for floating point
            
            logger.info(f"✅ Order OK: {order_id} | Filled: {filled_qty} @ {fill_price} in {exec_latency:.2f}ms")
            
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
                sl_pct=getattr(event, 'sl_pct', None),
                tp_pct=getattr(event, 'tp_pct', None),
                # Phase 31: Partial Fill Logic
                is_closed=is_fully_filled
            )
            self.events_queue.put(fill_event)
            
            # 6. TRACKING & PROTECTIVE ORDERS
            if self.order_manager and order_id and order_type == 'limit':
                self.order_manager.track_order(
                    order_id, 
                    symbol, 
                    order_type, 
                    side, 
                    final_price, 
                    final_qty, 
                    getattr(event, 'strategy_id', 'Unknown'),
                    ttl=getattr(event, 'ttl', None),
                    metadata=getattr(event, 'metadata', None) # Pass chase count
                )
            
            # Exchange-Based Protective Orders (Failsafe Layer 3)
            # [DF-C7 FIX] Use filled_qty (actual) NOT final_qty (requested)
            # POR QUÉ: If only 30% filled, placing SL for 100% would attempt
            #   to close a position larger than what we hold → Binance error
            #   or phantom position risk.
            if Config.BINANCE_USE_FUTURES and filled_qty > 0:
                try:
                    sl_pct = getattr(event, 'sl_pct', 0.003) or 0.003
                    tp_pct = getattr(event, 'tp_pct', 0.008) or 0.008
                    self._place_protective_orders(symbol_id, side.upper(), filled_qty, fill_price, sl_pct, tp_pct)
                except Exception as ex:
                    logger.warning(f"⚠️ Protective orders failed: {ex}")

        except Exception as e:
            handle_order_error(e, symbol, side, event.quantity)
            if self.portfolio and side == 'buy':
                # Release pending cash
                self.portfolio.release_cash(event.quantity * (event.price or 0))
    
    def _place_protective_orders(self, symbol_id, side, quantity, entry_price, sl_pct, tp_pct):
        """
        LAYER 3: Place stop-loss and take-profit orders on Binance servers.
        These act as failsafe if the bot crashes or loses connection.
        
        Args:
            symbol_id: Exchange symbol (e.g., 'BTCUSDT')
            side: 'BUY' or 'SELL' (direction of the ENTRY order)
            quantity: Position size
            entry_price: Entry price
            sl_pct: Stop Loss percentage
            tp_pct: Take Profit percentage
        """
        # Calculate stop and target prices
        if side.upper() == 'BUY':  # LONG position
            stop_price = entry_price * (1 - sl_pct)
            target_price = entry_price * (1 + tp_pct)
            stop_side = 'SELL'  # Close LONG with SELL
        else:  # SHORT position
            stop_price = entry_price * (1 + sl_pct)
            target_price = entry_price * (1 - tp_pct)
            stop_side = 'BUY'  # Close SHORT with BUY
        
        # Format prices to exchange precision
        stop_price_str = self.exchange.price_to_precision(symbol_id.replace('USDT', '/USDT'), stop_price)
        target_price_str = self.exchange.price_to_precision(symbol_id.replace('USDT', '/USDT'), target_price)
        qty_str = self.exchange.amount_to_precision(symbol_id.replace('USDT', '/USDT'), quantity)
        
        try:
            # Place STOP_MARKET order (stop-loss)
            stop_params = {
                'symbol': symbol_id,
                'side': stop_side,
                'type': 'STOP_MARKET',
                'quantity': qty_str,
                'stopPrice': stop_price_str,
                'closePosition': 'true',  # Automatically close the position
                'newOrderRespType': 'RESULT'
            }
            stop_order = self.exchange.fapiPrivatePostOrder(stop_params)
            logger.info(f"  🛑 Stop-Loss placed at {stop_price_str} (Order ID: {stop_order.get('orderId')})")
            
            # Place TAKE_PROFIT_MARKET order
            tp_params = {
                'symbol': symbol_id,
                'side': stop_side,
                'type': 'TAKE_PROFIT_MARKET',
                'quantity': qty_str,
                'stopPrice': target_price_str,
                'closePosition': 'true',
                'newOrderRespType': 'RESULT'
            }
            tp_order = self.exchange.fapiPrivatePostOrder(tp_params)
            logger.info(f"  💰 Take-Profit placed at {target_price_str} (Order ID: {tp_order.get('orderId')})")
            
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
                result = self.exchange.fapiPrivateDeleteOrder({
                    'symbol': symbol_id,
                    'orderId': order_id,
                    'recvWindow': 60000
                })
                logger.info(f"🗑️ [EXEC] Deleted Futures Order: {order_id} ({symbol})")
            else:
                # SPOT
                result = self.exchange.cancel_order(order_id, symbol)
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
            if Config.BINANCE_USE_FUTURES and ((hasattr(Config, 'BINANCE_USE_DEMO') and Config.BINANCE_USE_DEMO) or Config.BINANCE_USE_TESTNET):
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
            if (hasattr(Config, 'BINANCE_USE_DEMO') and Config.BINANCE_USE_DEMO) or Config.BINANCE_USE_TESTNET:
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
                except ccxt.NetworkError as e:
                    logger.warning(f"Network error in fetch_positions fallback: {e}. Trying raw v2 endpoint...")
                    # Fallback: Ensure URL exists and try raw
                    if 'fapiPrivateV2' not in self.exchange.urls['api']:
                        self.exchange.urls['api']['fapiPrivateV2'] = self.exchange.urls['api']['fapiPrivate'].replace('v1', 'v2')
                        if 'test' in self.exchange.urls:
                            self.exchange.urls['test']['fapiPrivateV2'] = self.exchange.urls['test']['fapiPrivate'].replace('v1', 'v2')
                    
                    positions = self.exchange.fapiPrivateV2GetPositionRisk()
                
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
                        
                        portfolio.positions[internal_symbol] = {
                            'quantity': amt,
                            'avg_price': entry_price,
                            'current_price': entry_price # Will be updated by data feed
                        }
                        
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
                    if hasattr(self, 'spot_exchange') and self.spot_exchange:
                        balance_data = self.spot_exchange.fetch_balance()
                    else:
                        balance_data = self.exchange.fetch_balance()
                    
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

    async def stop(self):
        """Graceful shutdown"""
        if hasattr(self, 'user_stream'):
            await self.user_stream.stop()
        if hasattr(self, 'stream_task'):
            self.stream_task.cancel()
        logger.info("✅ [Executor] Stopped.")
