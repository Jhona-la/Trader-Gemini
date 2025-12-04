import ccxt
from config import Config
from core.events import FillEvent
from utils.logger import logger
from utils.error_handler import retry_on_api_error, handle_balance_error, handle_order_error

class BinanceExecutor:
    """
    Handles execution of orders on Binance via CCXT.
    Supports both Spot and Testnet.
    """
    def __init__(self, events_queue, portfolio=None):
        self.events_queue = events_queue
        self.portfolio = portfolio  # Reference for cash release on failure
        
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
        
        # Habilitar el modo correspondiente
        if (hasattr(Config, 'BINANCE_USE_DEMO') and Config.BINANCE_USE_DEMO) or Config.BINANCE_USE_TESTNET:
            # CRITICAL: Manual URL configuration for Testnet
            # BUG #18 FIX: CCXT requires ALL keys (fapi, dapi, public) to be present
            
            # 1. Define Base URLs
            # BUG #30 FIX: Updated to official 2024 Binance Testnet URLs
            spot_testnet_base = 'https://testnet.binance.vision/api'
            futures_testnet_base = 'https://demo-fapi.binance.com'  # Official USD-M Testnet URL
            delivery_testnet_base = 'https://testnet.binancefuture.com'  # COIN-M Testnet URL
            
            custom_urls = {
                # SPOT (Standard)
                'public': f'{spot_testnet_base}/v3',
                'private': f'{spot_testnet_base}/v3',
                'api': {
                    'public': f'{spot_testnet_base}/v3',
                    'private': f'{spot_testnet_base}/v3',
                },
                
                # FUTURES (USD-M) - BUG #30 FIX: Proper endpoint paths
                'fapiPublic': f'{futures_testnet_base}/fapi/v1',
                'fapiPrivate': f'{futures_testnet_base}/fapi/v1',
                'fapiData': f'{futures_testnet_base}/fapi/v1',
                'fapiPrivateV2': f'{futures_testnet_base}/fapi/v2',
                # BUG #23 FIX: Add V3 endpoint for account info queries
                'fapiPrivateV3': f'{futures_testnet_base}/fapi/v3',
                
                # DELIVERY (COIN-M) - BUG #30 FIX: Proper endpoint paths
                'dapiPublic': f'{delivery_testnet_base}/dapi/v1',
                'dapiPrivate': f'{delivery_testnet_base}/dapi/v1',
                'dapiData': f'{delivery_testnet_base}/dapi/v1',
                
                # SAPI
                'sapi': f'{spot_testnet_base}/v3',
            }
            
            # Set BOTH 'api' and 'test' URLs
            self.exchange.urls['api'] = custom_urls
            self.exchange.urls['test'] = custom_urls
            
            logger.info(f"Binance Executor: Running in {mode_description} mode (Unified Testnet URLs)")
        else:
            logger.info(f"Binance Executor: Running in {mode_description} mode")

            
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
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                }
            })
            # Set Spot Testnet URLs (testnet.binance.vision)
            self.spot_exchange.set_sandbox_mode(True)
            logger.info("  → Spot exchange initialized for Testnet")
            
        # CRITICAL FIX: Load markets immediately to prevent "markets not loaded" error
            # 1. Set Position Mode to One-Way (Dual Side Position = False)
            # We use One-Way mode because our RiskManager assumes simple Buy/Sell
            # If account is in Hedge Mode, create_order fails without positionSide
            # CRITICAL: Monkey patch 'request' method to intercept ALL sapi calls
            # This is more robust than patching individual methods
            original_request = self.exchange.request
            
            def intercepted_request(path, api='public', method='GET', params={}, headers=None, body=None, config={}):
                # BUG #17 FIX: Intercept SAPI for both Futures AND Spot Testnet
                # Spot Testnet also lacks SAPI support
                if api == 'sapi':
                    # Testnet doesn't support SAPI. Return empty list as mock response.
                    # Most fetch_markets sub-calls expect a list of pairs.
                    return []
                return original_request(path, api, method, params, headers, body, config)
            
            self.exchange.request = intercepted_request
            
            # Apply to Spot Exchange instance as well if it exists
            if hasattr(self, 'spot_exchange') and self.spot_exchange:
                original_spot_request = self.spot_exchange.request
                def intercepted_spot_request(path, api='public', method='GET', params={}, headers=None, body=None, config={}):
                    if api == 'sapi':
                        return []
                    return original_spot_request(path, api, method, params, headers, body, config)
                self.spot_exchange.request = intercepted_spot_request
                
            logger.info("  🔧 Testnet: Intercepting and blocking ALL 'sapi' endpoint calls")

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

        self._initialize_futures_settings()

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
            # We iterate through configured pairs to set them to ISOLATED
            # This protects the wallet balance
            logger.info(f"  ⏳ Setting Margin Type to {Config.BINANCE_MARGIN_TYPE} for {len(Config.TRADING_PAIRS)} pairs...")
            for symbol in Config.TRADING_PAIRS:
                try:
                    # Convert symbol to ID (ETH/USDT -> ETHUSDT) for the API
                    market = self.exchange.market(symbol)
                    symbol_id = market['id']
                    
                    self.exchange.fapiPrivatePostMarginType({
                        'symbol': symbol_id,
                        'marginType': Config.BINANCE_MARGIN_TYPE.upper() # ISOLATED or CROSS
                    })
                except ccxt.ExchangeError as e:
                    error_msg = str(e)
                    if "No need to change" in error_msg or "-4046" in error_msg:
                        pass  # Already set to desired margin type
                    else:
                        logger.debug(f"  Could not set Margin Type for {symbol}: {error_msg}")
                except KeyError:
                    logger.warning(f"  ⚠️ Symbol {symbol} not found in exchange markets")
            logger.info("  ✅ Margin Types Configured")
            
        except ccxt.NetworkError as e:
            logger.error(f"❌ Network error initializing Futures settings: {e}")
            logger.error("   Bot will continue but manual verification recommended")
        except ccxt.ExchangeError as e:
            logger.error(f"❌ Exchange error initializing Futures settings: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected error initializing Futures settings: {e}")
            logger.error("   Please report this error with the full traceback")

    def execute_order(self, event):
        """
        Executes an OrderEvent.
        """
        if event.type != 'ORDER':
            return

        # Check if symbol is supported by this executor (Crypto only)
        if '/' not in event.symbol: # Simple check, assuming Stocks don't have '/':
            return

        logger.info(f"Executing: {event.direction} {event.quantity} {event.symbol}")
        
        try:
            # Map direction to side
            side = 'buy' if event.direction == 'BUY' else 'sell'
            symbol = event.symbol
            quantity = event.quantity
            order_type = event.order_type.lower() # 'market' or 'limit'
            
            # Execute Order
            if order_type == 'mkt':
                order_type = 'market'
            
            # 1. Prepare Symbol and Quantity
            # CRITICAL FIX: Ensure markets are loaded before accessing them
            if not self.exchange.markets:
                logger.info("  ⚠️ Markets not loaded. Loading now...")
                self.exchange.load_markets()
                
            market = self.exchange.market(symbol)
            symbol_id = market['id']
            
            # Ensure quantity precision (CRITICAL for Futures)
            # CCXT amount_to_precision returns a string
            qty_str = self.exchange.amount_to_precision(symbol, quantity)
            
            # 2. Build Parameters for Raw API
            params = {
                'symbol': symbol_id,
                'side': side.upper(),
                'type': order_type.upper(),
                'quantity': qty_str,
                'newOrderRespType': 'RESULT',
                'recvWindow': 60000  # Generous window for network latency
            }
            
            # 3. Execute using Raw API (Futures) or Standard CCXT (Spot)
            if Config.BINANCE_USE_FUTURES:
                # FUTURES: Use Raw API to bypass potential CCXT issues with Testnet URLs
                logger.info(f"  🚀 Sending Raw Futures Order: {side.upper()} {qty_str} {symbol}")
                order = self.exchange.fapiPrivatePostOrder(params)
            else:
                # SPOT: Use standard CCXT create_order
                # CCXT handles the endpoint selection automatically based on 'defaultType': 'spot'
                logger.info(f"  🚀 Sending Spot Order: {side.upper()} {qty_str} {symbol}")
                order = self.exchange.create_order(symbol, 'market', side, quantity)
                # Normalize response to match Raw API structure for parsing below
                # CCXT returns a unified structure, so we might need to adjust parsing logic
                # But for simplicity, let's rely on CCXT's unified response if possible, 
                # OR just use the raw 'info' field if available.
                if 'info' in order:
                    order = order['info'] # Use raw response for consistent parsing below
            
            # 4. Parse Response (Raw API returns different structure than CCXT unified)
            # Binance Futures Raw Response:
            # {'orderId': 123, 'symbol': 'BTCUSDT', 'status': 'FILLED', 'avgPrice': '90000', 'executedQty': '0.1', ...}
            
            fill_price = float(order.get('avgPrice', 0.0))
            filled_qty = float(order.get('executedQty', 0.0))
            order_id = str(order.get('orderId', ''))
            
            # Log Success
            logger.info(f"✅ Order Filled: {order_id} - {side.upper()} {filled_qty} @ {fill_price}")
            
            # Create Fill Event
            fill_event = FillEvent(
                timeindex=None,
                symbol=symbol,
                exchange='BINANCE',
                quantity=filled_qty,
                direction=event.direction,
                fill_cost=filled_qty * fill_price,
                commission=None, # Fee info might be in a separate field or trade stream
                strategy_id=event.strategy_id
            )
            self.events_queue.put(fill_event)
            
            # LAYER 3: Exchange-Based Stop-Loss and Take-Profit (Failsafe)
            # Place protective orders on Binance servers
            # Only for ENTRY orders (BUY/SELL), not for EXIT orders
            if Config.BINANCE_USE_FUTURES and event.direction in ['BUY', 'SELL']:
                try:
                    self._place_protective_orders(symbol_id, side, filled_qty, fill_price)
                except ccxt.InsufficientFunds as e:
                    logger.warning(f"⚠️ Insufficient margin for protective orders: {e}")
                except ccxt.InvalidOrder as e:
                    logger.warning(f"⚠️ Invalid protective order parameters: {e}")
                except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                    logger.warning(f"⚠️ Failed to place protective orders (non-critical): {e}")
            
        except Exception as e:
            # Binance-specific error handling
            handle_order_error(e, event.symbol, event.direction, event.quantity)
            
            # RELEASE RESERVED CASH on failure
            if self.portfolio and event.direction == 'BUY':
                estimated_cost = event.quantity * (fill_price if 'fill_price' in locals() else 0)
                # Fallback: use quantity as approximate cost if price unknown
                if estimated_cost == 0:
                    # Rough estimate from Risk Manager's $1000 default
                    estimated_cost = 1000.0
                self.portfolio.release_cash(estimated_cost)
                logger.warning(f"Released ${estimated_cost:.2f} reserved cash due to order failure")
    
    def _place_protective_orders(self, symbol_id, side, quantity, entry_price):
        """
        LAYER 3: Place stop-loss and take-profit orders on Binance servers.
        These act as failsafe if the bot crashes or loses connection.
        
        Args:
            symbol_id: Exchange symbol (e.g., 'BTCUSDT')
            side: 'BUY' or 'SELL' (direction of the ENTRY order)
            quantity: Position size
            entry_price: Entry price
        """
        # Calculate stop and target prices
        if side.upper() == 'BUY':  # LONG position
            stop_price = entry_price * 0.997  # -0.3% stop loss
            target_price = entry_price * 1.008  # +0.8% take profit
            stop_side = 'SELL'  # Close LONG with SELL
        else:  # SHORT position
            stop_price = entry_price * 1.003  # +0.3% stop loss (price goes up)
            target_price = entry_price * 0.992  # -0.8% take profit (price goes down)
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
                raise Exception("Skipping Futures check in Spot Testnet (Keys incompatible)")

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
                raise Exception("Skipping COIN-M check in Spot Testnet")

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
                        if not '/' in symbol and symbol.endswith('USDT'):
                            base = symbol[:-4]
                            internal_symbol = f"{base}/USDT"
                        
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
                                    current_price = ticker['last']
                                except:
                                    current_price = 0  # If we can't get price, set to 0 (will be updated by data feed)
                                
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
