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
        }
        
        if Config.BINANCE_USE_FUTURES:
            options['defaultType'] = 'future'
        else:
            options['defaultType'] = 'spot'
        
        # Determinar qué API keys usar según el modo
        if hasattr(Config, 'BINANCE_USE_DEMO') and Config.BINANCE_USE_DEMO:
            api_key = Config.BINANCE_DEMO_API_KEY
            secret_key = Config.BINANCE_DEMO_SECRET_KEY
            mode_description = "DEMO TRADING (Futures con capital virtual)"
        elif Config.BINANCE_USE_TESTNET:
            api_key = Config.BINANCE_TESTNET_API_KEY
            secret_key = Config.BINANCE_TESTNET_SECRET_KEY
            mode_description = "TESTNET"
        else:
            api_key = Config.BINANCE_API_KEY
            secret_key = Config.BINANCE_SECRET_KEY
            mode_description = "LIVE"
            
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret_key,
            'enableRateLimit': True,
            'options': options
        })
        
        # Habilitar el modo correspondiente
        if (hasattr(Config, 'BINANCE_USE_DEMO') and Config.BINANCE_USE_DEMO) or Config.BINANCE_USE_TESTNET:
            # CRITICAL: Manual URL configuration for Futures Testnet
            # DO NOT use set_sandbox_mode() as it overwrites these URLs
            # Per CCXT docs: use EITHER set_sandbox_mode() OR manual URLs, not both
            custom_urls = {
                'public': 'https://testnet.binancefuture.com/fapi/v1',
                'private': 'https://testnet.binancefuture.com/fapi/v1',
                'fapiPublic': 'https://testnet.binancefuture.com/fapi/v1',
                'fapiPrivate': 'https://testnet.binancefuture.com/fapi/v1',
                'fapiPrivateV2': 'https://testnet.binancefuture.com/fapi/v2',
                'fapiData': 'https://testnet.binancefuture.com/fapi/v1',
                'dapiPublic': 'https://testnet.binancefuture.com/dapi/v1',
                'dapiPrivate': 'https://testnet.binancefuture.com/dapi/v1',
                'dapiData': 'https://testnet.binancefuture.com/dapi/v1',
            }
            
            # Set BOTH 'api' and 'test' URLs
            self.exchange.urls['api'] = custom_urls
            self.exchange.urls['test'] = custom_urls
            
            logger.info(f"Binance Executor: Running in {mode_description} mode (Manual Futures URLs)")
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
        else:
            # In production, we can use the same exchange for both
            self.spot_exchange = self.exchange
            logger.info("  → Using main exchange for Spot queries (Production mode)")
            
        # ===================================================================
        # INITIALIZE ACCOUNT SETTINGS (Best Practices)
        # ===================================================================
        if Config.BINANCE_USE_FUTURES:
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
            except Exception as e:
                if "No need to change" in str(e):
                    logger.info("  ✅ Position Mode already ONE-WAY")
                else:
                    logger.warning(f"  ⚠️ Could not set Position Mode: {e}")

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
                except Exception as e:
                    if "No need to change" in str(e):
                        pass # Already set
                    else:
                        # Log only critical errors, ignore "No need to change"
                        # logger.warning(f"  ⚠️ Could not set Margin Type for {symbol}: {e}")
                        pass
            logger.info("  ✅ Margin Types Configured")
            
        except Exception as e:
            logger.error(f"Failed to initialize Futures settings: {e}")

    def execute_order(self, event):
        """
        Executes an OrderEvent.
        """
        if event.type != 'ORDER':
            return

        # Check if symbol is supported by this executor (Crypto only)
        if '/' not in event.symbol: # Simple check, assuming Stocks don't have '/':
            return

        print(f"Binance Executing: {event.direction} {event.quantity} {event.symbol}")
        
        try:
            # Map direction to side
            side = 'buy' if event.direction == 'BUY' else 'sell'
            symbol = event.symbol
            quantity = event.quantity
            order_type = event.order_type.lower() # 'market' or 'limit'
            
            # Execute Order using CCXT unified API
            if order_type == 'mkt':
                order_type = 'market'
            
            # Use standard CCXT method with manual URLs configured above
            params = {}
            if Config.BINANCE_USE_FUTURES:
                params['newOrderRespType'] = 'RESULT'  # Get immediate execution details
                
            order = self.exchange.create_order(symbol, order_type, side, quantity, params=params)
            
            # Log Success
            print(f"Order Filled: {order['id']} - {side} {order['filled']} @ {order['average']}")
            
            # Create Fill Event
            fill_event = FillEvent(
                timeindex=None,
                symbol=symbol,
                exchange='BINANCE',
                quantity=order['filled'],
                direction=event.direction,
                fill_cost=order['cost'],
                commission=order.get('fee', None),
                strategy_id=event.strategy_id  # PASS strategy_id from OrderEvent
            )
            self.events_queue.put(fill_event)
            
        except Exception as e:
            print(f"Binance Execution Error: {e}")
            
            # RELEASE RESERVED CASH on failure
            if self.portfolio and event.direction == 'BUY':
                estimated_cost = event.quantity * (fill_price if 'fill_price' in locals() else 0)
                # Fallback: use quantity as approximate cost if price unknown
                if estimated_cost == 0:
                    # Rough estimate from Risk Manager's $1000 default
                    estimated_cost = 1000.0
import ccxt
from config import Config
from core.events import FillEvent


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
        }
        
        if Config.BINANCE_USE_FUTURES:
            options['defaultType'] = 'future'
        else:
            options['defaultType'] = 'spot'
        
        # Determinar qué API keys usar según el modo
        if hasattr(Config, 'BINANCE_USE_DEMO') and Config.BINANCE_USE_DEMO:
            api_key = Config.BINANCE_DEMO_API_KEY
            secret_key = Config.BINANCE_DEMO_SECRET_KEY
            mode_description = "DEMO TRADING (Futures con capital virtual)"
        elif Config.BINANCE_USE_TESTNET:
            api_key = Config.BINANCE_TESTNET_API_KEY
            secret_key = Config.BINANCE_TESTNET_SECRET_KEY
            mode_description = "TESTNET"
        else:
            api_key = Config.BINANCE_API_KEY
            secret_key = Config.BINANCE_SECRET_KEY
            mode_description = "LIVE"
            
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret_key,
            'enableRateLimit': True,
            'options': options
        })
        
        # Habilitar el modo correspondiente
        if hasattr(Config, 'BINANCE_USE_DEMO') and Config.BINANCE_USE_DEMO:
            # MANUAL CONFIGURATION for Futures Demo (set_sandbox_mode is deprecated for futures)
            self.exchange.urls['api'] = {
                'public': 'https://testnet.binancefuture.com/fapi/v1',
                'private': 'https://testnet.binancefuture.com/fapi/v1',
                'fapiPublic': 'https://testnet.binancefuture.com/fapi/v1',
                'fapiPrivate': 'https://testnet.binancefuture.com/fapi/v1',
                'fapiData': 'https://testnet.binancefuture.com/fapi/v1',
                'dapiPublic': 'https://testnet.binancefuture.com/dapi/v1',
                'dapiPrivate': 'https://testnet.binancefuture.com/dapi/v1',
                'dapiData': 'https://testnet.binancefuture.com/dapi/v1',
                'sapi': 'https://testnet.binance.vision/api/v3', # Fallback for some endpoints
            }
            print(f"Binance Executor: Running in {mode_description} mode (Manual URL Config).")
        elif Config.BINANCE_USE_TESTNET:
            self.exchange.set_sandbox_mode(True)
            print(f"Binance Executor: Running in {mode_description} mode.")
        else:
            print(f"Binance Executor: Running in {mode_description} mode.")
            
        # Set Leverage for Futures
        # NOTE: Leverage endpoint not available on Testnet
        # Configure leverage manually in Binance Demo UI
        if Config.BINANCE_USE_FUTURES:
            print(f"Binance Executor: FUTURES MODE ENABLED")
            print(f"  → Using server-side leverage (configure in Binance Demo UI)")

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
                
            order = self.exchange.create_order(symbol, order_type, side, quantity)
            
            # Log Success
            fill_price = order.get('average', 0) or 0
            logger.info(f"✅ Order Filled: {order['id']} - {side.upper()} {order['filled']} {symbol} @ ${fill_price:.2f}")
            
            # Create Fill Event
            fill_event = FillEvent(
                timeindex=None,
                symbol=symbol,
                exchange='BINANCE',
                quantity=order['filled'],
                direction=event.direction,
                fill_cost=order['cost'],
                commission=order.get('fee', None),
                strategy_id=event.strategy_id  # PASS strategy_id from OrderEvent
            )
            self.events_queue.put(fill_event)
            
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

    def get_all_balances(self):
        """
        Fetches and displays COMPLETE account information from all Binance wallets:
        1. USDT-M Futures (USDT-Ⓜ) - Full Account Info
        2. COIN-M Futures (COIN-Ⓜ) - Full Account Info
        3. Spot (Balance Estimado)
        
        Returns the primary USDT-M total wallet balance for portfolio sync.
        """
        print("\n" + "=" * 70)
        print("💰 BINANCE ACCOUNT - COMPLETE ANALYSIS")
        print("=" * 70)
        
        primary_balance = None
        
        # ===================================================================
        # 1. USDT-M Futures - COMPLETE ACCOUNT INFORMATION
        # ===================================================================
        try:
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
            
            print("\n📊 USDT-Ⓜ FUTURES - COMPLETE ACCOUNT STATUS")
            print("-" * 70)
            print(f"  💵 Wallet Balance:       ${total_wallet:>15,.2f}")
            print(f"  📈 Margin Balance:       ${total_margin:>15,.2f}")
            print(f"  💰 Available Balance:    ${available:>15,.2f}")
            print(f"  🎯 Max Withdraw:         ${max_withdraw:>15,.2f}")
            print(f"  {'📉' if total_unpnl < 0 else ' 📈'} Unrealized PnL:       ${total_unpnl:>15,.2f}")
            print(f"\n  ⚠️  MARGIN METRICS:")
            print(f"     Initial Margin:       ${total_init_margin:>15,.2f}")
            print(f"     Maintenance Margin:   ${total_maint_margin:>15,.2f}")
            print(f"     Position Margin:      ${total_pos_margin:>15,.2f}")
            print(f"     Order Margin:         ${total_order_margin:>15,.2f}")
            if total_maint_margin > 0:
                print(f"     🚨 Margin Ratio:      {margin_ratio:>16,.2f}%")
            
            # Show positions if any
            positions = account_info.get('positions', [])
            active_positions = [p for p in positions if float(p.get('positionAmt', 0)) != 0]
            
            if active_positions:
                print(f"\n  📍 OPEN POSITIONS ({len(active_positions)}):")
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
                    
                    print(f"     {color} {symbol:12} {side:5} {abs(amt):>10.4f} @ ${entry_price:<10,.2f}")
                    print(f"        Leverage: {leverage}x | {margin_type} | PnL: ${unpnl:+,.2f} | Notional: ${abs(notional):,.2f}")
            else:
                print(f"\n  📍 No open positions")
            
            primary_balance = total_wallet
            
        except Exception as e:
            logger.warning(f"Could not fetch USDT-M Futures account info: {e}")
            print("\n⚠ USDT-Ⓜ Futures: Error fetching data")
            # Fallback to balance-only
            try:
                response = self.exchange.fapiPrivateV2GetBalance()
                for asset in response:
                    if asset['asset'] == 'USDT':
                        primary_balance = float(asset['balance'])
                        print(f"  (Fallback) Balance: ${primary_balance:,.2f}")
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
            
            # GET /dapi/v1/balance
            coin_balances = self.exchange.dapiPrivateGetBalance()
            
            has_coin_balance = False
            print(f"\n📊 COIN-Ⓜ FUTURES")
            print("-" * 70)
            
            for asset_info in coin_balances:
                balance = float(asset_info.get('balance', 0))
                if balance > 0:
                    asset = asset_info['asset']
                    available = float(asset_info.get('availableBalance', 0))
                    cross_unpnl = float(asset_info.get('crossUnPnl', 0))
                    
                    print(f"  💎 {asset:8} Balance: {balance:>12.6f} | Available: {available:>12.6f} | UnPnL: {cross_unpnl:+.6f}")
                    has_coin_balance = True
            
            if not has_coin_balance:
                print(f"  No balances")
                    
        except Exception as e:
            logger.warning(f"Could not fetch COIN-M Futures info: {e}")
            print(f"\n📊 COIN-Ⓜ FUTURES")
            print("-" * 70)
            print(f"  Error fetching data")
        
        # ===================================================================
        # 3. Spot - ACCOUNT INFORMATION
        # ===================================================================
        try:
            # Use the permanent Spot exchange instance
            response = self.spot_exchange.fetch_balance()
            
            # Calculate total estimated value
            total_usdt = 0.0
            total_btc = 0.0
            non_zero_balances = []
            
            for asset, amounts in response['total'].items():
                if amounts > 0:
                    if asset == 'USDT':
                        total_usdt += amounts
                    elif asset == 'BTC':
                        total_btc += amounts
                    non_zero_balances.append({
                        'asset': asset,
                        'free': response['free'].get(asset, 0),
                        'locked': response['used'].get(asset, 0),
                        'total': amounts
                    })
            
            print(f"\n📊 SPOT (Balance Estimado)")
            print("-" * 70)
            
            if total_usdt > 0 or total_btc > 0:
                if total_usdt > 0:
                    print(f"  💵 USDT Total:     ${total_usdt:>15,.2f}")
                if total_btc > 0:
                    print(f"  ₿  BTC Total:      {total_btc:>16.8f}")
           
            # Show top non-zero balances
            if non_zero_balances:
                # Sort by total value (descending)
                sorted_balances = sorted(non_zero_balances, key=lambda x: x['total'], reverse=True)
                
                print(f"\n  💰 Assets ({len(non_zero_balances)} total):")
                for b in sorted_balances[:5]:  # Top 5
                    asset = b['asset']
                    free = b['free']
                    locked = b['locked']
                    total = b['total']
                    print(f"     {asset:8} Free: {free:>12,.4f} | Locked: {locked:>12,.4f} | Total: {total:>12,.4f}")
                
                if len(non_zero_balances) > 5:
                    print(f"     ... and {len(non_zero_balances) - 5} more")
            else:
                print(f"  No balances")
                    
        except Exception as e:
            # Log the specific error for debugging
            error_msg = str(e)
            if 'testnet.binancefuture.com' in error_msg or '404' in error_msg:
                logger.debug(f"Spot query hit Futures server (expected in Futures-only mode): {e}")
                print(f"\n📊 SPOT (Balance Estimado)")
                print("-" * 70)
                print(f"  Not available (Futures Testnet - different server)")
            else:
                logger.warning(f"Could not fetch Spot balance: {e}")
                print(f"\n📊 SPOT (Balance Estimado)")
                print("-" * 70)
                print(f"  Error: {error_msg[:50]}...")
        
        
        print("=" * 70 + "\n")
        
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
                    logger.warning(f"Standard fetch_positions failed: {e}. Trying raw v2 endpoint...")
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
                        synced_count += 1
                        logger.info(f"  -> Found Position: {internal_symbol} {amt} @ ${entry_price:.2f}")
                
                if synced_count > 0:
                    logger.info(f"Synced {synced_count} open positions from Binance Futures.")
                else:
                    logger.info("No open positions found on Binance.")
                    
            else:
                # Spot Position Sync (Not implemented for now as we focus on Futures)
                pass
                
        except Exception as e:
            logger.error(f"Failed to sync positions: {e}")
