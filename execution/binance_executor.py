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
        # Habilitar el modo correspondiente
        if (hasattr(Config, 'BINANCE_USE_DEMO') and Config.BINANCE_USE_DEMO) or Config.BINANCE_USE_TESTNET:
            # MANUAL CONFIGURATION for Futures Demo (Robust way)
            # We enforce these URLs to avoid CCXT missing endpoints like fapiPrivateV2
            # MANUAL CONFIGURATION for Futures Demo (Robust way)
            # We enforce these URLs to avoid CCXT missing endpoints like fapiPrivateV2
            custom_urls = {
                'public': 'https://testnet.binancefuture.com/fapi/v1',
                'private': 'https://testnet.binancefuture.com/fapi/v1',
                'fapiPublic': 'https://testnet.binancefuture.com/fapi/v1',
                'fapiPrivate': 'https://testnet.binancefuture.com/fapi/v1',
                'fapiPrivateV2': 'https://testnet.binancefuture.com/fapi/v2', # Explicitly added
                'fapiData': 'https://testnet.binancefuture.com/fapi/v1',
                'dapiPublic': 'https://testnet.binancefuture.com/dapi/v1',
                'dapiPrivate': 'https://testnet.binancefuture.com/dapi/v1',
                'dapiData': 'https://testnet.binancefuture.com/dapi/v1',
                'sapi': 'https://testnet.binance.vision/api/v3', 
            }
            
            # Set BOTH 'api' and 'test' URLs to ensure CCXT finds them regardless of mode
            self.exchange.urls['api'] = custom_urls
            self.exchange.urls['test'] = custom_urls
            
            # We do NOT call set_sandbox_mode(True) because we manually set the URLs
            # This prevents CCXT from overwriting our custom map with incomplete defaults
            # We do NOT call set_sandbox_mode(True) because we manually set the URLs
            # This prevents CCXT from overwriting our custom map with incomplete defaults
            logger.info(f"Binance Executor: Running in {mode_description} mode (Manual URL Config)")
        else:
            logger.info(f"Binance Executor: Running in {mode_description} mode")
            
        # Set Leverage for Futures
        # NOTE: Leverage endpoint not available on Testnet
        # Configure leverage manually in Binance Demo UI
        if Config.BINANCE_USE_FUTURES:
            logger.info("Binance Executor: FUTURES MODE ENABLED")
            logger.info("  → Using server-side leverage (configure in Binance Demo UI)")

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
            
            # Execute Order
            if order_type == 'mkt':
                order_type = 'market'
                
            order = self.exchange.create_order(symbol, order_type, side, quantity)
            
            # Log Success
            print(f"Order Filled: {order['id']} - {side} {order['filled']} @ {order['average']}")
            
            # Create Fill Event
            fill_price = order.get('average', 0)
            if fill_price is None: 
                fill_price = 0
            
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

    def get_balance(self):
        """
        Fetches the current USDT balance from Binance Futures.
        Uses the correct Futures endpoint: GET /fapi/v2/balance
        Returns float or None if failed.
        """
        try:
            # For Futures, use fapiPrivateV2GetBalance (Futures Account Balance V2)
            # This is the correct endpoint for USDT-M Futures
            response = self.exchange.fapiPrivateV2GetBalance()
            
            # Response is a list of assets
            # Find USDT in the list
            for asset in response:
                if asset['asset'] == 'USDT':
                    balance = float(asset['balance'])
                    available = float(asset['availableBalance'])
                    print(f"💰 Binance Futures Balance: ${balance:.2f} (Available: ${available:.2f})")
                    return balance
            
            print("⚠️  USDT not found in Futures balance response.")
            return None
                
        except Exception as e:
            handle_balance_error(e)
            return None

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
                # Endpoint: GET /fapi/v2/positionRisk
                positions = self.exchange.fapiPrivateV2GetPositionRisk()
                
                synced_count = 0
                for pos in positions:
                    symbol = pos['symbol']
                    amt = float(pos['positionAmt'])
                    entry_price = float(pos['entryPrice'])
                    
                    # Only care about non-zero positions
                    if abs(amt) > 0:
                        # Update local portfolio
                        # Convert symbol format if needed (BTCUSDT -> BTC/USDT)
                        # CCXT usually handles this, but raw endpoint returns 'BTCUSDT'
                        # We need to map it back to our internal format 'BTC/USDT'
                        
                        # Simple mapping attempt
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
