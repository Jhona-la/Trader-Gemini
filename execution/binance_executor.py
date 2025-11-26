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
            self.exchange.enable_demo_trading(True)
            print(f"Binance Executor: Running in {mode_description} mode.")
        elif Config.BINANCE_USE_TESTNET:
            self.exchange.set_sandbox_mode(True)
            print(f"Binance Executor: Running in {mode_description} mode.")
        else:
            print(f"Binance Executor: Running in {mode_description} mode.")
            
        # Set Leverage for Futures
        if Config.BINANCE_USE_FUTURES:
            print(f"Binance Executor: FUTURES MODE ENABLED (Leverage {Config.BINANCE_LEVERAGE}x)")
            # Note: Leverage setting via API might require specific permissions or be done manually on UI first
            # We attempt to set it here for supported pairs
            try:
                # Iterate through active pairs to set leverage
                print(f"Binance Executor: Setting leverage to {Config.BINANCE_LEVERAGE}x for all pairs...")
                for symbol in Config.TRADING_PAIRS:
                    try:
                        # CCXT unified method for setting leverage
                        # market type is 'future' by default due to options
                        self.exchange.set_leverage(Config.BINANCE_LEVERAGE, symbol)
                        # print(f\"  ✅ Leverage set to {Config.BINANCE_LEVERAGE}x for {symbol}\")
                    except Exception as e:
                        # Some pairs might not support it or require margin mode change first
                        print(f"  ⚠️ Failed to set leverage for {symbol}: {e}")
            except Exception as e:
                print(f"Warning: Could not set leverage automatically: {e}")

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
                self.portfolio.release_cash(estimated_cost)
                print(f"  💰 Released ${estimated_cost:.2f} reserved cash")
