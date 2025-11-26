class Event:
    """
    Base class for all events.
    """
    pass

class MarketEvent(Event):
    """
    Handles the event of receiving a new market update with corresponding bars.
    """
    def __init__(self):
        self.type = 'MARKET'

class SignalEvent(Event):
    """
    Handles the event of sending a Signal from a Strategy object.
    This is received by a Portfolio object and acted upon.
    """
    def __init__(self, strategy_id, symbol, datetime, signal_type, strength=1.0, atr=None):
        """
        Parameters:
        strategy_id - The unique identifier for the strategy that generated the signal.
        symbol - The ticker symbol, e.g. 'BTC/USDT'.
        datetime - The timestamp at which the signal was generated.
        signal_type - 'LONG' or 'SHORT'.
        strength - Strength of the signal (0.0 to 1.0) for sizing.
        atr - Average True Range for volatility sizing.
        """
        self.type = 'SIGNAL'
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.datetime = datetime
        self.signal_type = signal_type
        self.strength = strength
        self.atr = atr

class OrderEvent(Event):
    """
    Handles the event of sending an Order to an execution system.
    The order contains a symbol (e.g. GOOG), a type (market or limit),
    quantity and a direction.
    """
    def __init__(self, symbol, order_type, quantity, direction, strategy_id=None):
        """
        Parameters:
        symbol - The instrument to trade.
        order_type - 'MKT' or 'LMT' for Market or Limit.
        quantity - Non-negative integer for quantity.
        direction - 'BUY' or 'SELL'.
        strategy_id - ID of the strategy that generated the order.
        """
        self.type = 'ORDER'
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction
        self.strategy_id = strategy_id

    def print_order(self):
        print(f"Order: Symbol={self.symbol}, Type={self.order_type}, Quantity={self.quantity}, Direction={self.direction}")

class FillEvent(Event):
    """
    Encapsulates the notion of a Filled Order, as returned
    from a brokerage. Stores the quantity of an instrument
    actually filled and at what price. In addition, stores
    the commission of the trade from the brokerage.
    """
    def __init__(self, timeindex, symbol, exchange, quantity, 
                 direction, fill_cost, commission=None, strategy_id=None):
        """
        Parameters:
        timeindex - The bar-resolution when the order was filled.
        symbol - The instrument which was filled.
        exchange - The exchange where the order was filled.
        quantity - The filled quantity.
        direction - The direction of fill ('BUY' or 'SELL').
        fill_cost - The holdings value in dollars.
        commission - An optional commission sent from IB/Binance.
        strategy_id - ID of the strategy that generated the trade.
        """
        self.type = 'FILL'
        self.timeindex = timeindex
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = quantity
        self.direction = direction
        self.fill_cost = fill_cost
        self.commission = commission
        self.strategy_id = strategy_id
