from enum import Enum, auto

class EventType(str, Enum):
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"
    AUDIT = "AUDIT"  # For TradeAuditEvent - distinguishes from SIGNAL

class SignalType(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"
    REVERSE = "REVERSE"  # Strategic flipping
    NEUTRAL = "NEUTRAL"

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"

class TimeFrame(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

class TradingMode(str, Enum):
    SPOT = "SPOT"
    FUTURES = "FUTURES"

class MarketRegime(str, Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"

class TradeStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    REVERSING = "REVERSING"      # In transition
    FLIP_PENDING = "FLIP_PENDING" # Sequential lock

class ReversalType(str, Enum):
    TREND_REVERSAL = "TREND_REVERSAL"
    BREAKOUT_REVERSAL = "BREAKOUT_REVERSAL"
    VOLATILITY_FLIP = "VOLATILITY_FLIP"

class TradeDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
