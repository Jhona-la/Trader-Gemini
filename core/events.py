"""
Core Events Module - Immutable event types for the trading system.
All events are frozen dataclasses to prevent race conditions.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from utils.time_helpers import ensure_utc_aware
from core.enums import EventType, SignalType, OrderSide, OrderType


import time

@dataclass(frozen=True, slots=True, kw_only=True)
class Event:
    """
    Base class for all events.
    Events are immutable after creation to prevent race conditions.
    __slots__ optimization reduces memory footprint by ~40%.
    """
    timestamp_ns: int = field(default_factory=time.time_ns)


@dataclass(frozen=True, slots=True, kw_only=True)
class MarketEvent(Event):
    """
    Handles the event of receiving a new market update with corresponding bars.
    Now carries metadata to avoid O(N) lookups in Engine.
    """
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    symbol: Optional[str] = None
    close_price: Optional[float] = None
    order_flow: Optional[Dict[str, Any]] = None # Added for Phalanx-Omega
    
    # 🏎️ [EXCELSIOR-TITAN] Phase III: CPU Cache Alignment
    # Force object size closer to 64-byte cache line to reduce false sharing
    _pad1: Optional[int] = field(default=None, repr=False)
    _pad2: Optional[int] = field(default=None, repr=False)
    
    type: EventType = field(default=EventType.MARKET, init=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class SignalEvent(Event):
    """
    Handles the event of sending a Signal from a Strategy object.
    This is received by a Portfolio object and acted upon.
    """
    strategy_id: str
    symbol: str
    datetime: datetime
    signal_type: SignalType  # ✅ FIXED: Use enum instead of str
    strength: float = 1.0
    atr: Optional[float] = None
    
    # Optional metadata for risk manager
    tp_pct: Optional[float] = None
    sl_pct: Optional[float] = None
    current_price: Optional[float] = None
    leverage: Optional[int] = None  # Added for strategies to specify leverage
    ttl: Optional[int] = None      # Phase 9.2: Adaptive TTL (seconds)
    metadata: Optional[Dict[str, Any]] = None # Flexible metadata container
    
    # AEGIS-ULTRA Phase 17: Telemetry
    trade_id: Optional[str] = None # UUID for forensic tracing
    
    # 🏎️ [EXCELSIOR-TITAN] Phase III: CPU Cache Alignment
    _pad1: Optional[int] = field(default=None, repr=False)
    _pad2: Optional[int] = field(default=None, repr=False)
    
    # 🧬 [Phase 19] Shadow Mode
    is_shadow: bool = False # If True, ExecutionHandler must IGNORE this for real trading
    
    type: EventType = field(default=EventType.SIGNAL, init=False)

    def __post_init__(self):
        """Validate datetime is UTC-aware"""
        try:
            ensure_utc_aware(self.datetime)
        except ValueError as e:
            raise ValueError(f"SignalEvent validation failed: {e}")


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderEvent(Event):
    """
    Handles the event of sending an Order to an execution system.
    """
    symbol: str
    order_type: OrderType  # ✅ FIXED: Use enum instead of str
    quantity: float
    direction: OrderSide   # ✅ FIXED: Use enum instead of str
    strategy_id: Optional[str] = None
    
    # Optional order parameters
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    sl_pct: Optional[float] = None  # NEW: Protective stop loss %
    tp_pct: Optional[float] = None  # NEW: Protective take profit %
    ttl: Optional[int] = None      # Phase 9.2: Adaptive TTL (seconds)
    metadata: Optional[Dict[str, Any]] = None # Flexible metadata container (Chase count, etc.)
    
    # AEGIS-ULTRA Phase 17: Telemetry
    trade_id: Optional[str] = None # UUID for forensic tracing
    
    # 🧬 [Phase 19] Shadow Mode
    is_shadow: bool = False # If True, Executor MUST DROP this order
    
    type: EventType = field(default=EventType.ORDER, init=False)

    def print_order(self):
        """Debug print for order details"""
        print(
            f"Order: Symbol={self.symbol}, Type={self.order_type.name}, "
            f"Quantity={self.quantity:.6f}, Direction={self.direction.name}"
        )
    
    def __str__(self):
        return (
            f"OrderEvent({self.direction.name} {self.quantity:.6f} {self.symbol} "
            f"@ {self.order_type.name})"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class FillEvent(Event):
    """
    Encapsulates the notion of a Filled Order.
    """
    timeindex: datetime
    symbol: str
    exchange: str
    quantity: float
    direction: OrderSide  # ✅ FIXED: Use enum instead of str
    fill_cost: float
    commission: Optional[float] = None
    strategy_id: Optional[str] = None
    
    # Additional fill info
    fill_price: Optional[float] = None  # Actual fill price
    order_id: Optional[str] = None      # Exchange order ID
    sl_pct: Optional[float] = None      # Protective stop loss %
    tp_pct: Optional[float] = None      # Protective take profit %
    
    # Phase 31: Partial Fill Handling
    is_closed: bool = True              # TRUE if fully filled or cancelled, FALSE if partial
    
    # AEGIS-ULTRA Phase 17: Telemetry
    trade_id: Optional[str] = None # UUID for forensic tracing
    
    type: EventType = field(default=EventType.FILL, init=False)
    
    def __post_init__(self):
        """Validate that timeindex is UTC-aware"""
        try:
            ensure_utc_aware(self.timeindex)
        except ValueError as e:
            raise ValueError(f"FillEvent validation failed: {e}")  # ✅ FIXED typo
    
    def __str__(self):
        return (
            f"FillEvent({self.direction.name} {self.quantity:.6f} {self.symbol} "
            f"@ ${self.fill_cost:.2f})"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TradeAuditEvent(Event):
    """
    Audit event for tracking trade decisions and outcomes.
    Used by pattern strategy for performance analysis.
    """
    strategy_id: str
    symbol: str
    timestamp: datetime
    action: str  # "SIGNAL", "ENTRY", "EXIT", "SKIP"
    reason: str  # Human-readable reason
    price: Optional[float] = None
    pnl: Optional[float] = None
    details: Optional[Dict[str, Any]] = None  # ✅ IMPROVED: Dict instead of str
    
    # ✅ FIXED: Use dedicated AUDIT type
    type: EventType = field(default=EventType.AUDIT, init=False)
    
    def __post_init__(self):
        """Validate timestamp is UTC-aware"""
        try:
            ensure_utc_aware(self.timestamp)
        except ValueError as e:
            raise ValueError(f"TradeAuditEvent validation failed: {e}")
    
    def __str__(self):
        pnl_str = f" PnL=${self.pnl:.2f}" if self.pnl else ""
        return f"Audit({self.action} {self.symbol}{pnl_str}: {self.reason})"
