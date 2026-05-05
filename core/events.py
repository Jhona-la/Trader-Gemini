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
import uuid
try:
    import orjson
    def json_dumps(obj): return orjson.dumps(obj).decode('utf-8')
    def json_loads(obj): return orjson.loads(obj)
except ImportError:
    import json
    def json_dumps(obj): return json.dumps(obj, default=str)
    def json_loads(obj): return json.loads(obj)

@dataclass(frozen=True, slots=True, kw_only=True)
class Event:
    """
    Base class for all events.
    Events are immutable after creation to prevent race conditions.
    __slots__ optimization reduces memory footprint by ~40%.
    """
    timestamp_ns: int = field(default_factory=time.time_ns)

    def to_json(self) -> str:
        """Fast serialization for IPC/Logging"""
        # asdict is slow, manual dict creation is faster but verbose
        # We use dataclasses.asdict for safety but orjson makes it fast
        from dataclasses import asdict
        return json_dumps(asdict(self))
        
    @classmethod
    def from_json(cls, json_str: str):
        """Fast deserialization"""
        data = json_loads(json_str)
        # Handle specific field conversions if necessary (e.g. datetime)
        # For now, simplistic implementation
        return cls(**data)


@dataclass(frozen=True, slots=True, kw_only=True)
class MarketEvent(Event):
    """
    Handles the event of receiving a new market update with corresponding bars.
    Now carries metadata to avoid O(N) lookups in Engine.
    """
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    symbol: Optional[str] = None
    close_price: Optional[float] = None
    high_price: Optional[float] = None # Added for High-Fidelity Backtest
    low_price: Optional[float] = None  # Added for High-Fidelity Backtest
    order_flow: Optional[Dict[str, Any]] = None # Added for Phalanx-Omega
    health_metrics: Optional[Dict[str, Any]] = None # Added for Data Integrity Hardening
    
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
    ml_confidence: Optional[float] = None # Added for ML-based scaling
    predicted_magnitude: Optional[float] = None # Predicted move %
    predicted_duration: Optional[int] = None # Hold duration
    setup_type: Optional[str] = None # NEW: Granular setup (e.g. RSI_OVERSOLD, BREAKOUT)
    
    # 🧠 META-COORDINATOR (Trade Intent Metadata)
    risk_score: float = 0.5            # 0.0 (Safe) to 1.0 (Risky)
    liquidity_score: float = 0.5       # 0.0 (Illiquid) to 1.0 (Deep)
    fee_tolerance: float = 0.5         # 0.0 (Strict limits only) to 1.0 (Market orders fine)
    regime_compatibility: float = 0.5  # 0.0 (Contra-trend) to 1.0 (Trend aligned)
    
    strategy_version: Optional[str] = "1.0.0" # Versioning for evolutionary tracking
    metadata: Optional[Dict[str, Any]] = None # Flexible metadata container
    
    # AEGIS-ULTRA Phase 17: Telemetry
    trade_id: Optional[str] = None # UUID for forensic tracing
    
    # 🏎️ [EXCELSIOR-TITAN] Phase III: CPU Cache Alignment
    _pad1: Optional[int] = field(default=None, repr=False)
    _pad2: Optional[int] = field(default=None, repr=False)
    
    # 🧬 [Phase 19] Shadow Mode
    is_shadow: bool = False # If True, ExecutionHandler must IGNORE this for real trading
    
    # ⏱️ Horizon Specialization (Strict enforcement)
    horizon: str  # "SCALPING" or "SWING" (No default!)
    
    # ⚡ Nano-Speeds QoS Priority (0 = Critical/Scalping, 1 = Normal/Swing, 2 = Background)
    priority: int = 1
    
    type: EventType = field(default=EventType.SIGNAL, init=False)

    def __post_init__(self):
        """Validate datetime is UTC-aware and ensure trade_id"""
        try:
            ensure_utc_aware(self.datetime)
        except ValueError as e:
            raise ValueError(f"SignalEvent validation failed: {e}")
        
        # Auto-generate trade_id if missing (using object.__setattr__ because frozen=True)
        if not self.trade_id:
            # Prefix based on horizon to prevent cross-contamination
            prefix = "SCL" if getattr(self, "horizon", "SCALPING") == "SCALPING" else "SWG"
            short_id = f"[{prefix}]-TRD-{str(uuid.uuid4())[:6].upper()}"
            object.__setattr__(self, 'trade_id', short_id)


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
    ml_confidence: Optional[float] = None # Added for ML-based scaling
    predicted_magnitude: Optional[float] = None # Predicted move %
    predicted_duration: Optional[int] = None # Hold duration
    setup_type: Optional[str] = None # NEW: Granular setup propagation
    exit_reason: Optional[str] = None # NEW: Specific reason for exit (e.g., "TIME_STOP_ZOMBIE", "TAKE_PROFIT")
    strategy_version: Optional[str] = "1.0.0"
    metadata: Optional[Dict[str, Any]] = None # Flexible metadata container (Chase count, etc.)
    
    # AEGIS-ULTRA Phase 17: Telemetry
    trade_id: Optional[str] = None # UUID for forensic tracing
    leverage: float = 1.0 # Phase 15: Precise margin tracking
    
    # 🧬 [Phase 19] Shadow Mode
    is_shadow: bool = False # If True, Executor MUST DROP this order
    
    # ⏱️ Horizon Specialization (Strict enforcement)
    horizon: str  # "SCALPING" or "SWING" (No default!)
    
    # ⚡ Nano-Speeds QoS Priority
    priority: int = 1
    
    # 🔴 FORENSIC FIX #0: Exit/Close routing fields (were MISSING → all exits failed with TypeError)
    # QUÉ: Flags que indican si esta orden cierra una posición existente.
    # POR QUÉ: Sin estos campos, OrderEvent(..., is_exit=True) lanza TypeError
    #   en Python frozen dataclass con slots=True. El except en generate_order()
    #   capturaba el error silenciosamente y retornaba None → EXITS MUERTOS.
    # PARA QUÉ: Permitir que RiskManager genere órdenes de cierre que:
    #   1) BinanceExecutor use para skip protective orders en exits
    #   2) OrderManager use para chase/fallback behavior de exits vs entries
    # CUÁNDO: Siempre que generate_order() o _generate_exit_order() cree una orden.
    # DÓNDE: core/events.py → OrderEvent
    # QUIÉN: Risk Manager, BinanceExecutor, OrderManager
    is_exit: bool = False   # True if this order closes/reduces a position
    is_close: bool = False  # True if this is a full position close
    
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
    setup_type: Optional[str] = None    # NEW: Forensic attribution
    exit_reason: Optional[str] = None   # NEW: Reason for exit
    order_type: Optional[OrderType] = None # NEW: Order type used
    strategy_version: Optional[str] = "1.0.0"
    sl_pct: Optional[float] = None      # Protective stop loss %
    tp_pct: Optional[float] = None      # Protective take profit %
    
    # 🔍 FORENSIC AUDITING FOR MICRO-ACCOUNTS
    gross_pnl: Optional[float] = None
    net_pnl: Optional[float] = None
    slippage_pct: Optional[float] = None
    fees_paid: Optional[float] = None
    duration_seconds: Optional[int] = None
    
    # Phase 31: Partial Fill Handling
    is_closed: bool = True              # TRUE if fully filled or cancelled, FALSE if partial
    
    # AEGIS-ULTRA Phase 17: Telemetry
    trade_id: Optional[str] = None # UUID for forensic tracing
    leverage: float = 1.0 # Phase 15: Precise margin tracking
    
    # ⏱️ Horizon Specialization
    horizon: str = "SCALPING" # "SCALPING" or "SWING"
    
    ml_confidence: Optional[float] = None # Added for ML-based scaling
    predicted_magnitude: Optional[float] = None # Predicted move %
    predicted_duration: Optional[int] = None # Hold duration
    
    metadata: Optional[Dict[str, Any]] = None # Phase 31 Fix: Carry metadata from Order
    
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
    
    # ⏱️ Horizon Specialization
    horizon: str = "SCALPING" # "SCALPING" or "SWING"
    
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
