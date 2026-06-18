"""
AITS Phase 6: Adaptive Execution & Smart Order Routing (Capa 5)
Smart Order Router

Decides HOW to execute an order based on market microstructure signals
from Phases 1-5. Implements three institutional execution algorithms:

1. Limit Maker (cheapest, 0.02% fee on Binance Futures)
2. TWAP (Time-Weighted Average Price) — splits large orders across time
3. Iceberg (fragmented hidden orders) — hides intent from other algos

The router selects the optimal algorithm based on:
- Spread tightness (from Phase 1 Order Book Collector)
- Volatility Burst flag (from Phase 2 Feature Warehouse)
- Prediction confidence (from Phase 4 PyTorch models)
- Available liquidity depth (from Phase 1 bid/ask volumes)
"""

import asyncio
import logging
import time
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ─── Data Structures ────────────────────────────────────────────────

class OrderType(Enum):
    LIMIT_MAKER = "LIMIT_MAKER"   # Post-only, 0.02% fee
    MARKET      = "MARKET"        # Immediate fill, 0.04% fee
    TWAP        = "TWAP"          # Time-sliced limit
    ICEBERG     = "ICEBERG"       # Fragmented hidden


class Side(Enum):
    BUY  = "BUY"
    SELL = "SELL"


@dataclass
class ExecutionOrder:
    """Represents a single child order produced by the router."""
    symbol: str
    side: Side
    order_type: OrderType
    price: float
    quantity: float
    delay_ms: int = 0            # For TWAP scheduling
    parent_id: str = ""          # Links back to the original intent
    created_at: float = field(default_factory=time.time)


@dataclass
class MarketContext:
    """Snapshot of micro-structure state fed by earlier AITS layers."""
    symbol: str
    best_bid: float
    best_ask: float
    spread: float               # Phase 1
    bid_volume_top5: float      # Phase 1
    ask_volume_top5: float      # Phase 1
    volatility_burst: bool      # Phase 2
    prediction_confidence: float # Phase 4 (0.0 – 1.0)
    predicted_direction: str    # "UP" | "DOWN"
    microprice: float = 0.0     # Phase 28: C_Orderbook Microprice
    ofi: float = 0.0            # Phase 28: Order Flow Imbalance


# ─── Core Router ────────────────────────────────────────────────────

class SmartOrderRouter:
    """
    Institutional-grade execution router.

    Decision matrix:
    ┌────────────────────┬──────────────┬───────────────────────┐
    │ Condition          │ Urgency      │ Algorithm Selected    │
    ├────────────────────┼──────────────┼───────────────────────┤
    │ Volatility Burst   │ EXTREME      │ MARKET (speed > cost) │
    │ Tight Spread       │ LOW          │ LIMIT_MAKER (cheapest)│
    │ Large Order        │ MEDIUM       │ ICEBERG (hide intent) │
    │ Medium Confidence  │ MEDIUM       │ TWAP (average price)  │
    └────────────────────┴──────────────┴───────────────────────┘
    """

    # Configurable thresholds
    SPREAD_TIGHT_BPS   = 3.0      # ≤3 bps → spread is tight enough for Limit
    CONFIDENCE_HIGH    = 0.75     # ≥75% model confidence → aggressive entry OK
    ICEBERG_THRESHOLD  = 0.10    # Order > 10% of top-5 volume → use Iceberg
    TWAP_SLICES        = 5       # Number of time slices for TWAP
    TWAP_INTERVAL_MS   = 2000    # Milliseconds between TWAP child orders

    def __init__(self):
        self.execution_log: List[ExecutionOrder] = []

    # ── Public API ──────────────────────────────────────────────────

    def route(self, ctx: MarketContext, quantity: float) -> List[ExecutionOrder]:
        """
        Main entry point. Returns a list of ExecutionOrder objects
        that the downstream executor should send to the exchange.
        """
        algo = self._select_algorithm(ctx, quantity)
        logging.info(
            f"[Router] {ctx.symbol} | Algo={algo.value} | "
            f"Spread={ctx.spread:.4f} | Conf={ctx.prediction_confidence:.2f} | "
            f"VolBurst={ctx.volatility_burst}"
        )

        side = Side.BUY if ctx.predicted_direction == "UP" else Side.SELL

        if algo == OrderType.MARKET:
            orders = self._build_market_order(ctx, side, quantity)
        elif algo == OrderType.LIMIT_MAKER:
            orders = self._build_limit_maker(ctx, side, quantity)
        elif algo == OrderType.ICEBERG:
            orders = self._build_iceberg(ctx, side, quantity)
        elif algo == OrderType.TWAP:
            orders = self._build_twap(ctx, side, quantity)
        else:
            orders = self._build_market_order(ctx, side, quantity)

        self.execution_log.extend(orders)
        return orders

    # ── Algorithm Selection ─────────────────────────────────────────

    def _select_algorithm(self, ctx: MarketContext, quantity: float) -> OrderType:
        # Rule 1: Extreme urgency during volatility bursts → MARKET
        if ctx.volatility_burst:
            return OrderType.MARKET

        spread_bps = (ctx.spread / ctx.best_bid) * 10_000 if ctx.best_bid else 999

        # Rule 2: Large order relative to visible liquidity → ICEBERG
        relevant_vol = ctx.bid_volume_top5 if ctx.predicted_direction == "UP" else ctx.ask_volume_top5
        if relevant_vol > 0 and (quantity / relevant_vol) > self.ICEBERG_THRESHOLD:
            return OrderType.ICEBERG

        # Rule 3: Tight spread + high confidence → LIMIT_MAKER (cheapest)
        if spread_bps <= self.SPREAD_TIGHT_BPS and ctx.prediction_confidence >= self.CONFIDENCE_HIGH:
            return OrderType.LIMIT_MAKER

        # Rule 4: Default → TWAP for patience
        return OrderType.TWAP

    # ── Order Builders ──────────────────────────────────────────────

    def _build_market_order(self, ctx: MarketContext, side: Side, qty: float) -> List[ExecutionOrder]:
        price = ctx.best_ask if side == Side.BUY else ctx.best_bid
        return [ExecutionOrder(
            symbol=ctx.symbol, side=side, order_type=OrderType.MARKET,
            price=price, quantity=qty, parent_id="MKT"
        )]

    def _build_limit_maker(self, ctx: MarketContext, side: Side, qty: float) -> List[ExecutionOrder]:
        # Microstructure-aware pricing (AITS P6 + Cython Orderbook)
        # Place limit INSIDE the spread to guarantee maker rebate,
        # but adjust aggressively or passively based on Order Flow Imbalance (OFI).
        
        if side == Side.BUY:
            if ctx.ofi < -2.0:
                # Strong sell pressure: Be passive, don't catch the falling knife too early
                price = ctx.best_bid - (ctx.spread * 0.2)
            elif ctx.microprice > ctx.best_bid + (ctx.spread * 0.6):
                # Imbalance is heavily towards ask, we must be aggressive
                price = ctx.best_bid + (ctx.spread * 0.2)
            else:
                price = ctx.best_bid + (ctx.spread * 0.05)
        else:
            if ctx.ofi > 2.0:
                # Strong buy pressure: Be passive, don't short the rocket too early
                price = ctx.best_ask + (ctx.spread * 0.2)
            elif ctx.microprice < ctx.best_ask - (ctx.spread * 0.6):
                # Imbalance is heavily towards bid, we must be aggressive
                price = ctx.best_ask - (ctx.spread * 0.2)
            else:
                price = ctx.best_ask - (ctx.spread * 0.05)
                
        return [ExecutionOrder(
            symbol=ctx.symbol, side=side, order_type=OrderType.LIMIT_MAKER,
            price=round(price, 5), quantity=qty, parent_id="LMT"
        )]

    def _build_iceberg(self, ctx: MarketContext, side: Side, qty: float) -> List[ExecutionOrder]:
        # Fragment into ~5 visible slices
        num_slices = 5
        slice_qty = qty / num_slices
        orders = []
        base_price = ctx.best_bid if side == Side.BUY else ctx.best_ask

        for i in range(num_slices):
            # Slight price ladder to catch different levels
            offset = i * (ctx.spread * 0.05)
            price = (base_price + offset) if side == Side.BUY else (base_price - offset)
            orders.append(ExecutionOrder(
                symbol=ctx.symbol, side=side, order_type=OrderType.ICEBERG,
                price=round(price, 2), quantity=round(slice_qty, 6),
                delay_ms=i * 500, parent_id=f"ICE_{i}"
            ))
        return orders

    def _build_twap(self, ctx: MarketContext, side: Side, qty: float) -> List[ExecutionOrder]:
        slice_qty = qty / self.TWAP_SLICES
        orders = []
        base_price = ctx.best_bid if side == Side.BUY else ctx.best_ask

        for i in range(self.TWAP_SLICES):
            orders.append(ExecutionOrder(
                symbol=ctx.symbol, side=side, order_type=OrderType.TWAP,
                price=round(base_price, 2), quantity=round(slice_qty, 6),
                delay_ms=i * self.TWAP_INTERVAL_MS, parent_id=f"TWAP_{i}"
            ))
        return orders


# ─── Demo / Self-Test ───────────────────────────────────────────────

if __name__ == "__main__":
    router = SmartOrderRouter()

    # Scenario A: Tight spread, high confidence → expect LIMIT_MAKER
    ctx_a = MarketContext(
        symbol="BTCUSDT", best_bid=67000.0, best_ask=67001.5,
        spread=1.5, bid_volume_top5=12.0, ask_volume_top5=11.5,
        volatility_burst=False, prediction_confidence=0.82,
        predicted_direction="UP", microprice=67001.0, ofi=3.5
    )
    orders_a = router.route(ctx_a, quantity=0.002)
    for o in orders_a:
        logging.info(f"  → {o.order_type.value} {o.side.value} {o.quantity} @ {o.price}")

    # Scenario B: Volatility burst → expect MARKET
    ctx_b = MarketContext(
        symbol="ETHUSDT", best_bid=3800.0, best_ask=3803.0,
        spread=3.0, bid_volume_top5=50.0, ask_volume_top5=48.0,
        volatility_burst=True, prediction_confidence=0.60,
        predicted_direction="DOWN"
    )
    orders_b = router.route(ctx_b, quantity=0.05)
    for o in orders_b:
        logging.info(f"  → {o.order_type.value} {o.side.value} {o.quantity} @ {o.price}")

    # Scenario C: Large order relative to book → expect ICEBERG
    ctx_c = MarketContext(
        symbol="SOLUSDT", best_bid=170.0, best_ask=170.10,
        spread=0.10, bid_volume_top5=5.0, ask_volume_top5=4.8,
        volatility_burst=False, prediction_confidence=0.70,
        predicted_direction="UP"
    )
    orders_c = router.route(ctx_c, quantity=2.0)  # 2 SOL vs 5 top-5 volume = 40%
    for o in orders_c:
        logging.info(f"  → {o.order_type.value} {o.side.value} {o.quantity} @ {o.price}")

    logging.info(f"\n✅ Total orders generated: {len(router.execution_log)}")
