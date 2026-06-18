"""
AITS Phase 8: Integration Bridge
Centralized Configuration

Feature flags and connection parameters for the entire AITS stack.
Each layer can be enabled/disabled independently, allowing the production
bot to run in any combination from "fully legacy" to "fully institutional".
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class AITSConfig:
    """
    Single source of truth for the AITS infrastructure.
    In production, these values would be loaded from environment variables.
    """

    # ─── Infrastructure Endpoints ────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379"
    TIMESCALE_DSN: str = "postgresql://aits_user:aits_pass@localhost:5432/aits_market_data"
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_AUTH: tuple = ("neo4j", "aits_neo4j_pass")

    # ─── Feature Flags (Layer Toggles) ───────────────────────────
    ENABLE_ORDERBOOK_COLLECTOR: bool = True   # Phase 1
    ENABLE_STREAM_COLLECTOR: bool = True      # Phase 2
    ENABLE_FEATURE_WAREHOUSE: bool = True     # Phase 2
    ENABLE_GRAPH_ENGINE: bool = True          # Phase 3
    ENABLE_DEEP_LEARNING: bool = True         # Phase 4
    ENABLE_RL_AGENT: bool = True              # Phase 5
    ENABLE_SMART_ROUTER: bool = True          # Phase 6
    ENABLE_SOVEREIGN_SHIELD: bool = True      # Phase 7

    # ─── Sovereign Shield Parameters ─────────────────────────────
    SHIELD_MAX_SESSION_DRAWDOWN: float = 0.02
    SHIELD_MAX_LOSS_PER_TRADE: float = 0.005
    SHIELD_MAX_OPEN_POSITIONS: int = 3
    SHIELD_MAX_TRADES_PER_DAY: int = 20
    SHIELD_MIN_MODEL_CONFIDENCE: float = 0.55
    SHIELD_BLOCK_ON_VOLATILITY_BURST: bool = True
    SHIELD_MIN_BTC_CORRELATION: float = 0.30
    SHIELD_HALT_DURATION_SECONDS: int = 900

    # ─── Smart Router Parameters ─────────────────────────────────
    ROUTER_SPREAD_TIGHT_BPS: float = 3.0
    ROUTER_CONFIDENCE_HIGH: float = 0.75
    ROUTER_ICEBERG_THRESHOLD: float = 0.10
    ROUTER_TWAP_SLICES: int = 5
    ROUTER_TWAP_INTERVAL_MS: int = 2000

    # ─── ML & RL Parameters ──────────────────────────────────────
    DEEPLOB_SEQ_LENGTH: int = 50
    DEEPLOB_FEATURES: int = 40
    TRANSFORMER_D_MODEL: int = 60
    TRANSFORMER_HEADS: int = 3
    TRANSFORMER_LAYERS: int = 2
    RL_INITIAL_BALANCE: float = 13.0
    RL_MAX_STEPS: int = 1000

    # ─── Feature Warehouse Parameters ────────────────────────────
    WAREHOUSE_LIQUIDATION_WINDOW_SEC: float = 60.0
    WAREHOUSE_VOLATILITY_BURST_THRESHOLD_USD: float = 1_000_000.0

    # ─── Redis Stream Keys ───────────────────────────────────────
    REDIS_STREAM_RAW: str = "aits:raw:events"
    REDIS_STREAM_FEATURES: str = "aits:features:computed"
    REDIS_STREAM_SIGNALS: str = "aits:signals:enriched"

    # ─── Trading Symbols ─────────────────────────────────────────
    SYMBOLS: List[str] = field(default_factory=lambda: [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"
    ])

    # ─── Capital ─────────────────────────────────────────────────
    TOTAL_CAPITAL_USD: float = 13.0


# Singleton instance
AITS_CFG = AITSConfig()
