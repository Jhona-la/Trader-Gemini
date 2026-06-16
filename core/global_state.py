"""
GlobalMarketState — Single Source of Truth (SSOT)
═══════════════════════════════════════════════════════════════
QUÉ: Fotografía inmutable del estado completo del mundo por tick.
POR QUÉ: Elimina la fragmentación donde cada módulo mantenía su propia
  "realidad" (symbol_state_matrix, portfolio.positions, risk_state).
PARA QUÉ: Todos los módulos leen de un único lugar → coherencia garantizada.
CÓMO: El GlobalClock congela el timestamp. El Engine actualiza el estado.
  Todos los módulos consumen el snapshot congelado.
CUÁNDO: Se actualiza en cada MarketEvent dentro del burst loop de engine.py.
DÓNDE: core/global_state.py (este archivo)
QUIÉN: Engine (escritor) → Strategies, RiskManager, MetaCoordinator (lectores)
═══════════════════════════════════════════════════════════════
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from core.state_vector import SymbolStateVector
from core.structs import PositionState
from core.clock import global_clock
from core.dark_alpha_worker import dark_alpha_worker
from core.mempool_worker import mempool_worker

logger = logging.getLogger(__name__)


@dataclass
class PortfolioSnapshot:
    """Read-only view of portfolio state for the SSOT."""
    total_equity: float = 0.0
    margin_used: float = 0.0
    margin_available: float = 0.0
    open_position_count: int = 0
    unrealized_pnl: float = 0.0
    realized_pnl_today: float = 0.0


@dataclass
class RiskSnapshot:
    """Read-only view of risk state for the SSOT."""
    current_drawdown_pct: float = 0.0
    kill_switch_armed: bool = False
    global_hazard_rate: float = 0.0  # 0.0 (Safe) to 1.0 (Catastrophic)
    consecutive_losses: int = 0


class GlobalMarketState:
    """
    Single Source of Truth (SSOT) — CTOS Core
    
    Absorbe la funcionalidad de SymbolStateMatrix (legacy) y agrega:
    - Features canónicas centralizadas
    - Snapshot de Portfolio y Risk
    - Timestamp congelado por el GlobalClock
    
    BACKWARD COMPATIBILITY:
    symbol_state_matrix.py ahora es un PROXY que delega aquí.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalMarketState, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self.timestamp_ns: int = 0
        
        # 1. Market Topography (absorbs SymbolStateMatrix)
        self.symbol_states: Dict[str, SymbolStateVector] = {}
        
        # 2. Legacy dict-based state (backward-compatible with SymbolStateMatrix consumers)
        self._legacy_matrix: Dict[str, Dict[str, Any]] = {}
        
        # 3. Canonical Features (populated by CanonicalFeatureEngine)
        self.canonical_features: Dict[str, Dict[str, float]] = {}
        
        # 4. Macro & System State
        self.market_regime: str = "UNKNOWN"
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
        self.btc_velocity: float = 0.0  # 🔮 Phase 5: Multi-Coin Oracle velocity
        self.cross_exchange_metrics: Dict[str, Dict[str, float]] = {}  # 🌐 Multi-Source Intelligence
        self.dark_alpha_pressure: float = 0.0  # 🌑 DEX Cascade Pressure
        self.rbf_panic_score: float = 0.0  # 🚨 Mempool Urgency
        
        # 5. Portfolio & Risk Projections
        self.portfolio: PortfolioSnapshot = PortfolioSnapshot()
        self.risk: RiskSnapshot = RiskSnapshot()
        
        # 6. Active Positions (PositionState from structs.py)
        self.active_positions: Dict[str, PositionState] = {}
        
        # Connect to GlobalClock
        global_clock.subscribe(self._on_tick)
        
        logger.info("🌐 [SSOT] GlobalMarketState initialized as Single Source of Truth.")

    # ════════════════════════════════════════════════════════════════
    # CLOCK INTEGRATION
    # ════════════════════════════════════════════════════════════════
    
    def _on_tick(self, timestamp_ns: int):
        """Callback del Clock para firmar/congelar el estado actual."""
        self.timestamp_ns = timestamp_ns
        self.dark_alpha_pressure = dark_alpha_worker.get_net_pressure()
        self.rbf_panic_score = mempool_worker.get_panic_score()

    def freeze(self, current_ns: int):
        """Explicit freeze (for use outside clock subscription)."""
        self.timestamp_ns = current_ns

    # ════════════════════════════════════════════════════════════════
    # SYMBOL STATE MANAGEMENT (Absorbs SymbolStateMatrix)
    # ════════════════════════════════════════════════════════════════
    
    def update_from_market_event(self, event) -> None:
        """
        Unified update from MarketEvent.
        Replaces SymbolStateMatrix.update_from_market_event().
        """
        symbol = getattr(event, 'symbol', None)
        if not symbol:
            return
        
        # Update SymbolStateVector (typed, structured)
        if symbol not in self.symbol_states:
            self.symbol_states[symbol] = SymbolStateVector(symbol=symbol)
        
        sv = self.symbol_states[symbol]
        updates = {}
        
        # Extract from order_flow metadata if present
        if hasattr(event, 'order_flow') and event.order_flow:
            of = event.order_flow
            updates['orderflow_imbalance'] = of.get('ofi', 0.0)
            updates['spread_cost_pct'] = of.get('spread_pct', 0.0)
            updates['liquidity_depth'] = of.get('depth', 0.0)
            
            # L2 Orderbook Vectorization
            micro_price = of.get('micro_price', 0.0)
            if micro_price > 0 and hasattr(event, 'close_price') and event.close_price > 0:
                updates['microprice'] = micro_price
                updates['microprice_divergence'] = (micro_price - event.close_price) / event.close_price
            elif micro_price > 0:
                updates['microprice'] = micro_price
        
        # Extract from health_metrics if present
        if hasattr(event, 'health_metrics') and event.health_metrics:
            hm = event.health_metrics
            updates['liquidity_depth'] = hm.get('liquidity', updates.get('liquidity_depth', 0.0))
        
        if updates:
            sv.update_from_dict(updates)
        
        # Update legacy dict (backward-compat for SymbolStateMatrix consumers)
        if symbol not in self._legacy_matrix:
            self._legacy_matrix[symbol] = self._get_default_legacy_state(symbol)
        
        legacy = self._legacy_matrix[symbol]
        if hasattr(event, 'order_flow') and event.order_flow:
            legacy['orderflow_pressure'] = event.order_flow.get('ofi', 0.0)
        if hasattr(event, 'health_metrics') and event.health_metrics:
            legacy['liquidity_score'] = event.health_metrics.get('liquidity', 0.5)
        
        # ═══════════════════════════════════════════════════════════════
        # LOW-LATENCY PHASE: Store OHLCV in legacy dict
        # QUÉ: Almacena close_price del MarketEvent en el state dict.
        # POR QUÉ: engine._get_validated_price() fast-path lee 
        #   global_state.get_state(symbol)['close'] para evitar
        #   el costoso get_latest_bars() (lock + Numba + np.empty).
        # PARA QUÉ: Habilitar reducción de 100-500μs por señal.
        # CUÁNDO: En cada MarketEvent procesado.
        # DÓNDE: core/global_state.py → update_from_market_event()
        # QUIÉN: SRE/DevOps
        # ═══════════════════════════════════════════════════════════════
        if hasattr(event, 'close_price') and event.close_price:
            legacy['close'] = float(event.close_price)
        if hasattr(event, 'open_price') and event.open_price:
            legacy['open'] = float(event.open_price)
        if hasattr(event, 'high_price') and event.high_price:
            legacy['high'] = float(event.high_price)
        if hasattr(event, 'low_price') and event.low_price:
            legacy['low'] = float(event.low_price)
        if hasattr(event, 'volume') and event.volume:
            legacy['volume'] = float(event.volume)
        
        legacy['last_update'] = time.time()
    
    def _get_default_legacy_state(self, symbol: str) -> Dict[str, Any]:
        """Default state for legacy SymbolStateMatrix consumers."""
        return {
            "symbol": symbol,
            "trend_score": 0.0,
            "micro_volatility": 0.0,
            "orderflow_pressure": 0.0,
            "liquidity_score": 0.5,
            "regime_class": "UNKNOWN",
            "funding_bias": 0.0,
            "correlation_cluster": 0,
            "signal_density": 0.0,
            "last_update": time.time()
        }

    # ════════════════════════════════════════════════════════════════
    # LEGACY API (SymbolStateMatrix backward-compatibility)
    # ════════════════════════════════════════════════════════════════
    
    def get_state(self, symbol: str) -> Dict[str, Any]:
        """Legacy API: Returns dict-based state (for meta_arbitrator, etc.)."""
        return self._legacy_matrix.get(symbol, self._get_default_legacy_state(symbol)).copy()

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Legacy API: Returns the entire legacy matrix."""
        return self._legacy_matrix.copy()

    # ════════════════════════════════════════════════════════════════
    # CTOS API (New typed accessors)
    # ════════════════════════════════════════════════════════════════
    
    def get_symbol_vector(self, symbol: str) -> Optional[SymbolStateVector]:
        """Returns the typed SymbolStateVector for a symbol."""
        return self.symbol_states.get(symbol)

    def get_features(self, symbol: str) -> Dict[str, float]:
        """Returns canonical features for a symbol (populated by FeatureEngine)."""
        return self.canonical_features.get(symbol, {})

    def get_open_position(self, symbol: str, horizon: str = None) -> Optional[PositionState]:
        """Returns active position state for a symbol, optionally filtered by horizon."""
        # FASE 7: active_positions keys are v_keys like 'BTC/USDT_SCALPING_LONG'
        for v_key, pos in self.active_positions.items():
            if pos.symbol == symbol:
                if horizon:
                    # Check if the horizon is in the v_key, or if pos has horizon attr
                    if hasattr(pos, 'horizon') and pos.horizon == horizon:
                        return pos
                    if horizon in v_key:
                        return pos
                else:
                    return pos
        return None
    
    def update_symbol_vector(self, symbol: str, data: Dict[str, float]):
        """Updates the typed vector for a specific symbol."""
        if symbol not in self.symbol_states:
            self.symbol_states[symbol] = SymbolStateVector(symbol=symbol)
        self.symbol_states[symbol].update_from_dict(data)
    
    def update_portfolio_snapshot(self, equity: float, margin_used: float,
                                  positions_count: int, unrealized_pnl: float,
                                  realized_pnl: float):
        """Called by Portfolio to sync its state into SSOT."""
        self.portfolio.total_equity = equity
        self.portfolio.margin_used = margin_used
        self.portfolio.margin_available = equity - margin_used
        self.portfolio.open_position_count = positions_count
        self.portfolio.unrealized_pnl = unrealized_pnl
        self.portfolio.realized_pnl_today = realized_pnl
    
    def update_risk_snapshot(self, drawdown_pct: float, kill_switch: bool,
                              hazard_rate: float, consecutive_losses: int):
        """Called by RiskManager to sync its state into SSOT."""
        self.risk.current_drawdown_pct = drawdown_pct
        self.risk.kill_switch_armed = kill_switch
        self.risk.global_hazard_rate = hazard_rate
        self.risk.consecutive_losses = consecutive_losses

    # ════════════════════════════════════════════════════════════════
    # CTOS PHASE 4: SYSTEM SELF-AWARENESS
    # QUÉ: Registry de estrategias y capabilities del sistema.
    # POR QUÉ: Las estrategias no se conocen entre sí.
    # PARA QUÉ: Cada componente puede consultar "¿quién más ve este símbolo?"
    # CÓMO: Dict con strategy_id → {type, capabilities, symbols, horizons}.
    # CUÁNDO: Estrategias se registran al iniciar el engine.
    # DÓNDE: core/global_state.py
    # QUIÉN: Engine (escritor) → Strategies, RiskManager (lectores)
    # ════════════════════════════════════════════════════════════════
    
    def register_strategy(self, strategy_id: str, strategy_type: str,
                          capabilities: list = None, symbols: list = None,
                          horizons: list = None, directions: list = None):
        """Registers a strategy with its capabilities in the SSOT."""
        if not hasattr(self, 'strategy_registry'):
            self.strategy_registry = {}
        self.strategy_registry[strategy_id] = {
            'type': strategy_type,
            'capabilities': capabilities or [],
            'symbols': symbols or [],
            'horizons': horizons or ['SCALPING', 'SWING'],
            'directions': directions or ['LONG', 'SHORT'],
            'last_signal_time': None,
            'total_signals': 0,
            'is_active': True,
        }
        logger.debug(f"🧠 [SSOT] Strategy registered: {strategy_id} ({strategy_type})")

    def get_competing_strategies(self, symbol: str, horizon: str = None) -> list:
        """Returns list of strategies active for a given symbol/horizon."""
        if not hasattr(self, 'strategy_registry'):
            return []
        result = []
        for sid, info in self.strategy_registry.items():
            if not info.get('is_active', True):
                continue
            if symbol in info.get('symbols', []) or not info.get('symbols'):
                if horizon is None or horizon in info.get('horizons', []):
                    result.append({'strategy_id': sid, **info})
        return result

    def get_system_capabilities(self) -> Dict[str, Any]:
        """Returns complete summary of system capabilities."""
        if not hasattr(self, 'strategy_registry'):
            self.strategy_registry = {}
        all_symbols = set()
        all_horizons = set()
        all_directions = set()
        all_types = set()
        for info in self.strategy_registry.values():
            all_symbols.update(info.get('symbols', []))
            all_horizons.update(info.get('horizons', []))
            all_directions.update(info.get('directions', []))
            all_types.add(info.get('type', 'UNKNOWN'))
        return {
            'total_strategies': len(self.strategy_registry),
            'strategy_types': list(all_types),
            'total_symbols': len(all_symbols),
            'symbols': list(all_symbols),
            'horizons': list(all_horizons),
            'directions': list(all_directions),
            'active_positions': len(self.active_positions),
            'portfolio_equity': self.portfolio.total_equity,
        }


# ════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ════════════════════════════════════════════════════════════════
global_state = GlobalMarketState()
