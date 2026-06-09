"""
Event-Driven Trading Engine - Optimized Version
Coordinates data, strategies, risk, and execution with enhanced validation and resource management.
"""

from core.events import MarketEvent, SignalEvent, OrderEvent, FillEvent
from core.enums import SignalType
from core.global_state import global_state  # CTOS: Single Source of Truth
from core.clock import global_clock          # CTOS: Master Clock Synchronization
from core.event_bus import event_bus, EventChannel  # CTOS: Event-Driven Architecture
from core.feedback_processor import feedback_processor  # CTOS: Learning Loop
from core.asset_classifier import AssetClassifier
from core.segment_policy_engine import SegmentPolicyEngine
import os
import dataclasses
import asyncio
import time
import collections
import traceback
from typing import Optional, Any, Union, Dict
from datetime import datetime, timezone
from config import Config
from utils.logger import logger
from utils.os_tuner import OSTuner # Protocol Nadir-Soberano
from utils.time_sync import TimeSynchronizer # Phase 26: Stochastic Purity
from core.system_monitor import SystemMonitor # Phase 27: Disaster Resilience
from utils.latency_monitor import latency_monitor
from core.temporal_supervisor import TemporalSupervisor
from core.gc_tuner import GCTuner
from core.forensics import ForensicRecorder # Phase 20: Forensic Logging
from core.world_awareness import world_awareness
from core.evolution import EvolutionEngine, FitnessCalculator
from core.genotype import Genotype
from core.market_regime import MarketRegimeDetector

# ═══════════════════════════════════════════════════════════════
# LOW-LATENCY PHASE: HOT-PATH IMPORTS (Moved from inline)
# QUÉ: Imports que antes estaban dentro de funciones hot-path.
# POR QUÉ: Cada import inline ejecuta sys.modules lookup + overhead
#   de interpretación (~5-20μs). En hot-path con 32 eventos/burst,
#   esto sumaba 1.6-6.4ms/burst innecesarios.
# PARA QUÉ: Reducir latencia del event loop en ~30%.
# CÓMO: Mover al top-level para que Python los resuelva UNA sola vez.
# CUÁNDO: Al cargar el módulo (import time).
# DÓNDE: core/engine.py (top-level)
# QUIÉN: Arquitecto Senior + SRE/DevOps
# ═══════════════════════════════════════════════════════════════
from utils.quantum_telemetry import QuantumTimer  # Was inline in _process_market_event L627
from data.macro_intelligence import macro_intelligence  # Was inline in start() L345

try:
    from core.meta_arbitrator import meta_arbitrator  # Was inline in 5+ locations
    _META_ARBITRATOR_AVAILABLE = True
except ImportError:
    meta_arbitrator = None
    _META_ARBITRATOR_AVAILABLE = False

try:
    from utils.cooldown_manager import cooldown_manager  # Was inline in _process_signal_event L861
    _COOLDOWN_AVAILABLE = True
except ImportError:
    cooldown_manager = None
    _COOLDOWN_AVAILABLE = False

try:
    from sophia.intelligence import MultiHorizonOracle  # Was inline in _process_signal_event L965
    _MULTI_HORIZON_ORACLE_AVAILABLE = True
except ImportError:
    MultiHorizonOracle = None
    _MULTI_HORIZON_ORACLE_AVAILABLE = False

try:
    from utils.interaction_monitor import get_interaction_monitor  # Was inline in _execute_approved_intent L1081
    _INTERACTION_MONITOR_AVAILABLE = True
except ImportError:
    get_interaction_monitor = None
    _INTERACTION_MONITOR_AVAILABLE = False

try:
    from utils.position_cleaner import position_cleaner  # Was inline in _process_fill_event L1183
    _POSITION_CLEANER_AVAILABLE = True
except ImportError:
    position_cleaner = None
    _POSITION_CLEANER_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════
# SOPHIA INTEGRATION (Phase SOPHIA-GLOBAL)
# QUÉ: Sophia AI actúa como filtro de veto GLOBAL para todas las señales.
# POR QUÉ: Antes solo TechnicalStrategy usaba Sophia internamente.
#   MLStrategy, Sniper y Statistical operaban sin supervisión de IA.
# PARA QUÉ: Proteger la cuenta de $13 con filtro probabilístico unificado.
# CÓMO: engine._sophia_veto_filter() examina cada SignalEvent antes
#   del RiskManager. Si win_probability < SOPHIA_MIN_CONFIDENCE → veto.
# CUÁNDO: En _process_signal_event(), después de TTL/precio validados.
# DÓNDE: core/engine.py → Engine._sophia_veto_filter()
# QUIÉN: Engine (orquestador) + SophiaIntelligence (facade)
# ═══════════════════════════════════════════════════════════════
from core.signal_scorer import SignalScorer

try:
    from sophia.intelligence import SophiaIntelligence
    _SOPHIA_AVAILABLE = True
except ImportError:
    _SOPHIA_AVAILABLE = False

try:
    import psutil
except ImportError:
    psutil = None

class PriorityBoundedQueue:
    """
    HFT Multi-Level Ring Buffer Queue (Phase OMNI: QOS Priority).
    - Uses collections.deque(maxlen) per priority level for O(1) auto-drop.
    - Non-blocking PUT.
    - Async GET (Checks Priority 0 -> 1 -> 2).
    """
    def __init__(self, maxsize=5000):
        self._deques = {
            0: collections.deque(maxlen=maxsize), # Critical: Fills, Executions, Scalping Signals
            1: collections.deque(maxlen=maxsize), # Normal: Swing Signals, Generic Orders
            2: collections.deque(maxlen=maxsize)  # Background: Market Data, Metrics
        }
        self._event = asyncio.Event()
    
    def put(self, item):
        """Put item into queue based on priority."""
        priority = getattr(item, 'priority', 1) 
        # Market events & Fills don't have priority by default.
        # Fills should be 0. Market can be 2. Let's infer if missing.
        if not hasattr(item, 'priority'):
            if hasattr(item, 'type'):
                # Handle Enum Types safely
                type_name = getattr(item.type, 'name', str(item.type))
                if type_name == 'FILL':
                    priority = 0
                elif type_name == 'MARKET':
                    priority = 0  # SUPREMO-V4: Market data is CRITICAL for HFT
                else:
                    priority = 1
            else:
                priority = 1
                
        # Clamp priority
        priority = max(0, min(2, priority))
        self._deques[priority].append(item)
        self._event.set()

    async def get(self):
        """Wait for and get next item strictly respecting priority."""
        while True:
            # Fast path
            for p in range(3):
                if self._deques[p]:
                    return self._deques[p].popleft()
            
            # Slow path: wait
            self._event.clear()
            
            # Double-check
            for p in range(3):
                if self._deques[p]:
                    return self._deques[p].popleft()
            
            await self._event.wait()

    def empty(self):
        return not any(d for d in self._deques.values())

    def task_done(self):
        pass # Not tracked for speed

# Backward compatibility alias
AsyncBoundedQueue = PriorityBoundedQueue

from utils.metrics_exporter import metrics
from core.omniscient_tracer import omniscient_trace

class Engine:
    """
    Event-Driven Trading Engine - EXTREME OPTIMIZATION EDITION
    - Zero global locks (Lock-free orchestration)
    - Direct payload processing (No lookups)
    - Burst-capable Event Loop
    """
    def __init__(self, events_queue: Optional[Any] = None):
        self.events = events_queue if events_queue else PriorityBoundedQueue(maxsize=5000) # Fast QoS queue
        self.data_handlers = []
        self.strategies = []
        self._strategies_by_symbol = collections.defaultdict(list) # O(1) Optimization
        self.execution_handler = None
        self.portfolio = None
        self.risk_manager = None
        self.order_manager = None
        self.running = True
        
        # Strategy Coordination
        self._strategy_cooldowns = {}
        
        # Metrics (optimized int counters)
        self.metrics = {
            'processed_events': 0,
            'discarded_events': 0,
            'strategy_executions': 0,
            'errors': 0,
            'avg_latency_ms': 0.0,
            'max_latency_ms': 0.0,
            'burst_events': 0,  # Phase OMNI: Burst-mode drain counter
        }
        
        # 🕵️ Phase 20: Forensic Recorder
        self.forensics = ForensicRecorder(self)
        
        # ⚡ LOW-LATENCY PHASE: Reusable QuantumTimer (Object Pooling)
        # QUÉ: Timer reutilizable para medir latencia de cada ciclo de mercado.
        # POR QUÉ: Antes se creaba un QuantumTimer() nuevo en CADA _process_market_event
        #   (L627-628). Con 26 símbolos × ~12 ticks/min = 312 allocaciones/min
        #   que luego GC debe recolectar.
        # PARA QUÉ: Reducir allocaciones en heap y presión sobre GC (~20-50μs/evento).
        # CÓMO: Un solo timer a nivel de instancia, con reset via __init__ en cada ciclo.
        self._market_timer = QuantumTimer()
        
        # 🌑 PHASE 24: LAYER 0 OPTIMIZATION (Protocol Nadir-Soberano)
        OSTuner.optimize()
        
        # 🧬 PHASE 26: Time Synchronization Integration
        TimeSynchronizer.sync()
        
        # 🏥 PHASE 27: System Monitor
        self.system_monitor = SystemMonitor()
        
        # 🌐 FORENSIC-FIX: Centralized Market Regime Detector
        self.market_regime = MarketRegimeDetector()
        
        # 🧠 SOPHIA-GLOBAL: Initialize Sophia Intelligence Engine
        self.sophia_intelligence = None
        
        # 💯 PHASE 1 ANTIFRAGIL: Signal Scorer
        self.signal_scorer = SignalScorer()
        # ═══════════════════════════════════════════════════════════════
        # AUDIT FIX: SOPHIA_MIN_CONFIDENCE was hardcoded to 0.70,
        # ignoring Config.Horizons.GlobalThresholds['sophia_win_prob_min'].
        # POR QUÉ: technical.py checked at 0.60 (Config) but engine.py
        #   checked AGAIN at 0.70 (hardcoded), creating a silent gap that
        #   killed all signals with 60-70% win probability.
        # PARA QUÉ: Single source of truth → Config controls threshold.
        # ═══════════════════════════════════════════════════════════════
        try:
            self.SOPHIA_MIN_CONFIDENCE = Config.Horizons.GlobalThresholds.get('sophia_win_prob_min', 0.60)
        except Exception:
            self.SOPHIA_MIN_CONFIDENCE = 0.60
        if _SOPHIA_AVAILABLE:
            try:
                self.sophia_intelligence = SophiaIntelligence(bar_minutes=5.0)
                self.sophia_intelligence.set_horizon_profile(1)  # Default: SCALPING
                logger.info("🧠 [SOPHIA-GLOBAL] Intelligence engine attached to Engine pipeline.")
            except Exception as e:
                logger.warning(f"⚠️ [SOPHIA-GLOBAL] Failed to init: {e}")
                self.sophia_intelligence = None
                
        # 🔭 AITS SHADOW BRIDGE (Read-Only Mode)
        self.shadow_bridge = None
        try:
            from aits_research.shadow_bridge import ShadowBridge
            self.shadow_bridge = ShadowBridge()
            logger.info("🔭 [SHADOW] AITS Shadow Bridge initialized in Engine.")
        except ImportError:
            logger.warning("⚠️ [SHADOW] aits_research.shadow_bridge not found. Shadow mode disabled.")
        except Exception as e:
            logger.warning(f"⚠️ [SHADOW] Failed to init Shadow Bridge: {e}")
            
        # 🌌 PHASE 9: OMEGA PROTOCOL
        self.omega_protocol = None
        try:
            from core.omega_protocol import OmegaProtocolManager
            self.omega_protocol = OmegaProtocolManager()
            logger.info("🌌 [OMEGA] Protocol Manager initialized in Engine.")
        except ImportError:
            logger.warning("⚠️ [OMEGA] core.omega_protocol not found. Omega Protocol disabled.")
        except Exception as e:
            logger.warning(f"⚠️ [OMEGA] Failed to init Omega Protocol: {e}")
            
        # 🧬 PHASE 3: EVOLUTION ENGINE (Live Mutation & Módulo Omega)
        try:
            self.evolution_engine = EvolutionEngine(population_size=10, mutation_rate=0.1)
            self._last_mutation_time = time.time()
            logger.info("🧬 [EVOLUTION] Engine initialized for Live Mutation.")
            
            # [ShadowDarwin] Continuous Evolution Daemon
            try:
                from core.evolution.shadow_darwin import ShadowDarwinDaemon
                self.shadow_darwin = ShadowDarwinDaemon()
                logger.info("🐉 [SHADOW-DARWIN] Evolutionary daemon successfully bound to Engine.")
            except ImportError:
                self.shadow_darwin = None
                logger.warning("⚠️ [SHADOW-DARWIN] Module not found. Evolution will be manual.")
                
        except Exception as e:
            logger.warning(f"⚠️ [EVOLUTION] Failed to init EvolutionEngine: {e}")
            self.evolution_engine = None
            self.shadow_darwin = None
        
        # 🧟 PHASE 2 ZOMBIES
        self.correlation_manager = None
        self.liquidity_guardian = None
        try:
            from core.sentiment_processor import SentimentProcessor
            self.sentiment_processor = SentimentProcessor()
        except Exception as e:
            logger.warning(f"Failed to init SentimentProcessor: {e}")
            self.sentiment_processor = None
            
        # 🏛️ INSTITUTIONAL SEGMENTATION (Major vs Meme)
        self.asset_classifier = AssetClassifier()
        self.segment_policy_engine = SegmentPolicyEngine(self.asset_classifier)
        logger.info("🏛️ [SEGMENT POLICY] Asset Classifier and Segment Policy Engine initialized.")

        # ⏳ [TEMPORAL] Initialize Supervisor
        self.temporal_supervisor = None

        # 🧬 Subscribe to Live Mutations
        event_bus.subscribe(EventChannel.MUTATION, self._on_mutation_event)


    # ... [Registration methods unchanged] ...
    def register_data_handler(self, handler: Any) -> None: 
        self.data_handlers.append(handler)
        
        if not self.correlation_manager:
            try:
                from core.correlation_manager import CorrelationManager
                self.correlation_manager = CorrelationManager(handler)
            except Exception as e:
                logger.error(f"Failed to initialize CorrelationManager: {e}")
                
        if not self.liquidity_guardian:
            try:
                from core.liquidity_guardian import LiquidityGuardian
                self.liquidity_guardian = LiquidityGuardian(handler)
            except Exception as e:
                logger.error(f"Failed to initialize LiquidityGuardian: {e}")
        
    def register_strategy(self, strategy: Any) -> None: 
        self.strategies.append(strategy)
        
        # ═══════════════════════════════════════════════════════════════
        # FORENSIC FIX #11: Inject Engine back-reference into strategy
        # QUÉ: La estrategia necesita acceder a risk_manager.prediction_tracker
        #   para calcular edge en tiempo real en request_exit_opinion().
        # POR QUÉ: Sin esta referencia, self.engine es None → edge_prob
        #   nunca se calcula → Alpha Decay Detection MUERTO.
        # ═══════════════════════════════════════════════════════════════
        strategy._engine_ref = self
        
        # 🧠 FORENSIC-V35: Inject Sophia AI directly into the strategy
        # FIX-FORENSIC-V82: Only inject if strategy doesn't already have a calibrated Sophia.
        # Bug: Engine.sophia_intelligence is ALWAYS 5min/SCALPING (L171-172).
        #   Overwriting strategy.sophia destroyed the horizon-calibrated singleton
        #   that SWING strategies created in __init__ (bar_minutes=60.0).
        # Impact: ALL SWING strategies operated with SCALPING-calibrated Sophia.
        if hasattr(self, 'sophia_intelligence') and not getattr(strategy, 'sophia', None):
            strategy.sophia = self.sophia_intelligence

        if hasattr(strategy, 'symbol') and strategy.symbol:
            self._strategies_by_symbol[strategy.symbol].append(strategy)
        else:
            # Multi-symbol strategies
            self._strategies_by_symbol['ALL'].append(strategy)
        
        # ═══════════════════════════════════════════════════════════════
        # CAPA 1 REGISTRATION: Register in RiskManager
        # QUÉ: Registra la estrategia en el RiskManager para permitir
        #   la delegación de la lógica de salida aislada por estrategia.
        # ═══════════════════════════════════════════════════════════════
        if self.risk_manager:
            self.risk_manager.register_strategy(strategy)

        # ═══════════════════════════════════════════════════════════════
        # FORENSIC AUDIT FIX: Register strategy in ExitOracle
        # QUÉ: Las estrategias con request_exit_opinion DEBEN estar
        #   registradas en el ExitOracle para que el Oracle las consulte.
        # POR QUÉ: Sin esto, oracle.strategies es un dict vacío y
        #   request_exit_opinion NUNCA se ejecuta → dynamic_targets
        #   NUNCA se genera → Phase 8 es código muerto.
        # PARA QUÉ: Completar la conexión neural del sistema omnisciente.
        # ═══════════════════════════════════════════════════════════════
        if hasattr(strategy, 'request_exit_opinion') and self.risk_manager:
            oracle = getattr(self.risk_manager, 'exit_oracle', None)
            if oracle:
                strat_id = getattr(strategy, 'id', None) or getattr(strategy, 'strategy_id', None) or strategy.__class__.__name__
                sym = getattr(strategy, 'symbol', 'ALL')
                oracle_key = f"{strat_id}_{sym}"
                oracle.register_strategy(oracle_key, strategy)
                logger.debug(f"🔮 [ORACLE-WIRE] Registered {oracle_key} in ExitOracle")
        
    def register_portfolio(self, portfolio: Any) -> None: 
        self.portfolio = portfolio
        # PHASE 2 POWER: Back-reference for HotAdapterRL feedback loop
        portfolio._engine = self
        
    def register_execution_handler(self, handler: Any) -> None: 
        self.execution_handler = handler
        # Inject OrderManager to ExecutionHandler so UserDataStream can map fills correctly
        if hasattr(self, 'order_manager') and hasattr(self.execution_handler, 'set_order_manager'):
            self.execution_handler.set_order_manager(self.order_manager)
        
    def register_risk_manager(self, manager: Any) -> None: 
        self.risk_manager = manager
        # Inject Phase 2 Zombie Features into Risk Manager
        self.risk_manager.correlation_manager = getattr(self, 'correlation_manager', None)
        self.risk_manager.liquidity_guardian = getattr(self, 'liquidity_guardian', None)
        self.risk_manager.sentiment_processor = getattr(self, 'sentiment_processor', None)
        
        # 🧠 SOPHIA-GLOBAL: Inject into Risk Manager for DCA / exit validation
        self.risk_manager.sophia = getattr(self, 'sophia_intelligence', None)

        # ⏳ Initialize Temporal Supervisor if not present
        if not self.temporal_supervisor and self.portfolio and self.risk_manager:
             self.temporal_supervisor = TemporalSupervisor(self.portfolio, self.risk_manager, self)
             
        # 🛡️ Phase 20: Link Forensics to Kill Switch
        if self.risk_manager and hasattr(self.risk_manager, 'kill_switch'):
             # Define callback that captures snapshot
             def forensic_dump(reason):
                 if hasattr(self, 'forensics'):
                     self.forensics.capture_snapshot(trigger_reason=f"KILL_SWITCH: {reason}")
             
             self.risk_manager.kill_switch.set_forensic_callback(forensic_dump)

    def register_order_manager(self, manager: Any) -> None:
        self.order_manager = manager

    def unregister_strategy(self, symbol: str) -> None:
        """Removes all strategies associated with a symbol."""
        to_remove = [s for s in self.strategies if getattr(s, 'symbol', None) == symbol]
        for s in to_remove:
            try:
                if hasattr(s, 'stop'):
                    s.stop()
                self.strategies.remove(s)
            except Exception as e:
                logger.error(f"Error unregistering strategy for {symbol}: {e}")
        
        if to_remove:
            logger.info(f"♻️ Engine: Unregistered {len(to_remove)} strategies for {symbol}")

    async def start(self):
        """Main event loop - 100% AsyncIO-Driven"""
        # 🌐 CTOS DATA OMNISCIENCE: Start Macro/Micro Polling
        macro_intelligence.start_background()
        
        # 🧠 Supreme Meta-Coordinator
        try:
            if _META_ARBITRATOR_AVAILABLE and meta_arbitrator:
                meta_arbitrator.start()
                asyncio.create_task(self._drain_meta_arbitrator())
        except Exception as e:
            logger.error(f"Failed to start Meta-Arbitrator: {e}")
            
        # 🐉 ShadowDarwin Continuous Evolution
        if hasattr(self, 'shadow_darwin') and self.shadow_darwin:
            try:
                self.shadow_darwin.start()
            except Exception as e:
                logger.error(f"Failed to start ShadowDarwin: {e}")
            
        # ⏳ Start Temporal Supervisor Loop
        if self.temporal_supervisor:
             asyncio.create_task(self.temporal_supervisor.run_temporal_loop())
             
        # 🔄 Start periodic reconciliation loop
        if self.execution_handler and self.portfolio:
             asyncio.create_task(self._reconciliation_loop())
            
        # AEGIS-ULTRA: Core Pinning (Phase 5)
        if Config.Aegis.CORE_PINNING and psutil:
            try:
                p = psutil.Process()
                # Pin to physical cores only (Ryzen 7 5700U: 0, 2, 4, 6, 8, 10, 12, 14)
                # We select the first 4 physical cores for the main Engine loop
                physical_cores = [0, 2, 4, 6] 
                p.cpu_affinity(physical_cores)
                p.nice(psutil.HIGH_PRIORITY_CLASS)
                logger.info(f"🛡️ AEGIS-ULTRA: Engine Pinned to Cores {physical_cores} | Priority: HIGH")
            except Exception as e:
                logger.warning(f"⚠️ Failed to set CPU Affinity: {e}")
        
        # ✅ PHASE IV: COLD BOOT (Operational Continuity)
        # Verify if we need to recover state from Exchange
        if self.execution_handler and self.portfolio:
             try:
                 logger.info("🔌 [COLD BOOT] Initiating State Recovery Protocol...")
                 await asyncio.to_thread(self.execution_handler.sync_portfolio_state, self.portfolio)
             except Exception as e:
                 logger.critical(f"❌ [COLD BOOT] FAILED: {e} - Proceeding with local state.")
        
        logger.info(f"🚀 Engine started. Active Strategies: {len(self.strategies)}")
        
        while self.running:
            try:
                # 🏥 PHASE 27: Graceful Degradation Check
                if not self.system_monitor.check_health():
                    await asyncio.sleep(0.1) # Cool down

                # 1. Get first event (Wait max 1s to allow maintenance)
                try:
                    event = await asyncio.wait_for(self.events.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Idle Cycle: Smart GC (Phase 15)
                    regime = self._get_current_market_regime()
                    if 'VOLATILE' not in regime and 'TRENDING' not in regime:
                        GCTuner.check_maintenance()
                        
                    # 🧟 ZOMBIE UPDATE: Recalculate Correlation Matrix during idle time
                    if hasattr(self, 'correlation_manager') and self.correlation_manager:
                        now = time.time()
                        if now - getattr(self.correlation_manager, 'last_update', 0) > 300: # 5 mins
                            self.correlation_manager.update_correlations()
                            self.correlation_manager.last_update = now
                            
                            # 🌐 GRAPH LAYER UPDATE
                            try:
                                if _META_ARBITRATOR_AVAILABLE and meta_arbitrator:
                                    if hasattr(self.correlation_manager, 'correlation_matrix') and self.correlation_manager.correlation_matrix is not None:
                                        meta_arbitrator.graph_layer.update_graph_edges(self.correlation_manager.correlation_matrix)
                            except Exception as e:
                                logger.error(f"Failed to update Graph Edges: {e}")
                    
                    # 🧬 LIVE MUTATION TRIGGER
                    if hasattr(self, 'evolution_engine') and self.evolution_engine:
                        now = time.time()
                        if now - getattr(self, '_last_mutation_time', 0) > 3600: # Every 1 hour
                            self._trigger_live_mutation()
                            self._last_mutation_time = now
                            
                    # 🧹 MEMORIA OMNISCIENTE (Pruning & Cleanup)
                    # QUÉ: Ejecuta limpieza de base de datos para evitar bloat en uptimes largos.
                    # POR QUÉ: Las tablas de telemetría forense crecen masivamente.
                    now_time = time.time()
                    if now_time - getattr(self, '_last_db_prune_time', 0) > 86400: # Every 24 hours
                        if hasattr(self, 'portfolio') and hasattr(self.portfolio, 'db'):
                            try:
                                self.portfolio.db.prune_historical_data(days_to_keep=7)
                            except Exception as e:
                                logger.error(f"Error en Pruning de Memoria: {e}")
                        self._last_db_prune_time = now_time

                    # ═══════════════════════════════════════════════════════════════
                    # CTOS Phase 5: SYSTEMIC INTEGRITY FORENSICS
                    # QUÉ: Registrar estado general del sistema (strategies activas, posiciones, health)
                    # POR QUÉ: Permite analizar divergencias retrospectivamente
                    # ═══════════════════════════════════════════════════════════════
                    if now_time - getattr(self, '_last_system_awareness_time', 0) > 300: # Every 5 minutes
                        if hasattr(self, 'portfolio') and hasattr(self.portfolio, 'db'):
                            try:
                                active_strategies = global_state.get_system_capabilities()
                                positions = {s: p for s,p in self.portfolio.positions.items()} if getattr(self.portfolio, 'positions', None) else {}
                                self.portfolio.db.log_system_awareness_snapshot(active_strategies, positions)
                            except Exception as e:
                                logger.error(f"Error logging system awareness: {e}")
                        self._last_system_awareness_time = now_time
                            
                    continue
                
                # ⚡ PHASE OMNI: BURST-MODE EVENT DRAIN
                # Process up to 32 events per yield cycle to reduce asyncio overhead.
                # QUÉ: Drain multiple events in a single GC-free critical section.
                # POR QUÉ: Cada await cede control al event loop (~15μs overhead).
                # PARA QUÉ: Reducir latencia total bajo carga alta (bursts de mercado).
                # CÓMO: Recoger hasta 32 eventos de la deque sin await entre ellos.
                burst_batch = [event]
                _BURST_MAX = 32
                while len(burst_batch) < _BURST_MAX and not self.events.empty():
                    try:
                        # Drain from priority deques in order (0→1→2)
                        drained = False
                        for p in range(3):
                            if self.events._deques[p]:
                                burst_batch.append(self.events._deques[p].popleft())
                                drained = True
                                break
                        if not drained:
                            break
                    except (IndexError, KeyError):
                        break
                
                # 2. Critical Section (GC Disabled) — process entire burst
                # ════ CTOS: CLOCK TICK ════
                # QUÉ: Dispara un nuevo ciclo de reloj maestro ANTES del burst.
                # POR QUÉ: Todos los módulos deben operar sobre el mismo instante t.
                # CÓMO: global_clock.tick() congela timestamp y notifica suscriptores.
                global_clock.tick()
                
                start_loop = time.perf_counter()
                with GCTuner.critical_section():
                    for evt in burst_batch:
                        await self.process_event(evt)

                # [DF-A2] Jitter Detection: Measure processing time
                end_loop = time.perf_counter()
                batch_size = len(burst_batch)
                loop_duration = (end_loop - start_loop) * 1_000_000 # microseconds
                per_event_us = loop_duration / batch_size
                
                if per_event_us > 500: # 500μs per-event threshold
                    logger.warning(f"⚠️ [JITTER] Burst({batch_size}) Avg: {per_event_us:.0f}μs/evt (Total: {loop_duration:.0f}μs)")
                    latency_monitor.track('engine_jitter_warning', loop_duration / 1000)
                
                # Update rolling avg latency
                for _ in burst_batch:
                    current_avg = self.metrics['avg_latency_ms']
                    processed = self.metrics['processed_events']
                    self.metrics['avg_latency_ms'] = (current_avg * processed + (per_event_us/1000)) / (processed + 1)
                    self.metrics['processed_events'] += 1
                
                if (loop_duration/1000) > self.metrics['max_latency_ms']:
                    self.metrics['max_latency_ms'] = loop_duration / 1000
                
                # Track burst size for telemetry
                if batch_size > 1:
                    self.metrics['burst_events'] += batch_size
                
                # Mark as done
                for _ in burst_batch:
                    self.events.task_done()
                
                # ✅ PHASE 18: RYZEN 7 SNIPER (Dynamic Orchestration)
                if self.metrics['processed_events'] % 100 == 0:
                     await self._optimize_ryzen_resources()
                
                # ════ CTOS PHASE 2: EVENT BUS DRAIN ════
                # QUÉ: Procesa eventos encolados en el EventBus.
                # POR QUÉ: Sin esto, los subscribers del EventBus nunca reciben eventos.
                # CUÁNDO: Al final de cada burst loop, fuera del GC critical section.
                try:
                    event_bus.process_queue(max_items=50)
                except Exception as e:
                    logger.error(f"Error processing EventBus queue: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Engine Loop Error: {e}")
                self.metrics['errors'] += 1

    async def _optimize_ryzen_resources(self):
        """
        [PHASE 18] Dynamic Core Pinning & Thermal Throttling.
        """
        if not psutil: return
        try:
            throttle = 0.0
            
            # 2. Dynamic Affinity (Load Balancer)
            cpu_pct = psutil.cpu_percent()
            p = psutil.Process()
            
            if cpu_pct > 80.0:
                # High Load: Unleash all Cores (Ryzen 7 5700U has 8 cores / 16 threads)
                # We use all logical processors
                current_affinity = p.cpu_affinity()
                if len(current_affinity) < 16:
                    p.cpu_affinity(list(range(16)))
                    logger.info(f"⚡ [SNIPER] High Load ({cpu_pct}%). Expanding to ALL 16 Threads.")
            elif cpu_pct < 20.0:
                # Low Load: Conserve Power (Eco Mode)
                # Pin to first 4 physical cores (0, 2, 4, 6)
                current_affinity = p.cpu_affinity()
                target = [0, 2, 4, 6]
                if current_affinity != target:
                    p.cpu_affinity(target)
                    logger.info(f"🍃 [SNIPER] Low Load ({cpu_pct}%). Eco Mode Active (4 Cores).")
                    
        except Exception:
            pass # Fail silently on permission/OS errors

    async def process_event(self, event: Union[MarketEvent, SignalEvent, OrderEvent, FillEvent]) -> None:
        """Route event asynchronously"""
        try:
            metrics.inc_event() # Phase 53: Metrics
            
            # NORMALIZACIÓN DE TIPO (Phase OMNI: Enum Resilience)
            # QUÉ: Asegura que 'etype' sea siempre un string para las comparaciones.
            # POR QUÉ: Event.type es un Enum (EventType.SIGNAL), pero Engine
            #   comparaba contra strings ('SIGNAL'), resultando en falsos negativos.
            raw_type = getattr(event, 'type', 'UNKNOWN')
            etype = getattr(raw_type, 'name', str(raw_type))
            
            # AEGIS-ULTRA: LATENCY CIRCUIT BREAKER (Phase 16)
            # Check latency before processing Signals/Orders (Market data always processed)
            if etype in ['SIGNAL', 'ORDER'] and self.data_handlers:
                dh = self.data_handlers[0]
                if hasattr(dh, 'get_latency_metrics'):
                    avg_ping, max_ping = dh.get_latency_metrics()
                    if avg_ping > 150.0:
                        if etype == 'SIGNAL':
                            logger.warning(f"🛑 [CIRCUIT BREAKER] High Latency ({avg_ping:.1f}ms). Signal Dropped.")
                            self.metrics['discarded_events'] += 1
                            return
                        elif etype == 'ORDER':
                            # Optional: allow closing orders? For now block all to be safe against stale prices.
                            # Better: Allow CLOSE, Block ENTRY. But Engine doesn't know intent easily.
                            # Conservative: Block ALL new actions. Risk Manager handles emergency exits via direct API?
                            # For now, just log and block.
                            logger.warning(f"🛑 [CIRCUIT BREAKER] High Latency ({avg_ping:.1f}ms). Order Blocked.")
                            return

            if etype == 'MARKET':
                await self._process_market_event(event) # type: ignore
            elif etype == 'SIGNAL':
                await self._process_signal_event(event) # type: ignore
            elif etype == 'ORDER':
                await self._process_order_event(event) # type: ignore
            elif etype == 'FILL':
                await self._process_fill_event(event) # type: ignore
            elif etype == 'AUDIT':
                pass 
            else:
                pass
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"🚨 [FATAL-ENGINE] Event Logic Error for {getattr(event, 'type', 'UNKNOWN')}:\nException: {e}\nTraceback:\n{tb}")
            self.metrics['errors'] += 1


    @omniscient_trace(layer="CORTEX")
    async def _process_market_event(self, event) -> None:
        # ⏱️ [QUANTUM TELEMETRY] Iniciar reloj atómico (LOW-LATENCY: reusable timer)
        self._market_timer.start_ns = time.perf_counter_ns()
        timer = self._market_timer
        
        global_state.update_from_market_event(event)
        
        if _META_ARBITRATOR_AVAILABLE and meta_arbitrator:
            try:
                sym_state = global_state.get_state(event.symbol)
                if sym_state:
                    features = {
                        'orderflow_imbalance': sym_state.get('order_imbalance', 0.0),
                        'spread_cost_pct': sym_state.get('spread', 0.0),
                        'trend_score_m5': sym_state.get('trend_strength', 0.0),
                        'volatility_atr_pct': sym_state.get('volatility', 0.0)
                    }
                    if meta_arbitrator.graph_layer:
                        meta_arbitrator.graph_layer.update_symbol_state(event.symbol, features)
            except Exception as e:
                pass
        
        if event.symbol and event.close_price and self.portfolio:
             self.portfolio.update_market_price(event.symbol, event.close_price)
             
             if self.risk_manager and hasattr(self.risk_manager, 'prediction_tracker'):
                 if self.risk_manager.prediction_tracker:
                     try:
                         self.risk_manager.prediction_tracker.update_forward_returns(
                             symbol=event.symbol,
                             current_price=event.close_price,
                             timestamp=getattr(event, 'datetime', None)
                         )
                     except Exception as e:
                         pass

             has_position = event.symbol in self.portfolio.positions or \
                            any(k.startswith(event.symbol) for k in self.portfolio.virtual_ledger)
             if has_position and self.risk_manager and self.data_handlers:
                 try:
                     stop_signals = self.risk_manager.check_stops(
                         self.portfolio, self.data_handlers[0],
                         symbol_filter=event.symbol
                     )
                     if stop_signals:
                         for sig in stop_signals:
                             self.events.put(sig)
                 except Exception as e:
                     logger.error(f"Error checking stops: {e}")
                     pass

        strategies_to_run = self._strategies_by_symbol.get(event.symbol, [])
        strategies_to_run.extend(self._strategies_by_symbol.get('ALL', []))
        
        # ================================================================
        # NANO-LATENCY SYNCHRONOUS EVALUATION
        # ================================================================
        # FORENSIC-AUDIT-FIX: Removed ThreadPoolExecutor.
        # Since strategy.calculate_signals is an O(1) cached evaluation,
        # ThreadPool switching and asyncio.wrap_future overhead was slowing it down.
        # Running synchronously eliminates context switching latency entirely.
        for strategy in strategies_to_run:
            if _COOLDOWN_AVAILABLE and cooldown_manager:
                if not cooldown_manager.can_evaluate(strategy.strategy_id, event.symbol):
                    continue
            
            try:
                signal = strategy.calculate_signals(event)
                if signal:
                    if isinstance(signal, list):
                        for s in signal:
                            if s: self.events.put(s)
                    else:
                        self.events.put(signal)
            except Exception as e:
                logger.error(f"Strategy {strategy.strategy_id} error: {e}")
                pass

    @omniscient_trace(layer="CORTEX")
    async def _process_signal_event(self, event):
        """Process SIGNAL event asynchronously"""
        logger.info(f"⚡ [ENGINE] Processing Signal: {event.symbol} {event.signal_type} (Conf: {getattr(event, 'confidence', 0):.2f})")
        
        if not self._validate_signal_ttl(event):
            logger.warning(f"❌ [ENGINE] Signal {event.symbol} rejected: TTL Expired")
            self.metrics['discarded_events'] += 1
            return
            
        current_price = self._get_validated_price(event.symbol)
        if not current_price:
            logger.warning(f"❌ [ENGINE] Signal {event.symbol} rejected: Price Validation Failed")
            self.metrics['discarded_events'] += 1
            return
            
        # ⏳ TEMPORAL CONSTRAINTS
        is_exit = (getattr(event, 'signal_type', None) == SignalType.EXIT) or str(getattr(event, 'signal_type', None)) == 'SignalType.EXIT' or getattr(event, 'is_exit', False) or (getattr(event, 'strategy_id', '') == "FLIP_EXIT")
        if not is_exit and self.temporal_supervisor and not hasattr(event, 'is_vetoed'):
             # Modify position size or reject signal if temporal rule violates constraints
             adj_size, adj_score, allowed = self.temporal_supervisor.apply_temporal_constraints(
                 event, 
                 getattr(event, 'quantity_pct', 1.0),
                 getattr(event, 'confidence', 0.0)
             )
             if not allowed or getattr(event, 'confidence', 0.0) < adj_score:
                  logger.warning(f"⏳ [TEMPORAL] Signal rejected for temporal maturity constraints. Confidence {getattr(event, 'confidence', 0.0):.2f} < {adj_score:.2f}")
                  self.metrics['discarded_events'] += 1
                  return
             # Update modified size via property injection
             event = dataclasses.replace(event, temporal_size_modifier=adj_size / max(getattr(event, 'quantity_pct', 1.0), 0.01))

        if self.portfolio:
             # 🌌 MULTIVERSE SIMULATOR (Quantum Shadow Validation)
             # Adapt TP and SL dynamically without blocking the Engine
             if event.signal_type in (SignalType.LONG, SignalType.SHORT):
                 from core.multiverse_simulator import multiverse_simulator
                 from config import Config
                 
                 base_tp = getattr(event, 'tp_pct', 0.0)
                 base_sl = getattr(event, 'sl_pct', 0.0)
                 
                 # Fetch default if not set by strategy
                 if base_tp == 0.0 or base_sl == 0.0:
                     horizon = getattr(event, 'horizon', 'SCALPING').upper()
                     if horizon == 'SCALPING':
                         base_tp = base_tp or Config.Risk.SCALPING_PARAMS.get('tp_pct', 0.0068)
                         base_sl = base_sl or Config.Risk.SCALPING_PARAMS.get('sl_pct', 0.0035)
                     else:
                         base_tp = base_tp or Config.Risk.SWING_PARAMS.get('tp_pct', 0.045)
                         base_sl = base_sl or Config.Risk.SWING_PARAMS.get('sl_pct', 0.015)
                 
                 opt_tp, opt_sl, reasoning = multiverse_simulator.simulate_trajectories(
                     event, self.data_provider, base_tp, base_sl
                 )
                 
                 # Inject optimal params into event
                 event = dataclasses.replace(event, tp_pct=opt_tp, sl_pct=opt_sl)
                 if opt_tp != base_tp or opt_sl != base_sl:
                     logger.debug(f"🌌 [MULTIVERSE] Optimized TP/SL for {event.symbol}: TP {base_tp*100:.2f}%->{opt_tp*100:.2f}%, SL {base_sl*100:.2f}%->{opt_sl*100:.2f}% ({reasoning})")

             self.portfolio.update_signal(event)
             
             # [PHASE 9: ML Metacognition] Propagate Brier Score & Entropy to Portfolio for Dashboard Persistence
             if getattr(event, 'metadata', None):
                 if 'brier_score' in event.metadata:
                     self.portfolio.current_brier_score = event.metadata['brier_score']
                 if 'ppo_entropy' in event.metadata:
                     self.portfolio.current_ppo_entropy = event.metadata['ppo_entropy']

        # --- SHOCK REGIME & FUNDING EVASION (DYNAMIC FREEZE) ---
        is_exit_signal = hasattr(event, 'signal_type') and str(event.signal_type) == 'SignalType.EXIT'
        if not is_exit_signal and self.data_handlers:
            # 🕒 FUNDING FEE EVASION (Phase 1 Protection)
            try:
                # datetime/timezone already imported at top-level
                now_utc = datetime.now(timezone.utc)
                # Binance funding occurs at 00:00, 08:00, 16:00 UTC.
                # We block entries from XX:45 to XX:59 to avoid funding drag.
                if (now_utc.hour in (23, 7, 15)) and (now_utc.minute >= 45):
                    logger.warning(f"🕒 [FUNDING EVASION] Entry signal {event.symbol} blocked. Funding snapshot in < 15 mins.")
                    self.metrics['discarded_events'] += 1
                    return
            except Exception as e:
                logger.error(f"Funding evasion error: {e}")
            try:
                # cooldown_manager imported at top-level
                # Check if currently frozen
                frozen_key = f"SHOCK_FREEZE_{event.symbol}"
                # Peeking into custom dictionary to act as a lock check
                if hasattr(cooldown_manager, 'custom_cooldowns') and frozen_key in cooldown_manager.custom_cooldowns:
                    # datetime/timezone already imported at top-level
                    elapsed = (datetime.now(timezone.utc) - cooldown_manager.custom_cooldowns[frozen_key]).total_seconds()
                    if elapsed < 300: # 5 min freeze
                        logger.warning(f"❄️ [SHOCK BLOCK] Entry signal for {event.symbol} blocked. Frozen for {(300 - elapsed):.0f}s.")
                        self.metrics['discarded_events'] += 1
                        return
                
                # If not frozen, check if we SHOULD freeze (evaluate Market Regime shock)
                dh = self.data_handlers[0]
                bars = dh.get_latest_bars(event.symbol, n=50)
                if bars and len(bars) >= 15:
                    oi_delta = 0.0
                    derivatives = {}
                    if hasattr(dh, 'get_derivatives_metrics'):
                        derivatives = dh.get_derivatives_metrics(event.symbol)
                        oi_delta = derivatives.get('oi_delta_15m', 0.0)
                        
                    # 🌌 OMEGA PROTOCOL ECOSYSTEM CHECK
                    if getattr(self, 'omega_protocol', None):
                        # Simulating OFI and Liquidation metrics locally
                        local_ofi = oi_delta # Using OI delta as a proxy for POC
                        local_liq = 0.0 # Could be fetched from DH if available
                        self.omega_protocol.assess_ecosystem_health(event.symbol, local_ofi, local_liq)
                        
                    if self.market_regime.is_volatility_shock(bars, oi_delta=oi_delta):
                        logger.critical(f"🌪️ [SHOCK REGIME DETECTED] Squeeze on {event.symbol}! Activating 5-min Freeze.")
                        # Activate the freeze (cooldown_manager will set the timestamp)
                        cooldown_manager.check_custom_cooldown(frozen_key, 300)
                        self.metrics['discarded_events'] += 1
                        return

                    # 🧟 ZOMBIE FEATURE INTEGRATION: Calculate tension for RiskManager
                    try:
                        shift_pred = self.market_regime.predict_regime_shift(event.symbol, bars)
                        event = dataclasses.replace(event, tension=shift_pred.get('tension', 0.0))
                    except Exception as e:
                        logger.error(f"Error predicting regime shift tension: {e}")
                        event = dataclasses.replace(event, tension=0.0)
            except Exception as e:
                logger.error(f"Shock Evasion Error: {e}")

        # 🧟 ZOMBIE FEATURE INTEGRATION: Record Signal in Prediction Tracker
        if self.risk_manager and getattr(self.risk_manager, 'prediction_tracker', None):
            try:
                _strat_id = getattr(event, 'strategy_id', 'Unknown')
                _dir = getattr(event, 'signal_type', 'UNKNOWN')
                _direction_str = 'long' if 'LONG' in str(_dir) else ('short' if 'SHORT' in str(_dir) else str(_dir).lower())
                _horizon = getattr(event, 'horizon', 'SCALPING')
                
                if _direction_str in ['long', 'short']:
                    # CTOS Phase 4 Fix: Correct kwargs for record_signal
                    self.risk_manager.prediction_tracker.record_signal(
                        strategy_id=_strat_id,
                        symbol=event.symbol,
                        direction=_direction_str,
                        horizon=_horizon,
                        entry_price=current_price,
                        sl_pct=getattr(event, 'sl_pct', 0.0) or 0.0,
                        tp_pct=getattr(event, 'tp_pct', 0.0) or 0.0,
                        confidence=getattr(event, 'strength', 0.5) or 0.5,
                        timestamp=getattr(event, 'datetime', None)
                    )
            except Exception as e:
                logger.error(f"Failed to record signal in PredictionTracker: {e}")
        # CTOS Phase 5: SYSTEMIC INTEGRITY (CROSS-STRATEGY AWARENESS)
        # QUÉ: Verificar `get_competing_strategies()` antes de procesar señales contradictorias.
        # POR QUÉ: Evitar "pisarse las patas" entre Swing y Scalping.
        # ═══════════════════════════════════════════════════════════════
        try:
            _signal_dir = 'LONG' if 'LONG' in str(event.signal_type) else ('SHORT' if 'SHORT' in str(event.signal_type) else None)
            if _signal_dir and self.portfolio:
                _horizon = getattr(event, 'horizon', 'SCALPING')
                _opposing_horizon = 'SWING' if _horizon == 'SCALPING' else 'SCALPING'
                _opp_pos = self.portfolio.get_horizon_position(event.symbol, _opposing_horizon)
                
                if _opp_pos and _opp_pos.get('quantity', 0) != 0:
                    _opp_actual_dir = 'LONG' if _opp_pos['quantity'] > 0 else 'SHORT'
                    if _opp_actual_dir != _signal_dir:
                        # Obtenemos las estrategias que compiten
                        competing = global_state.get_competing_strategies(event.symbol, _opposing_horizon)
                        comp_names = [c.get('strategy_id') for c in competing]
                        
                        logger.warning(
                            f"⚠️ [SYSTEMIC INTEGRITY] Conflicto direccional en {event.symbol}: "
                            f"Señal {_horizon} {_signal_dir} vs Posición {_opposing_horizon} {_opp_actual_dir}. "
                            f"Estrategias compitiendo: {comp_names}. Pasando a Meta-Arbitrator para resolución explícita."
                        )
                        new_metadata = dict(event.metadata or {})
                        new_metadata['competing_horizon'] = _opposing_horizon
                        new_metadata['competing_direction'] = _opp_actual_dir
                        event = dataclasses.replace(event, metadata=new_metadata)
        except Exception as e:
            logger.error(f"Failed to check competing strategies: {e}")

        # ═══════════════════════════════════════════════════════════════
        # PHASE 3: MULTI-HORIZON ORACLE VETO (Centralized SignalBroker)
        # QUÉ: Valida el "Clash Vector" macro (1d, 1w) contra la dirección local.
        # POR QUÉ: Extraído de technical.py para asegurar que la estrategia
        #   solo emita la matemática técnica pura.
        # ═══════════════════════════════════════════════════════════════
        try:
            # MultiHorizonOracle imported at top-level
            meta = getattr(event, 'metadata', {})
            tf_data = meta.get('timeframe_data', {})
            if tf_data and self.data_handlers:
                _dir_str = 'LONG' if 'LONG' in str(event.signal_type) else ('SHORT' if 'SHORT' in str(event.signal_type) else 'UNKNOWN')
                _horizon = getattr(event, 'horizon', 'SCALPING')
                
                if _dir_str != 'UNKNOWN' and _MULTI_HORIZON_ORACLE_AVAILABLE and MultiHorizonOracle:
                    oracle_verdict = MultiHorizonOracle.evaluate_clash_vector(tf_data, _dir_str, horizon=_horizon)
                    if oracle_verdict.get('is_vetoed', False):
                        _clash = oracle_verdict.get('clash_score', 0.0)
                        # SOFT VETO: Penalize confidence instead of hard blocking immediately.
                        # The Meta-Arbitrator and RiskManager will ultimately decide.
                        if _clash > 0.85:
                            logger.info(f"🔮 [ORACLE VETO] {event.symbol} {_dir_str} BLOCKED (EXTREME) | Clash: {_clash:.1%} | Macro: {oracle_verdict.get('macro_context', '')}")
                            self.metrics['discarded_events'] += 1
                            return
                        else:
                            _penalty = max(0.4, 1.0 - _clash)
                            event = dataclasses.replace(event, strength=getattr(event, 'strength', 0.5) * _penalty)
                            logger.info(f"🔮 [ORACLE SOFT] {event.symbol} {_dir_str} PENALIZED x{_penalty:.2f} | Clash: {_clash:.1%} | Macro: {oracle_verdict.get('macro_context', '')}")
        except Exception as e:
            logger.warning(f"🔮 [ORACLE] Evaluation failed for {event.symbol}: {e}")

        # ═══════════════════════════════════════════════════════════════
        # SOPHIA-GLOBAL VETO FILTER
        # QUÉ: Sophia revisa la señal ANTES del RiskManager.
        # POR QUÉ: RiskManager valida sizing/risk. Sophia valida CALIDAD
        #   probabilística de la señal con calibración Bayesiana.
        # PARA QUÉ: Filtro dual: Sophia (calidad) → RiskManager (riesgo).
        # ═══════════════════════════════════════════════════════════════
        if not self._sophia_veto_filter(event):
            self.metrics['discarded_events'] += 1
            return
            
        # 🌌 OMEGA PROTOCOL VETO
        if getattr(self, 'omega_protocol', None):
            sig_dict = {
                "symbol": getattr(event, 'symbol', 'UNKNOWN'),
                "horizon": getattr(event, 'horizon', 'SCALPING')
            }
            if self.omega_protocol.should_veto_signal(sig_dict):
                self.metrics['discarded_events'] += 1
                return

        # 🏛️ INSTITUTIONAL SEGMENTATION (Policy Enforcement)
        try:
            current_regime_str = self._get_current_market_regime()
            event = self.segment_policy_engine.enforce_policy(event, current_regime_str)
            if hasattr(event, 'segment_policy') and event.segment_policy.veto:
                logger.warning(f"🛑 [SEGMENT POLICY] Signal {event.symbol} VETOED by Segment Policy Engine.")
                self.metrics['discarded_events'] += 1
                return
        except Exception as e:
            logger.error(f"Segment Policy Engine Error: {e}")

        # 💯 SISTEMA AUTÓNOMO ANTIFRÁGIL - FASE 1: Scoring y Breakeven
        if hasattr(self, 'signal_scorer') and self.signal_scorer:
            _dir_str = 'LONG' if 'LONG' in str(event.signal_type) else ('SHORT' if 'SHORT' in str(event.signal_type) else 'UNKNOWN')
            if _dir_str in ['LONG', 'SHORT']:
                try:
                    dh = self.data_handlers[0] if self.data_handlers else None
                    regime_str = self._get_current_market_regime()
                    
                    total_score, breakdown = self.signal_scorer.calculate_score(event, dh, self.portfolio, regime_str)
                    
                    if not hasattr(event, 'metadata'):
                        event.metadata = {}
                    event.metadata['final_score'] = total_score
                    event.metadata['score_breakdown'] = breakdown
                    
                    logger.debug(f"💯 [SCORER] {event.symbol} {_dir_str} Score: {total_score}/100 Breakdown: {breakdown}")
                    
                    horizon_str = getattr(event, 'horizon', '').upper()
                    required_score = self.signal_scorer.min_pass_score
                    if horizon_str == 'SCALPING':
                        required_score = getattr(self.signal_scorer, 'scalping_min_score', self.signal_scorer.min_pass_score)
                    elif horizon_str == 'SWING':
                        required_score = getattr(self.signal_scorer, 'swing_min_score', self.signal_scorer.min_pass_score)

                    if total_score < required_score:
                        logger.warning(f"🛑 [SCORER VETO] {event.symbol} Score {total_score} < {required_score} ({horizon_str})")
                        self.metrics['discarded_events'] += 1
                        return
                        
                    is_viable, be_msg = self.signal_scorer.check_breakeven_viability(event, current_price)
                    if not is_viable:
                        logger.warning(f"🛑 [BREAKEVEN VETO] {event.symbol}: {be_msg}")
                        self.metrics['discarded_events'] += 1
                        return
                        
                except Exception as e:
                    logger.error(f"Error en SignalScorer: {e}")

        # 🧠 META-COORDINATOR INJECTION
        # Instead of directly processing through the RiskManager, we submit this to the Supreme Arbitrator
        # for conflict resolution, portfolio checks, and global context weighing.
        try:
            if _META_ARBITRATOR_AVAILABLE and meta_arbitrator:
                await meta_arbitrator.submit_intent(event)
                logger.debug(f"📥 [ENGINE] Intent {event.symbol} passed to Meta-Arbitrator.")
        except Exception as e:
            logger.error(f"Meta-Arbitrator Submit Error: {e}")
            self.metrics['errors'] += 1

    async def _drain_meta_arbitrator(self):
        """Background task that continuously processes approved intents from the Meta-Arbitrator."""
        try:
            if not (_META_ARBITRATOR_AVAILABLE and meta_arbitrator):
                return
            while self.running:
                approved_intent = await meta_arbitrator.get_approved_intent()
                await self._execute_approved_intent(approved_intent)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Meta-Arbitrator Drain Error: {e}")

    async def _reconciliation_loop(self):
        """Periodic background task that reconciles local position registry with Binance every 60s."""
        logger.info("🔄 [RECONCILIATION] Starting periodic Binance Futures reconciliation loop...")
        while self.running:
            try:
                await asyncio.sleep(60.0)
                if self.execution_handler and self.portfolio:
                    logger.debug("🔄 [RECONCILIATION] Syncing local portfolio registry with Binance...")
                    await asyncio.to_thread(self.execution_handler.sync_portfolio_state, self.portfolio)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ [RECONCILIATION] Error in periodic sync: {e}")

    async def _execute_approved_intent(self, event):
        """Processes intents that have survived the Meta-Arbitrator conflict resolution."""
        current_price = self._get_validated_price(event.symbol)
        if not current_price: return
        
        if self.risk_manager:
            # 🛡️ CAPA 7: Registro Omnisciente (Pre-Flight Axiom Check)
            try:
                from core.omniscient_registry import registry
                trade_risk_pct = getattr(event, 'sl_pct', 0.0) or getattr(event, 'sl_pct_max', 0.02)
                current_heat = 0.0
                if self.portfolio:
                    total_eq = max(self.portfolio.get_total_equity(), 1.0)
                    for pos in self.portfolio.positions.values():
                        current_heat += (abs(pos.get('quantity', 0)) * pos.get('current_price', 0)) / pos.get('leverage', 10) / total_eq
                is_exit = (getattr(event, 'signal_type', None) == SignalType.EXIT) or str(getattr(event, 'signal_type', None)) == 'SignalType.EXIT' or getattr(event, 'is_exit', False) or (getattr(event, 'strategy_id', '') == "FLIP_EXIT")
                has_sl = True if is_exit else bool(getattr(event, 'sl_pct', 0.0) > 0)
                
                # Check absolute axioms
                # Exit signals bypass Capa 7 check_trade_validity heat/sl checks
                if not is_exit and not registry.check_trade_validity(trade_risk_pct, current_heat, has_sl):
                    logger.warning(f"🛑 [CAPA 7] VETO ABSOLUTO de Registro Omnisciente para {event.symbol}.")
                    self.metrics['discarded_events'] += 1
                    return
            except Exception as e:
                logger.error(f"Error en Capa 7 (OmniscientRegistry): {e}")

            logger.info(f"🛡️ [ENGINE] Handing APPROVED intent to RiskManager for {event.symbol}")
            
            # 🔭 AITS SHADOW BRIDGE OBSERVATION
            if getattr(self, 'shadow_bridge', None):
                try:
                    sig_dict = {
                        "symbol": event.symbol,
                        "side": "BUY" if "LONG" in str(event.signal_type) else ("SELL" if "SHORT" in str(event.signal_type) else "EXIT"),
                        "quantity": getattr(event, 'suggested_quantity', 0.0),
                        "price": current_price,
                        "confidence": getattr(event, 'confidence', 0.0),
                        "horizon": getattr(event, 'horizon', 'SCALPING')
                    }
                    acct_state = {"total_capital": 13.0, "equity": 13.0, "open_positions": 0}
                    if self.portfolio:
                        acct_state["equity"] = getattr(self.portfolio, 'total_equity', 13.0)
                        acct_state["open_positions"] = len(getattr(self.portfolio, 'positions', {})) + len(getattr(self.portfolio, 'virtual_ledger', {}))
                    self.shadow_bridge.observe(sig_dict, acct_state)
                except Exception as e:
                    logger.error(f"[SHADOW] Observation error: {e}")
                    
            order_event = self.risk_manager.generate_order(event, current_price)
            if order_event:
                logger.info(f"🚀 [ENGINE] Order Generated by RiskManager for {event.symbol}: {order_event.quantity} {order_event.side}")
                dt_ms = (time.time_ns() - event.timestamp_ns) / 1_000_000
                latency_monitor.track('signal_to_order', dt_ms)
                self.events.put(order_event)
            else:
                 logger.warning(f"🛑 [ENGINE] RiskManager REJECTED signal for {event.symbol}")
                 # INTERACTION MONITOR: Order blocked by Risk Manager
                 try:
                     # AUDIT FIX: Guard against None when interaction_monitor unavailable
                     if _INTERACTION_MONITOR_AVAILABLE and get_interaction_monitor:
                         strat_id = getattr(event, 'strategy_id', 'Strategy')
                         get_interaction_monitor().log_interaction(
                             source=strat_id,
                             action="Signal Rejected",
                             details=f"RiskManager rejected signal for {event.symbol}"
                         )
                 except Exception as e:
                     logger.debug(f"Interaction monitor logging failed: {e}")

    @omniscient_trace(layer="CORTEX")
    async def _process_order_event(self, event):
        """Process ORDER event asynchronously"""
        if self.execution_handler:
            if asyncio.iscoroutinefunction(self.execution_handler.execute_order):
                await self.execution_handler.execute_order(event)
            else:
                self.execution_handler.execute_order(event)
        else:
            logger.warning("No Execution Handler registered. Order ignored.")

    @omniscient_trace(layer="CORTEX")
    async def _process_fill_event(self, event):
        """Process FILL event asynchronously"""
        dt_ms = (time.time_ns() - event.timestamp_ns) / 1_000_000
        latency_monitor.track('e2e_signal_to_fill', dt_ms)
        
        if self.portfolio:
            result = self.portfolio.update_fill(event)
            if result is not None:
                # OMEGA MIND: Unpack result and outcome
                # Legacy support: if portfolio returns just pnl (float), handle it
                if isinstance(result, tuple):
                    pnl, trade_outcome = result
                else:
                    pnl = result
                    trade_outcome = 1.0 if pnl > 0 else 0.0 # Fallback for old strategies
                    
                # 🔭 AITS SHADOW BRIDGE OUTCOME UPDATE
                if getattr(self, 'shadow_bridge', None):
                    try:
                        out_str = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "FLAT")
                        self.shadow_bridge.update_outcome(event.symbol, out_str, pnl)
                    except Exception as e:
                        logger.debug(f"Shadow bridge outcome update failed: {e}")

                for strategy in self.strategies:
                    # Filtramos por símbolo para asegurar que actualizamos la instancia correcta
                    if getattr(strategy, 'symbol', None) == event.symbol and \
                       hasattr(strategy, 'update_recursive_weights'):
                        strategy.update_recursive_weights(trade_outcome)

                # ════════════════════════════════════════════════════════════════
                # CTOS PHASE 2: FEEDBACK LOOP + EVENT BUS FILL PUBLISH
                # QUÉ: Alimenta el FeedbackProcessor con el resultado del trade
                #   y publica el fill al EventBus para suscriptores downstream.
                # POR QUÉ: Sin esto, el sistema NUNCA aprende de sus trades.
                #   FeedbackProcessor analiza slippage, WR por régimen, y calidad
                #   de exits → emite ajustes adaptativos.
                # PARA QUÉ: Cerrar el loop EXEC→RESULT→LEARNING→ADJUSTMENT.
                # CÓMO: Usa _last_closed_trade_data del portfolio para contexto.
                # CUÁNDO: Después de que portfolio procesó el fill y las estrategias
                #   actualizaron pesos. No bloquea el path crítico.
                # DÓNDE: core/engine.py → _process_fill_event()
                # QUIÉN: Engine (emisor) → FeedbackProcessor (receptor)
                # ════════════════════════════════════════════════════════════════
                try:
                    if pnl is not None and pnl != 0:
                        _closed = getattr(self.portfolio, '_last_closed_trade_data', None) if self.portfolio else None
                        feedback_processor.process_fill_outcome(
                            fill_event=event,
                            pnl=pnl,
                            strategy_id=_closed.get('strategy_id', 'Unknown') if _closed else 'Unknown',
                            entry_price=_closed.get('entry_price', 0.0) if _closed else 0.0,
                            duration_seconds=_closed.get('duration_seconds', 0.0) if _closed else 0.0,
                        )
                except Exception as _fb_e:
                    logger.debug(f"[FEEDBACK] Fill processing skipped: {_fb_e}")
                
                # Publish to EventBus FILLS channel
                try:
                    event_bus.publish(EventChannel.FILLS, {
                        'symbol': event.symbol,
                        'pnl': pnl,
                        'strategy_id': getattr(event, 'strategy_id', 'Unknown'),
                        'horizon': getattr(event, 'horizon', 'SCALPING'),
                        'timestamp_ns': time.time_ns(),
                    })
                except Exception as e:
                    logger.error(f"EventBus publish FILLS error: {e}")

        
        if self.order_manager and hasattr(event, 'order_id') and event.order_id:
            if asyncio.iscoroutinefunction(self.order_manager.remove_order):
                await self.order_manager.remove_order(event.order_id, event=event)
            else:
                self.order_manager.remove_order(event.order_id, event=event)

        # 🧹 [ORPHAN CLEANER] Check if position closed to cancel hanging orders
        try:
            if self.portfolio and self.execution_handler:
                horizon = getattr(event, 'horizon', 'SCALPING')
                pos = self.portfolio.get_horizon_position(event.symbol, horizon)
                if not pos or abs(pos.get('quantity', 0.0)) < 1e-8:
                    # position_cleaner imported at top-level
                    if _POSITION_CLEANER_AVAILABLE and position_cleaner:
                        asyncio.create_task(position_cleaner.clean_orphan_orders(self.execution_handler, event.symbol))
        except Exception as e:
            logger.error(f"Error checking position for orphan cleanup: {e}")

    # ==================================================================
    # HELPER METHODS
    # ==================================================================

    def _get_current_market_regime(self) -> str:
        """Determines current market regime (TRENDING_BULL, TRENDING_BEAR, RANGING, HIGH_VOLATILITY)
        CTOS PHASE 2: Syncs result to global_state.market_regime.
        """
        _regime = 'UNKNOWN'
        
        # 1. Trust Risk Manager first (centralized analysis)
        if self.risk_manager and hasattr(self.risk_manager, 'current_regime'):
             _regime = self.risk_manager.current_regime
        else:
            # 2. Fallback: Simple heuristic using BTC proxy
            # This prevents 'RANGING' deadlock if risk manager isn't updating
            try:
                if self.data_handlers:
                    dh = self.data_handlers[0]
                    bars = dh.get_latest_bars('BTC/USDT', n=50)
                    if bars and len(bars) >= 20:
                         close_prices = [b['close'] for b in bars]
                         # 1. Calcular Retornos y Volatilidad (Desviación Estándar)
                         returns = [(close_prices[i] - close_prices[i-1])/close_prices[i-1] for i in range(1, len(close_prices))]
                         mean_ret = sum(returns) / len(returns) if returns else 0.0
                         var_ret = sum((r - mean_ret)**2 for r in returns) / len(returns) if returns else 0.0
                         std_ret = var_ret**0.5
                         
                         # 2. Calcular Drift (Dirección del Precio)
                         drift = (close_prices[-1] - close_prices[0]) / close_prices[0] if close_prices[0] > 0 else 0.0
                         
                         # 3. Clasificación de Umbrales Seguros
                         if std_ret > 0.002:  # Volatilidad alta (~0.2% por barra)
                             _regime = 'HIGH_VOLATILITY'
                         elif drift > 0.003: # Tendencia alcista > 0.3%
                             _regime = 'TRENDING_BULL'
                         elif drift < -0.003: # Tendencia bajista < -0.3%
                             _regime = 'TRENDING_BEAR'
                         else:
                             _regime = 'RANGING'
            except Exception as e:
                logger.error(f"🛑 [FATAL] Fallback regime calculation failed: {e}")
        
        # CTOS PHASE 2: Sync regime to SSOT (single write point)
        if _regime != 'UNKNOWN':
            global_state.market_regime = _regime
        
        return _regime

    def _should_strategy_run(self, strategy, event, regime: str) -> bool:
        """
        Coordination Logic:
        - Prevents conflicting strategies
        - Enforces regime compatibility
        """
        strat_name = strategy.__class__.__name__
        
        # 0. Symbol Matching (Rule 4.2)
        # PROFESSOR METHOD: No procesar eventos de otros símbolos para reducir latencia.
        if hasattr(strategy, 'symbol') and strategy.symbol != event.symbol:
            return False

        # 0.5 Data Leakage Prevention (Repainting FIX)
        # QUÉ: Bloquea la evaluación si la vela no está cerrada, salvo que la estrategia
        #   esté explícitamente diseñada para operar intraday (HFT/Tick data).
        # POR QUÉ: Las estrategias evalúan el índice [-1]. Si la vela está viva,
        #   el RSI, MACD o ML model repinta y da señales falsas.
        if hasattr(event, 'is_closed') and not getattr(event, 'is_closed', True):
            if not getattr(strategy, 'handles_tick_data', False):
                return False


        # 1. Regime Compatibility
        if 'Statistical' in strat_name:
            # Mean reversion is dangerous in strong trends
            if 'TRENDING' in regime:
                 return False
                 
        if 'ML' in strat_name:
            # ML typically trained for trends
            if regime == 'CHOPPY':
                # Optional: reduce frequency or block
                pass
                
        # 2. Existing Position Check (Virtual Ledger Isolated)
        if self.portfolio and hasattr(self.portfolio, 'has_position_for_horizon'):
            horizon = getattr(strategy, 'horizon', 'SCALPING')
            if self.portfolio.has_position_for_horizon(event.symbol, horizon):
                # If we have a position IN THIS HORIZON, only allow strategies that manage exits or pyramids
                # For simplicity in this engine, we let them run but RiskManager filters adds
                pass

        return True

    def _update_portfolio_prices(self):
        """Update market prices ONLY for symbols with open positions"""
        if not self.portfolio or not self.data_handlers:
            return
            
        # FORENSIC FIX #16: Use virtual_ledger to find ALL active symbols
        # aggregate positions can miss symbols where LONG + SHORT net to 0
        active_symbols = list(set(
            v_key.split('_')[0] for v_key, pos in self.portfolio.virtual_ledger.items()
            if abs(pos.get('quantity', 0)) > 1e-8
        ))
        
        if not active_symbols:
            return

        dh = self.data_handlers[0]
        for symbol in active_symbols:
            try:
                bars = dh.get_latest_bars(symbol, n=1)
                if bars:
                    self.portfolio.update_market_price(symbol, bars[-1]['close'])
            except Exception:
                continue

    def _get_validated_price(self, symbol: str) -> Optional[float]:
        """
        Get and validate current price.
        
        LOW-LATENCY OPTIMIZATION:
        QUÉ: Fast-path usando global_state antes de acceder al data handler.
        POR QUÉ: global_state ya tiene el precio actualizado por _process_market_event
          vía update_from_market_event(). Acceder al data handler requiere:
          1. Adquirir self._data_lock (lock contention)
          2. Extraer datos del ring buffer Numba
          3. Crear numpy structured array nuevo (allocation)
          Total: ~100-500μs por llamada.
        PARA QUÉ: Reducir latencia de validación de señales.
        CÓMO: Leer de global_state primero (O(1), sin lock); fallback a data handler.
        """
        # ⚡ FAST PATH: global_state already has latest price (updated by _process_market_event)
        try:
            state = global_state.get_state(symbol)
            if state:
                price = state.get('close', 0.0)
                if price and price > 0:
                    return float(price)
        except Exception:
            pass  # Fall through to slow path
        
        # 🐢 SLOW PATH: Full data handler validation (original logic)
        if not self.data_handlers:
            return None
            
        try:
            dh = self.data_handlers[0]
            # Fetch recent bars to validate
            bars = dh.get_latest_bars(symbol, n=3)
            
            if not bars:
                return None
                
            current_price = bars[-1]['close']
            
            if current_price <= 0:
                return None
                
            # Basic anomaly check (spike detection)
            if len(bars) >= 2:
                prev_price = bars[-2]['close']
                if prev_price > 0:
                    pct_change = abs(current_price - prev_price) / prev_price
                    if pct_change > 0.15: # >15% jump in one timeframe is suspicious
                        logger.warning(f"Price anomaly detected for {symbol}: {pct_change*100:.1f}% jump")
                        return None
                        
            return current_price
            
        except Exception as e:
            logger.error(f"Validation error for {symbol}: {e}")
            return None

    def _validate_signal_ttl(self, event) -> bool:
        """Check if signal has expired based on its absolute expiration_timestamp"""
        now = datetime.now(timezone.utc)
        
        expiration = getattr(event, 'expiration_timestamp', None)
        if expiration:
            if now > expiration:
                logger.warning(f"Discarding EXPIRED signal {event.symbol} (Expired at: {expiration}, Current time: {now})")
                return False
        else:
            age = (now - event.datetime).total_seconds()
            if age > Config.MAX_SIGNAL_AGE:
                if age > 5.0: # Log only significant delays
                     logger.warning(f"Discarding STALE signal {event.symbol} (Age: {age:.2f}s)")
                return False
            
        return True

    def _sophia_veto_filter(self, event) -> bool:
        """
        🧠 SOPHIA-GLOBAL: Probabilistic Signal Quality Filter.
        
        QUÉ: Examina la metadata de la señal para validar que Sophia
          (o el sistema ML) haya asignado una win_probability suficiente.
        POR QUÉ: Señales de baja calidad queman capital en fees + slippage.
        PARA QUÉ: Proteger la micro-cuenta $13 de señales ruidosas.
        CÓMO:
          1. Si la señal ya tiene sophia.win_probability → usa ese valor.
          2. Si NO tiene (MLStrategy, Sniper, Statistical) → usa
             ml_confidence si disponible, o pasa sin veto.
          3. EXIT signals SIEMPRE pasan (protección de capital).
        CUÁNDO: Antes del RiskManager en _process_signal_event().
        DÓNDE: core/engine.py → Engine._sophia_veto_filter()
        QUIÉN: Engine (orquestador).
        
        Returns:
            True = signal passes, False = signal vetoed.
        """
        # EXIT signals always pass — never block a protective close
        is_exit = hasattr(event, 'signal_type') and 'EXIT' in str(event.signal_type)
        if is_exit:
            return True
        
        if not self.sophia_intelligence:
            return True  # If Sophia not available, pass through
        
        meta = getattr(event, 'metadata', None) or {}
        sophia_data = meta.get('sophia', {})
        strategy_id = getattr(event, 'strategy_id', 'Unknown')
        symbol = getattr(event, 'symbol', '???')
        
        # 1. Check Sophia win_probability (set by TechnicalStrategy)
        win_prob = sophia_data.get('win_probability', None)
        
        # 2. Fallback: Check ml_confidence (set by MLStrategy)
        if win_prob is None:
            win_prob = getattr(event, 'ml_confidence', None)
        
        # 3. Fallback: Check generic confidence
        if win_prob is None:
            win_prob = getattr(event, 'confidence', None)
        
        # 4. If no probability available at all, allow through
        #    (Strategy didn't produce confidence → trust RiskManager's PREDICTION_GATE)
        if win_prob is None:
            return True
        
        # 5. Veto if below threshold
        if win_prob < self.SOPHIA_MIN_CONFIDENCE:
            logger.warning(
                f"🧠 [SOPHIA-VETO] {symbol} signal REJECTED | "
                f"Strategy: {strategy_id} | P(Win)={win_prob:.1%} < "
                f"{self.SOPHIA_MIN_CONFIDENCE:.0%} threshold"
            )
            if self.risk_manager:
                try:
                    from risk.risk_manager import RejectionReason
                    self.risk_manager._reject_trade(event, RejectionReason.SOPHIA_VETO)
                except Exception as e:
                    logger.debug(f"Could not log SOPHIA_VETO rejection to RiskManager: {e}")
            return False
        
        # 6. Log approved signal
        logger.info(
            f"🧠 [SOPHIA-OK] {symbol} | Strategy: {strategy_id} | "
            f"P(Win)={win_prob:.1%} ✓"
        )
        return True

    def stop(self):
        self.running = False

        # Phase 3: Hardware Optimization (GC Tuner)
        for strategy in self.strategies:
            try:
                strategy.stop()
            except Exception as e:
                logger.error(f"Error stopping strategy {getattr(strategy, 'symbol', 'Unknown')}: {e}")
                
        if hasattr(self, '_tpool') and self._tpool:
            try:
                self._tpool.shutdown(wait=True)
            except Exception as e:
                logger.error(f"Error shutting down Engine thread pool: {e}")

    def _on_mutation_event(self, payload: Dict[str, Any]):
        """
        🧬 Applies mutated genes to global Config dynamically.
        """
        try:
            genes = payload.get('mutated_genes', {})
            if not genes: return
            
            from config import Config
            # Update Risk Parameters
            if 'tp_pct' in genes:
                Config.Risk.RISK_THRESHOLDS['take_profit_base'] = genes['tp_pct']
            if 'sl_pct' in genes:
                Config.Risk.RISK_THRESHOLDS['stop_loss_base'] = genes['sl_pct']
                
            # Update Strategy Parameters
            if 'adx_threshold' in genes:
                Config.Strategies.ML_THRESHOLDS['trending_min_adx'] = genes['adx_threshold']
                
            logger.info(f"🧬 [MUTATION APPLIED] New Config parameters injected live! ({len(genes)} genes)")
        except Exception as e:
            logger.error(f"Error applying live mutation: {e}")


    def _trigger_live_mutation(self):
        """
        🧬 PHASE 3: Integración del Motor Evolutivo (Live Mutation).
        Evaluates recent trades and mutates Config parameters dynamically.
        """
        if not self.evolution_engine: return
        
        try:
            from core.feedback_processor import feedback_processor
            from core.evolution import FitnessCalculator
            outcomes = feedback_processor.get_recent_outcomes(n=50)
            if len(outcomes) < 10:
                logger.debug("🧬 [LIVE MUTATION] Not enough trades to evaluate fitness.")
                return 
                
            # Simulate a Genotype evaluation
            score = FitnessCalculator.calculate_fitness(outcomes)
            
            # Use current Config as base Genotype
            from config import Config
            current_genes = Genotype(symbol="LIVE_CONFIG")
            current_genes.genes['tp_pct'] = Config.Risk.RISK_THRESHOLDS.get('take_profit_base', 0.015)
            current_genes.genes['sl_pct'] = Config.Risk.RISK_THRESHOLDS.get('stop_loss_base', 0.02)
            current_genes.genes['adx_threshold'] = Config.Strategies.ML_THRESHOLDS.get('trending_min_adx', 25)
            
            # Mutate
            mutated_genes = self.evolution_engine.mutate(current_genes)
            
            # Publish Mutation to EventBus
            from core.event_bus import event_bus, EventChannel
            event_bus.publish(EventChannel.MUTATION, {
                'source': 'EvolutionEngine',
                'fitness_score': score,
                'mutated_genes': mutated_genes.genes,
                'timestamp_ns': time.time_ns()
            })
            
            logger.info(f"🧬 [LIVE MUTATION] Triggered! Fitness: {score:.2f} | Mutated Genes: {list(mutated_genes.genes.keys())[:3]}")
        except Exception as e:
            logger.error(f"Live Mutation Error: {e}")

