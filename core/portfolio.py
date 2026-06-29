
import os
import time
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone


from core.enums import TradeDirection, TradeStatus, EventType, OrderSide, OrderType, SignalType
from core.events import SignalEvent
from core.reward_system import TradeOutcome
from core.omniscient_tracer import omniscient_trace
from utils.logger import logger
from utils.notifier import Notifier
from data.database import DatabaseHandler
from utils.data_manager import safe_append_csv, safe_read_csv
from utils.atomic_guard import AtomicGuard
from core.state_manager import AtomicStateManager # Phase 27
from sophia.post_mortem import PostMortemComparator, PostMortemResult  # SOPHIA-INTELLIGENCE Protocol
from sophia.nemesis import NemesisEngine  # NÉMESIS-RETROSPECCIÓN Protocol
from utils.axioma_math import PrecisionAuditor  # CRITERIO-AXIOMA Protocol
from core.meta_optimizer import meta_optimizer # Phase 46: Sovereign Meta-Predictor
from core.compounding_engine import get_compounding_engine  # Phase 7 AITS: Dynamic Allocation

from typing import Dict, Any, Optional, Tuple

from config import Config
from utils.debug_tracer import trace_execution
from decimal import Decimal, getcontext

try:
    from core.nano_core import calculate_kelly_fraction, calculate_unrealized_pnl_fast, update_hwm_lwm
except ImportError:
    calculate_kelly_fraction = None
    calculate_unrealized_pnl_fast = None
    update_hwm_lwm = None

try:
    from core.nano_portfolio import NanoVirtualLedger
except ImportError:
    NanoVirtualLedger = None

class Portfolio:
    def __init__(self, initial_capital: float = 13.0, 
                 csv_path: str = "dashboard/data/trades.csv", 
                 status_path: str = "dashboard/data/status.csv", 
                 auto_save: bool = True):
        self.initial_capital = initial_capital
        self.current_cash = initial_capital
        self.pending_cash = 0.0  # Cash reserved for pending orders
        self.used_margin = 0.0   # Margin locked in Futures positions
        self._pending_reservations = {} # OrderID -> reserved_amount (AEGIS-V15 Atomic Tracking)
        
        # [FASE 4: SWEEP DE COLATERAL] Ganancias de Microscalping para inyectar en Swing
        self.swept_micro_profits = 0.0
        
        # [FASE 5: AUTO-SECUESTRO ANTI-CISNE NEGRO]
        self.base_initial_capital = initial_capital
        self.initial_risk_secured = False
        self.total_secured_capital = 0.0
        
        # [PRECISION-AXIOMA]
        self.precision_drift_accumulated = Decimal('0.0')
        getcontext().prec = 28 # Satoshi-level precision for drift auditing
        
        # 🛡️ SOVEREIGN CONTEXT MEMORY (Removed ZMQ - Native dictionaries for speed)
        self.positions = {} # Symbol -> {'quantity': 0, 'avg_price': 0, 'current_price': 0}
        
        # FASE 30: MULTIPROCESSING SHARED MEMORY (Heat & NetExposure)
        # QUÉ: Mover estados críticos a memoria compartida a nivel de OS (C-level array).
        # POR QUÉ: Acceso en nanosegundos desde múltiples procesos (Ej. RiskManager, HFT Executor)
        # PARA QUÉ: Evitar overhead de locks/ZMQ en comprobaciones de calor de cuenta ultra-rápidas.
        from multiprocessing import shared_memory
        try:
            self._shm = shared_memory.SharedMemory(name="portfolio_critical_state", create=True, size=16) # 2 floats * 8 bytes (float64)
        except FileExistsError:
            self._shm = shared_memory.SharedMemory(name="portfolio_critical_state", create=False)
        # [0] = Heat (0.0 to 1.0)
        # [1] = NetExposure (Total USDT exposed)
        self.critical_state_shm = np.ndarray((2,), dtype=np.float64, buffer=self._shm.buf)
        self.critical_state_shm[:] = [0.0, 0.0]
        
        # OMNIBUS VIRTUAL LEDGER
        # Tracks true Avg Entry Price separately per Horizon (e.g. BTC/USDT_SCALP).
        # Prevents high-frequency strategies from being overwritten by Swing entries.
        self.virtual_ledger = {} # f"{symbol}_{horizon}" -> position_dict
        
        # [FASE 4: NANO OPTIMIZATION]
        if NanoVirtualLedger is not None:
            self._nano_ledger = NanoVirtualLedger(self.virtual_ledger, self.positions)
        else:
            self._nano_ledger = None
        
        # SUPREMO-V4: CANNIBALIZATION GUARD (VIRTUAL LEDGER SYNC)
        # Tracks net intended position per symbol across all horizons to avoid
        # paying double fees/margin when horizons have opposite directions.
        self._net_intended_positions = {} # {symbol: net_qty}
        
        # 📂 FORENSIC AUDITING: Isolated Ledgers
        self.scalping_ledger = []
        self.swing_ledger = []
        self.active_scalping_trades = []
        self.active_swing_trades = []
        
        # CTOS Phase 3: Session-level balance tracking for accurate reporting
        self._session_start_equity = initial_capital  # Equity when session started
        self._pre_trade_equity = initial_capital  # Fallback
        self._pre_trade_equity_map = {} # trade_id -> Equity when trade opened
        self._session_net_pnl = 0.0  # Accumulated net PnL this session
        self._daily_growth_target = 0.0473  # 4.73% daily for 100% in 15 days
        self._chronicle_tick_counters = {}  # {trade_id: tick_count} for chronicle logging
        
        self.realized_pnl = 0.0
        self.total_fees_paid = 0.0  # CRITERIO-AXIOMA: Explicit fee tracking
        self.trade_history = []     # Centralized history for Meritocratic Sizing (Phase 3.9)
        self._last_prices = {}       # Private price cache for RiskManager compatibility (Phase 3.13)
        
        # STRATEGY ATTRIBUTION: Track PnL per strategy
        # Format: {strategy_id: {'pnl': 0.0, 'wins': 0, 'losses': 0, 'trades': 0}}
        self.strategy_performance = {}
        
        # ═══════════════════════════════════════════════════════════════
        # FORENSIC-V35: SESSION-LEVEL WIN RATE TRACKING
        # QUÉ: Contadores de wins/losses para la SESIÓN ACTUAL.
        # POR QUÉ: El WR acumulado de por vida (strategy_performance) mostraba
        #   90% WR después de un trade perdedor porque las sesiones anteriores
        #   tenían wins. Esto es engañoso y peligroso.
        # PARA QUÉ: Telegram muestra WR de sesión (preciso) como principal,
        #   WR all-time como secundario.
        # CÓMO: Contadores separados reseteados en cada sesión.
        # ═══════════════════════════════════════════════════════════════
        # SOPHIA-GLOBAL FIX: Session stats always start fresh on Portfolio init.
        # This ensures WR=0% until the FIRST trade closes (not inherited from crashes).
        self._session_stats = {
            'wins': 0,
            'losses': 0,
            'total': 0,
            'net_pnl': 0.0,
            'gross_pnl': 0.0,
            'start_time': datetime.now(timezone.utc).isoformat(),
        }
        # ⚡ FASE 8: ANTI-MARTINGALE STREAK COUNTERS
        # QUÉ: Trackers de rachas consecutivas para sizing exponencial.
        # POR QUÉ: Sin esto, el Kelly calcula un promedio plano. Con rachas,
        #   escalamos el tamaño en wins consecutivos (crecimiento exponencial)
        #   y reducimos en losses consecutivos (protección de capital).
        self._win_streak = 0
        self._loss_streak = 0
        self._max_win_streak = 0
        
        # PHASE 14: Dynamic Kelly Criterion Tracking
        self.kelly_trades_history = []  # List of dicts: {'pnl': float, 'is_win': bool}
        self.kelly_winrate = 0.0
        self.kelly_payoff_ratio = 1.0
        self.KELLY_WINDOW = 20
        
        self.csv_path = csv_path
        self.status_path = status_path
        self.auto_save = auto_save
        
        # Phase 50: Atomic Metal-Core Protection
        self.guard = AtomicGuard()
        
        # Isolated state caches (Fast Access)
        self._equity_cache = initial_capital
        self._last_snapshot = None
        self._last_snapshot_ts = 0
        
        self.io_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="PortfolioIO")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        
        # Initialize CSVs
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w') as f:
                f.write('datetime,symbol,type,direction,quantity,price,fill_cost,strategy_id,setup_type,strategy_version,details\n')
            
        # Create initial status file
        self.save_status()
        
        # Initialize Database
        self.db = DatabaseHandler()
        
        # Phase 6: Math Stats Tracking
        self.math_stats = {
            'hurst': 0.5,
            'beta': 1.0,
            'half_life': 0,
            'last_update': None
        }
        # Phase 7: Meta-Brain Stats
        self.strategy_rankings = {}
        
        # Phase 5 (Macro): Relative Strength Tracking
        self.relative_strength_scores = {}  # {symbol: float}
        self.last_rs_update = 0
        
        # SOPHIA-INTELLIGENCE: Post-Mortem Calibration Tracker
        self.sophia_post_mortem = PostMortemComparator(rolling_window=100)
        
        # NÉMESIS-RETROSPECCIÓN: Deep Post-Mortem Autopsy Engine
        self.nemesis_engine = NemesisEngine()
        self._nemesis_sophia_linked = False  # C-1 FIX: Track if feedback loop is connected
        
        # AEGIS-V16: Clock Sync
        self.data_provider = None

    def close(self):
        try:
            self.save_status()
            # FASE 30: Limpiar memoria compartida
            if hasattr(self, '_shm'):
                self._shm.close()
                self._shm.unlink()
        except Exception as e:
            logger.error(f"Error saving final status: {e}")

    def _get_current_time(self):
        """Devuelve el tiempo simulado (backtest) o el tiempo real (producción)."""
        if hasattr(self, 'data_provider') and self.data_provider and hasattr(self.data_provider, 'current_time_ms'):
            try:
                ms = self.data_provider.current_time_ms
                return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
            except Exception as e:
                logger.debug(f"Silent exception caught: {e}")
        return datetime.now(timezone.utc)
        
    def link_nemesis_to_sophia(self, sophia_instance):
        """
        🔗 C-1 FIX: Connect Némesis to Sophia for closed-loop feedback.
        
        QUÉ: Enlaza la instancia de Sophia con NemesisEngine para activar el feedback loop.
        POR QUÉ: Sin este enlace, apply_nemesis_feedback() nunca recibe datos reales y
                 Sophia opera en modo open-loop permanente, sin aprender de errores.
        PARA QUÉ: Cerrar el bucle adaptativo: Trade→Diagnóstico→Ajuste→Mejora continua.
        CÓMO: Se llama desde la estrategia al primer signal, o desde engine bootstrap.
        CUÁNDO: Una sola vez durante el lifecycle del sistema.
        DÓNDE: core/portfolio.py → Portfolio
        QUIÉN: Strategies (technical/ml) o Engine durante bootstrap.
        """
        if not self._nemesis_sophia_linked and sophia_instance is not None:
            self.nemesis_engine.set_sophia_ref(sophia_instance)
            self._nemesis_sophia_linked = True
            logger.info("🔗 [C-1 FIX] Némesis→Sophia feedback loop CONNECTED")
        
    def get_atomic_snapshot(self) -> Dict[str, Any]:
        """
        Returns a thread-safe deep copy of the portfolio state.
        Guarantees internal consistency (e.g. Equity = Cash + PnL).
        """
        self.guard.acquire()
        try:
            # Snapshot critical fields
            snapshot = {
                'timestamp': self._get_current_time().isoformat(),
                'cash': float(self.current_cash),
                'realized_pnl': float(self.realized_pnl),
                'used_margin': float(self.used_margin),
                'pending_cash': float(self.pending_cash),
                'equity': float(self._equity_cache),
                'positions': {
                    sym: pos.copy() for sym, pos in self.positions.items()
                },
                'math_stats': self.math_stats.copy()
            }
            return snapshot
        finally:
            self.guard.release()

    def get_horizon_position(self, symbol: str, horizon: str, direction: str = None) -> Optional[Dict[str, Any]]:
        """
        Returns the isolated position for a specific trading horizon from the Virtual Ledger.
        Returns None if no active position exists for that horizon.
        🚀 PHOENIX V3: Hedge Mode Aware. Checks both LONG and SHORT isolated ledgers.
        """
        self.guard.acquire()
        try:
            if direction:
                direction_val = direction.name if hasattr(direction, 'name') else str(direction)
                dir_upper = direction_val.upper()
                exact_pos = self.virtual_ledger.get(f"{symbol}_{horizon}_{dir_upper}")
                if exact_pos and abs(exact_pos['quantity']) > 1e-8:
                    return exact_pos.copy()
                return None

            # Fallback legacy si no se provee dirección (no recomendado en Hedge Mode)
            long_pos = self.virtual_ledger.get(f"{symbol}_{horizon}_LONG")
            short_pos = self.virtual_ledger.get(f"{symbol}_{horizon}_SHORT")
            legacy_pos = self.virtual_ledger.get(f"{symbol}_{horizon}")
            
            valid_positions = []
            if long_pos and abs(long_pos['quantity']) > 1e-8:
                valid_positions.append(long_pos)
            if short_pos and abs(short_pos['quantity']) > 1e-8:
                valid_positions.append(short_pos)
            if legacy_pos and abs(legacy_pos['quantity']) > 1e-8:
                valid_positions.append(legacy_pos)
                
            if not valid_positions:
                return None
                
            # If multiple exist sin dirección, devuelve la más grande (Peligro de Colisión)
            best_pos = max(valid_positions, key=lambda p: abs(p['quantity']))
            return best_pos.copy()
        finally:
            self.guard.release()
            
    def has_position_for_horizon(self, symbol: str, horizon: str) -> bool:
        """
        Checks if there is an active position for a specific trading horizon.
        """
        return self.get_horizon_position(symbol, horizon) is not None

    def update_math_stats(self, stats: Dict[str, Any]) -> None:
        """Update live mathematical statistics from strategies."""
        if not stats: return
        self.guard.acquire()
        try:
            self.math_stats.update(stats)
        finally:
            self.guard.release()
        # Optional: Auto-save if critical diff? 
        # For now, rely on standard loop save
        
    def restore_state_from_db(self):
        """
        Restore portfolio state (positions) from SQLite database.
        Used for crash recovery.
        """
        try:
            # Restore Positions
            db_positions = self.db.get_open_positions()
            if db_positions:
                self.positions = db_positions
                logger.info(f"🔄 RESTORED {len(self.positions)} active positions from DB.")
                
                for sym, pos in self.positions.items():
                    logger.info(f"   - {sym}: {pos['quantity']} @ ${pos['entry_price']:.4f}")
                    # Map entry_price to avg_price for internal consistency
                    if 'avg_price' not in pos:
                        pos['avg_price'] = pos['entry_price']
                        
                    # --- RECONSTRUCT VIRTUAL LEDGER ---
                    if '_' in sym:
                        v_key = sym
                    else:
                        horizon = pos['horizon']
                        pos_side = 'LONG' if pos['quantity'] > 0 else 'SHORT'
                        v_key = f"{sym}_{horizon}_{pos_side}"
                    
                    self.virtual_ledger[v_key] = {
                        'quantity': pos['quantity'],
                        'avg_price': pos['avg_price'],
                        'current_price': pos['current_price'],
                        'sl_pct': pos['sl_pct'],
                        'tp_pct': pos['tp_pct'],
                        'opener_strategy_id': pos['strategy_id'],
                        'entry_time': self._get_current_time() # Fallback
                    }
                    if getattr(self, '_nano_ledger', None) is not None:
                        self._nano_ledger.register_vkey(sym, v_key)
                    
                    sl_print = f"{float(pos['sl_pct'])*100:.2f}%" if pos['sl_pct'] else "None"
                    logger.info(f"   ↳ 🧬 Virtual Ledger Reconstructed: {v_key} (SL: {sl_print})")
            else:
                logger.info("✅ No active positions found in DB.")
                
            return True
        except Exception as e:
            logger.error(f"⚠️  Failed to restore portfolio state from DB: {e}")
            return False
        
    def load_portfolio_state(self, state_path):
        """
        Load portfolio state (positions, cash, pnl) from a JSON file.
        Used for crash recovery.
        """
        if not os.path.exists(state_path):
            return False
            
        try:
            from utils.fast_json import FastJson
            data = FastJson.load_from_file(state_path)
            if data is None: return False
                
            if data['status'] == 'OFFLINE':
                pass
                
            # Restore Cash & PnL
            self.current_cash = data['cash']
            self.realized_pnl = data['realized_pnl']
            self.used_margin = data['used_margin'] # Restore margin
            
            # Restore Positions
            loaded_positions = data['positions']
            if loaded_positions:
                self.positions = loaded_positions
                print(f"🔄 RESTORED {len(self.positions)} active positions from previous session.")
                
                loaded_vl = data['virtual_ledger']
                if loaded_vl:
                    self.virtual_ledger = loaded_vl
                    print(f"🔄 RESTORED {len(self.virtual_ledger)} virtual ledger entries natively.")
                    if getattr(self, '_nano_ledger', None) is not None:
                        for k in self.virtual_ledger.keys():
                            _sym = k.split('_')[0]
                            self._nano_ledger.register_vkey(_sym, k)
                else:
                    for sym, pos in self.positions.items():
                        print(f"   - {sym}: {pos['quantity']} @ ${pos['avg_price']:.4f}")
                        
                        # --- RECONSTRUCT VIRTUAL LEDGER ---
                        if '_' in sym:
                            v_key = sym
                        else:
                            horizon = pos['horizon']
                            pos_side = 'LONG' if pos['quantity'] > 0 else 'SHORT'
                            v_key = f"{sym}_{horizon}_{pos_side}"
                        
                        self.virtual_ledger[v_key] = {
                            'quantity': pos['quantity'],
                            'avg_price': pos['avg_price'],
                            'current_price': pos['current_price'],
                            'sl_pct': pos['sl_pct'],
                            'tp_pct': pos['tp_pct'],
                            'opener_strategy_id': pos['strategy_id'],
                            'entry_time': self._get_current_time() # Fallback
                        }
                        if getattr(self, '_nano_ledger', None) is not None:
                            self._nano_ledger.register_vkey(sym, v_key)
                        sl_print = f"{float(pos['sl_pct'])*100:.2f}%" if pos['sl_pct'] else "None"
                        print(f"   ↳ 🧬 Virtual Ledger Reconstructed: {v_key} (SL: {sl_print})")
            else:
                print("✅ No active positions to restore.")
                
            return True
        except Exception as e:
            print(f"⚠️  Failed to restore portfolio state: {e}")
            return False
            
    def check_systemic_risk(self):
        """
        [PHASE 20] Calculates Fleet Beta / Correlation (Immunology).
        If Correlation > 0.9, marks regime as SYSTEMIC_COLLAPSE.
        """
        try:
            # Lazy import to avoid circular dependency
            from core.data_handler import get_data_handler
            dh = get_data_handler()
            if not dh: return
            
            symbols = dh.symbol_list
            if len(symbols) < 3: return
            
            # Fetch returns for all symbols
            returns_map = {}
            for s in symbols:
                bars = dh.get_latest_bars(s, n=50) # Last 50m
                if bars is not None and len(bars) > 40:
                     # Access structured array 'close' via safe getter if possible, 
                     # but here we assume numpy array access from loader
                     # Loader returns structured array with 'close' field
                     closes = bars['close']
                     # Calculate returns
                     rets = np.diff(closes) / closes[:-1]
                     returns_map[s] = rets
            
            if not returns_map: return

            # Pad/Align (simplified: assumes sync or just takes min length)
            min_len = min(len(r) for r in returns_map.values())
            if min_len < 30: return
            
            # Slice to same length
            data = {s: r[-min_len:] for s, r in returns_map.items()}
            matrix = np.array(list(data.values())) # shape (N_symbols, min_len)
            corr_matrix = np.corrcoef(matrix)
            avg_corr = np.mean(corr_matrix)
            
            # Store in Stats
            self.math_stats['fleet_corr'] = avg_corr
            self.math_stats['last_update'] = time.time()
            
            if avg_corr > 0.9:
                logger.warning(f"☢️ [IMMUNOLOGY] HIGH SYSTEMIC RISK! Avg Corr: {avg_corr:.3f}")
                self.math_stats['systemic_risk'] = True
            else:
                 self.math_stats['systemic_risk'] = False
                 
        except Exception as e:
            logger.error(f"Systemic Risk Check Failed: {e}")
            
    def update_relative_strength(self):
        """
        [PHASE 5] Calculates Relative Strength (RSI + Momentum) across the fleet.
        [EVOLUCIÓN CUÁNTICA] Swarm Volatility Targeting.
        Calcula la Volatilidad Cuántica (ATR / Close) para direccionar el capital líquido.
        """
        import time
        now = time.time()
        if now - self.last_rs_update < 60: # Update every 1 min para HFT
            return
            
        try:
            if not hasattr(self, 'data_provider') or not self.data_provider: return
            dh = self.data_provider
            
            symbols = dh.symbols if hasattr(dh, 'symbols') else getattr(dh, 'symbol_list', [])
            if not symbols and hasattr(dh, 'get_all_symbols'): symbols = dh.get_all_symbols()
            if len(symbols) < 2: return
            
            rs_scores = {}
            for s in symbols:
                bars = dh.get_latest_bars(s, n=20) # 20 periods
                if bars is not None and len(bars) > 15:
                    closes = bars['close']
                    highs = bars['high']
                    lows = bars['low']
                    
                    # Calculate simple return
                    ret = (closes[-1] - closes[0]) / closes[0]
                    # Calculate RSI approx
                    diff = np.diff(closes)
                    gains = np.maximum(diff, 0).mean()
                    losses = np.maximum(-diff, 0).mean()
                    rs = gains / losses if losses > 0 else 0
                    rsi = 100 - (100 / (1 + rs)) if losses > 0 else 100
                    
                    # Calculate Quantum Volatility (ATR Proxy / Close)
                    tr = np.maximum(highs[1:] - lows[1:], np.abs(highs[1:] - closes[:-1]))
                    tr = np.maximum(tr, np.abs(lows[1:] - closes[:-1]))
                    atr = tr.mean()
                    quantum_volatility = (atr / closes[-1]) * 100 # % volatility
                    
                    # Score = Volatility * 50% + Return * 30% + RSI * 20%
                    score = (quantum_volatility * 50) + (abs(ret) * 100 * 30) + (rsi / 100 * 20)
                    rs_scores[s] = score
            
            if rs_scores:
                self.relative_strength_scores = rs_scores
                self.last_rs_update = now
                logger.debug(f"📊 [PORTFOLIO] Swarm Volatility Updated for {len(rs_scores)} symbols. Top Score: {max(rs_scores.values()):.2f}")
                
        except Exception as e:
            logger.error(f"Swarm Volatility Update Failed: {e}")
            
    def get_allocation_multiplier(self, symbol: str, is_long: bool) -> float:
        """
        [EVOLUCIÓN CUÁNTICA] Swarm Volatility Targeting.
        Retorna el multiplicador de asignación fraccional basado en el Swarm Ranking.
        El Activo Top 1 recibe un bono masivo para absorber todo el margen ($13 USD).
        """
        if not self.relative_strength_scores or symbol not in self.relative_strength_scores:
            return 1.0 # Default
            
        # Sort symbols by Swarm Score
        sorted_symbols = sorted(self.relative_strength_scores.keys(), key=lambda k: self.relative_strength_scores[k], reverse=True)
        total = len(sorted_symbols)
        if total == 0: return 1.0
        
        rank = sorted_symbols.index(symbol)
        percentile = 1.0 - (rank / total) # 1.0 is highest, 0.0 is lowest
        
        # 🎯 SWARM TARGETING (Capital Líquido Inteligente)
        if rank == 0:
            return 3.0 # Activo más caliente de Binance: +200% asignación (absorbe los 13 USD)
        if rank <= 2:
            return 1.5 # Top 3: +50%
        
        # Congelamos o minimizamos la asignación a activos estancados
        if percentile < 0.5:
            return 0.2 # Mitad inferior: -80% asignación (no inmovilizar capital)
            
        return 1.0

    def _get_available_cash_internal(self, horizon: str = None):
        # ═══════════════════════════════════════════════════════════════
        # FORENSIC FIX: SAFETY RECONCILIATION (Margin Leak Prevention)
        # QUÉ: Si NO hay posiciones abiertas en NINGÚN ledger, forzar
        #   used_margin=0 y pending_cash=0.
        # POR QUÉ: El pending_cash se incrementa en reserve_cash() pero
        #   en edge cases (rechazos intermedios, exits sin dollar_size,
        #   doble-booking) puede quedar locked → TotalAvail negativo
        #   → NINGÚN trade puede ejecutarse → cuenta muerta.
        # PARA QUÉ: Auto-reparación de estado corrupto sin intervención.
        # CÓMO: Contar posiciones en self.positions + self.virtual_ledger.
        #   Si ambos están vacíos, los valores DEBEN ser cero por definición.
        # CUÁNDO: Cada vez que se consulta available cash.
        # ═══════════════════════════════════════════════════════════════
        # ═══════════════════════════════════════════════════════════════
        # FORENSIC-V99: HFT O(1) STATE EVALUATION
        # QUÉ: Reemplazo de iteradores O(N) por evaluaciones booleanas/len O(1).
        # POR QUÉ: Las List Comprehensions consumían ciclos en el hot-path del Event Loop.
        # PARA QUÉ: Reducir latencia del Portfolio de microsegundos a nanosegundos.
        # ═══════════════════════════════════════════════════════════════
        has_physical = len(self.positions) > 0
        has_virtual = any(v['quantity'] != 0 for v in self.virtual_ledger.values()) if self.virtual_ledger else False
        
        if not has_physical and not has_virtual:
            if self.used_margin > 0.01 or self.pending_cash > 0.01 or len(self._pending_reservations) > 0:
                logger.warning(
                    f"🔧 [RECONCILE] No positions open but used_margin=${self.used_margin:.4f}, "
                    f"pending_cash=${self.pending_cash:.4f}, pending_reservations={len(self._pending_reservations)} → RESETTING to 0 (leak fix)"
                )
                self.used_margin = 0.0
                self.pending_cash = 0.0
                self._pending_reservations.clear()
        
        if Config.BINANCE_USE_FUTURES:
            total_avail = self.current_cash - self.used_margin - self.pending_cash
        else:
            total_avail = self.current_cash - self.pending_cash
            
        if horizon is None:
            return total_avail
            
        # ═══════════════════════════════════════════════════════════════
        # CAPA 3: PORTFOLIO CONSCIOUSNESS & INSTANT COMPOUNDING
        # QUÉ: Distribución dinámica de capital basada en mérito.
        # POR QUÉ: Evita límites estáticos rígidos y permite el flujo de
        #   liquidez instantáneo hacia las estrategias más rentables.
        # PARA QUÉ: Crecimiento compuesto exponencial verdadero.
        # ═══════════════════════════════════════════════════════════════
        from core.compounding_engine import get_compounding_engine
        engine = get_compounding_engine()
        equity = getattr(self, '_equity_cache', self.current_cash)
        micro_pct, scalp_pct, swing_pct = engine.get_3way_allocation(equity)
        
        # [FASE 4: SWEEP DE COLATERAL] 
        # Restamos las ganancias barridas del equity base para que el Compounding Engine
        # no las distribuya globalmente.
        base_equity = equity - self.swept_micro_profits
        
        if horizon == 'MICROSCALPING':
            alloc_pct = micro_pct
            allocated_total = base_equity * alloc_pct
        elif horizon == 'SCALPING':
            alloc_pct = scalp_pct
            allocated_total = base_equity * alloc_pct
        elif horizon == 'SWING':
            alloc_pct = swing_pct
            # TODO el dinero barrido se inyecta EXCLUSIVAMENTE a Swing
            allocated_total = (base_equity * alloc_pct) + self.swept_micro_profits
        else:
            alloc_pct = 1.0  # fallback
            allocated_total = equity * alloc_pct
        
        horizon_used = 0.0
        for v_key, pos in self.virtual_ledger.items():
            if pos['horizon'] == horizon:
                qty = abs(pos['quantity'])
                avg_price = pos['avg_price']
                if qty > 0 and avg_price > 0:
                    if Config.BINANCE_USE_FUTURES:
                        # AEGIS-V15: Usar el apalancamiento real de la posición
                        eff_lev = pos['leverage'] or Config.BINANCE_LEVERAGE
                        horizon_used += (qty * avg_price) / eff_lev
                    else:
                        horizon_used += (qty * avg_price)
                        
        # FORENSIC FIX: Track pending cash per horizon using order_id prefix
        if horizon == "MICROSCALPING":
            horizon_prefix = "MSC"
        elif horizon == "SCALPING":
            horizon_prefix = "SCL"
        else:
            horizon_prefix = "SWG"
            
        horizon_pending = sum(
            amt for oid, amt in self._pending_reservations.items() 
            if f"TG_{horizon_prefix}_" in oid
        )
        
        # Rigorous Partitioning: Available in this Silo = Allocated Total - Used - Pending
        horizon_avail = allocated_total - horizon_used - horizon_pending
        
        # [QUANTUM EVOLUTION: FASE 8] OMNI-MARGIN (Flujo Cruzado de Rentabilidad Flotante)
        # Unificamos el flujo de capital. TODO el PnL no realizado (flotante) positivo de CUALQUIER posición
        # se desbloquea en tiempo real al 100% para financiar nuevas posiciones (Cross-Margin Sintético).
        omni_float_pnl = 0.0
        for v_key, pos in self.virtual_ledger.items():
            qty = pos['quantity']
            if qty != 0:
                avg = pos['avg_price']
                curr = pos['current_price']
                if calculate_unrealized_pnl_fast:
                    direction = 1 if qty > 0 else -1
                    pnl = calculate_unrealized_pnl_fast(float(curr), float(avg), float(abs(qty)), direction)
                else:
                    pnl = (curr - avg) * qty
                if pnl > 0:
                    omni_float_pnl += pnl
        
        if omni_float_pnl > 0.05:
            horizon_avail += omni_float_pnl
            total_avail += omni_float_pnl
            logger.debug(f"🌌 [OMNI-MARGIN] Desbloqueando ${omni_float_pnl:.2f} USD de ganancia virtual cruzada al margen {horizon}.")

        
        # Eliminamos el Soft Cap (cascade priority) para forzar AISLAMIENTO ESTRICTO.
        # Un trade Swing no debe tocar nunca el dinero del Scalping y viceversa.
        
        # FORENSIC TRACE: Why is cash 0?
        if min(total_avail, horizon_avail) <= 2.0:
            logger.warning(f"🔍 [CASH-TRACE] Horizon {horizon} | TotalAvail: ${total_avail:.2f} | AllocTotal: ${allocated_total:.2f} | HorizonUsed: ${horizon_used:.2f} | HorizonPending: ${horizon_pending:.2f} | Result: ${min(total_avail, horizon_avail):.2f}")
            
        return max(0.0, float(min(total_avail, horizon_avail)))

    @trace_execution
    def get_available_cash(self, horizon: str = None):
        """Return cash available for trading, partitioned by horizon if specified."""
        self.guard.acquire()
        try:
            return self._get_available_cash_internal(horizon)
        finally:
            self.guard.release()

    @property
    def unrealized_pnl(self):
        """
        Calculate total unrealized PnL from all open positions using Atomic Guard.
        """
        pnl = 0.0
        self.guard.acquire()
        try:
            for v_key, pos in self.virtual_ledger.items():
                qty = pos['quantity']
                if qty != 0:
                    avg_price = pos['avg_price']
                    current_price = pos['current_price']
                    if calculate_unrealized_pnl_fast:
                        direction = 1 if qty > 0 else -1
                        pnl += calculate_unrealized_pnl_fast(float(current_price), float(avg_price), float(abs(qty)), direction)
                    else:
                        pnl += (current_price - avg_price) * qty
            return pnl
        finally:
            self.guard.release()

    def get_total_equity(self):
        """
        Return Total Equity = Cash + Unrealized PnL.
        SUPREMO-V3: Cached for O(1) read access.
        """
        # Periodic update of cache happens in update_market_price
        return self._equity_cache

    def _refresh_equity_cache(self):
        """
        Internal heavy calculation of equity for HEDGE MODE.
        Source of Truth: virtual_ledger (isolated horizons).
        """
        self.guard.acquire()
        try:
            equity = self.current_cash
            
            total_long_beta = 0.0
            total_short_beta = 0.0
            net_delta = 0.0
            beta_scalp = 0.0
            beta_swing = 0.0
            
            n = len(self.virtual_ledger)
            for v_key, pos in self.virtual_ledger.items():
                qty = pos['quantity']
                if qty != 0:
                    avg = pos['avg_price']
                    curr = pos['current_price']
                    
                    pos_beta = avg * abs(qty)
                    
                    if qty > 0:
                        total_long_beta += pos_beta
                        net_delta += pos_beta
                    else:
                        total_short_beta += pos_beta
                        net_delta -= pos_beta
                        
                    strat_id = str(pos.get('opener_strategy_id', 'Unknown')).upper()
                    horizon = pos['horizon']
                    if "SWING" in strat_id or horizon not in ["1m", "5m"]:
                        beta_swing += pos_beta
                    else:
                        beta_scalp += pos_beta
                        
                    if calculate_unrealized_pnl_fast:
                        direction = 1 if qty > 0 else -1
                        pnl = calculate_unrealized_pnl_fast(float(curr), float(avg), float(abs(qty)), direction)
                    else:
                        pnl = (curr - avg) * qty
                    equity += pnl

            self._equity_cache = equity
            self.math_stats['total_long_beta'] = total_long_beta
            self.math_stats['total_short_beta'] = total_short_beta
            self.math_stats['net_delta'] = net_delta
            
            if equity > 0:
                self.math_stats['heat_scalp'] = beta_scalp / equity
                self.math_stats['heat_swing'] = beta_swing / equity
            else:
                self.math_stats['heat_scalp'] = 0.0
                self.math_stats['heat_swing'] = 0.0
            
            # FASE 30: Actualizar estados críticos en Shared Memory
            # Heat: Qué % del portfolio está expuesto
            try:
                if equity > 0:
                    heat = (total_long_beta + total_short_beta) / equity
                else:
                    heat = 0.0
                self.critical_state_shm[0] = float(heat)
                self.critical_state_shm[1] = float(net_delta)
            except Exception as e:
                pass
            
            return equity
        finally:
            self.guard.release()
    
    def get_portfolio_exposure(self) -> Dict[str, float]:
        """Returns the cached global exposure metrics."""
        return {
            'TOTAL_LONG_BETA': self.math_stats['total_long_beta'],
            'TOTAL_SHORT_BETA': self.math_stats['total_short_beta'],
            'NET_DELTA': self.math_stats['net_delta'],
            'HEAT_SCALP': self.math_stats['heat_scalp'],
            'HEAT_SWING': self.math_stats['heat_swing']
        }

    def get_shm_state(self) -> np.ndarray:
        """
        [FASE 31] Exporta el buffer subyacente para el Backtester Vectorizado o Cython.
        """
        if hasattr(self, 'critical_state_shm'):
            return self.critical_state_shm
        return np.zeros(2, dtype=np.float64)

    @property
    def scalp_positions(self) -> Dict[str, Any]:
        """
        Devuelve estrictamente las posiciones del horizonte SCALPING.
        Evita cruces con Swing.
        """
        scalps = {}
        for k, v in self.virtual_ledger.items():
            if 'SCALPING' in k and abs(v['quantity']) > 1e-8:
                scalps[k] = v
        return scalps
        
    @property
    def swing_positions(self) -> Dict[str, Any]:
        """
        Devuelve estrictamente las posiciones del horizonte SWING.
        Evita cruces con Scalping.
        """
        swings = {}
        for k, v in self.virtual_ledger.items():
            if 'SWING' in k and abs(v['quantity']) > 1e-8:
                swings[k] = v
        return swings
    
    def reserve_cash(self, amount, horizon='SCALPING', order_id=None):
        """
        Reserva capital para una orden pendiente.
        🚀 AEGIS-V15: Seguimiento atómico por order_id.
        """
        self.guard.acquire()
        try:
            avail = self._get_available_cash_internal(horizon=horizon)
            if avail >= amount:
                amount_val = float(amount)
                self.pending_cash += amount_val
                
                # Rastreo atómico por ID para evitar leaks
                if order_id:
                    self._pending_reservations[order_id] = amount_val
                
                # [CASH-TRACE] Log reservation
                logger.debug(f"📜 [CASH-TRACE] RESERVE | Order: {order_id} | Amt: ${amount_val:.2f} | New Pending: ${self.pending_cash:.2f} | Horizon: {horizon}")
                return True
            logger.warning(f"⚠️ [RESERVE-FAIL] {horizon} requested ${amount:.2f} but only ${avail:.2f} available. Pending: {self.pending_cash}, Used: {self.used_margin}, Equity: {self.current_cash}")
            return False
        finally:
            self.guard.release()

    def release_order_margin(self, amount=None, order_id=None, skip_lock=False):
        """
        🚀 AEGIS-V15: Liberación atómica de margen basada en metadatos o ID.
        QUÉ: Reduce pending_cash exactamente el monto reservado.
        POR QUÉ: Evita fugas de capital cuando las órdenes se llenan o cancelan.
        PARA QUÉ: Mantener la liquidez de cuentas micro ($13).
        CÓMO: Prioriza el order_id para buscar en el mapa de reservaciones.
        """
        if not skip_lock: self.guard.acquire()
        try:
            amount_val = 0.0
            
            # 🚀 AEGIS-V16: Handle Swarm Grid IDs (e.g., TG_SCL_LONG_168812345_BTCUSDT_0)
            base_id = order_id
            if order_id and '_' in order_id and order_id.split('_')[-1].isdigit():
                possible_base = order_id.rsplit('_', 1)[0]
                if possible_base in self._pending_reservations:
                    base_id = possible_base
            
            # Prioridad 1: Segumiento por ID de Orden
            if base_id and base_id in self._pending_reservations:
                # Si viene un amount explícito y es un partial fill o grid, restamos
                if amount is not None and float(amount) > 0 and float(amount) < self._pending_reservations[base_id]:
                    amount_val = float(amount)
                    self._pending_reservations[base_id] -= amount_val
                    logger.debug(f"🎯 [CASH-TRACE] PARTIAL RELEASE BY ID | Base: {base_id} | Amt: ${amount_val:.2f} | Remaining: ${self._pending_reservations[base_id]:.2f}")
                else:
                    amount_val = self._pending_reservations.pop(base_id)
                    logger.debug(f"🎯 [CASH-TRACE] FULL RELEASE BY ID | Base: {base_id} | Amt: ${amount_val:.2f}")
            
            # Prioridad 2: Monto explícito (Fallback)
            elif amount is not None:
                try:
                    amount_val = float(amount)
                except (ValueError, TypeError):
                    amount_val = 0.0
                logger.debug(f"⚠️ [CASH-TRACE] RELEASE BY AMT (Fallback) | Amt: ${amount_val:.2f}")
            
            if amount_val > 0:
                old_pending = self.pending_cash
                self.pending_cash = max(0.0, self.pending_cash - amount_val)
                logger.info(f"🔓 [CASH-TRACE] Margin Released: ${amount_val:.2f} | Total Pending: {self.pending_cash:.2f}")
                
        finally:
            if not skip_lock: self.guard.release()

    def release_cash(self, amount):
        """Standard compatibility alias for release_order_margin."""
        self.release_order_margin(amount)

    def cancel_order(self, order_event):
        """Cancela una orden y libera su margen reservado."""
        meta = getattr(order_event, 'metadata', {}) or {}
        reserved_amt = meta['dollar_size']
        order_id = meta['client_order_id']
        if reserved_amt or order_id:
            safe_amt = float(reserved_amt) if reserved_amt is not None else 0.0
            self.release_order_margin(amount=reserved_amt, order_id=order_id)
            logger.info(f"🚫 Order Cancelled/Expired: {order_event.symbol} | Released ${safe_amt:.2f} (ID: {order_id})")

    def get_active_symbols(self):
        symbols = set()
        for v_key in self.virtual_ledger.keys():
            if self.virtual_ledger[v_key]['quantity'] != 0:
                parts = v_key.split('_')
                if parts:
                    symbols.add(parts[0])
        return list(symbols)
    
    def update_timeindex(self, event):
        """
        Update current market prices for all positions.
        """
        if event.type == EventType.MARKET:
            pass 
            
    def _check_auto_secuestro(self):
        """
        FASE 5: Auto-Secuestro (Anti-Black Swan).
        Si duplicamos el capital base por primera vez, mandamos un log especial y marcamos
        para que el Engine / Executor puedan transferir la semilla inicial (Risk-Free Mode).
        """
        # Manejamos rounding drifts comparando con un pequeño delta
        target_capital = self.base_initial_capital * 2.0
        if not self.initial_risk_secured and self.current_cash >= (target_capital - 1e-6):
            logger.critical(f"🌌 [AUTO-SECUESTRO] ¡Capital duplicado! De ${self.base_initial_capital:.2f} a ${self.current_cash:.2f}.")
            logger.critical(f"🌌 [AUTO-SECUESTRO] El sistema está listo para asegurar la semilla inicial en Spot.")
            # En vez de ejecutar la API de Binance aquí, simplemente guardamos el estado y restamos el capital
            # asumiendo que el Engine/Executor recogerán este flag.
            self.initial_risk_secured = True
            
            # Barrido exacto del capital base original para evitar drift
            sweep_amount = round(self.base_initial_capital, 4)
            self.total_secured_capital += sweep_amount
            self.current_cash -= sweep_amount
            
            logger.critical(f"🛡️ [RISK-FREE MODE ACTIVO] Semilla extraída (${sweep_amount:.2f}). Operando con 'Dinero de la Casa'. Cash restante: ${self.current_cash:.2f}")

    def update_snapshot(self, equity: float):
        """
        Helper to update current price of a symbol for PnL calculation.
        Updates HWM for LONG positions, LWM for SHORT positions.
        """
        pass

    def update_market_price(self, symbol, price):
        """
        Helper to update current price of a symbol for PnL calculation.
        Updates HWM for LONG positions, LWM for SHORT positions.
        """
        if price <= 0:
            logger.warning(f"Portfolio: Ignoring invalid price for {symbol}: {price}")
            return

        self.guard.acquire()
        try:
            # 👻 FORENSIC-V130: GHOST TICK PREVENTION FOR MFE/MAE
            # QUÉ: Evitar que un spike anómalo de la API altere HWM/LWM.
            # POR QUÉ: Un ghost tick dispararía el HWM, causando que el Trailing Stop
            #   se active prematuramente en el siguiente tick normal.
            _is_ghost_tick = False
            _last_prices_dict = getattr(self, '_last_prices', {})
            _last_p = _last_prices_dict[symbol] if symbol in _last_prices_dict else 0.0
            if _last_p and _last_p > 0:
                _jump = abs(price - _last_p) / _last_p
                # Si el tick salta más de 2% de golpe, es probablemente un artefacto de la API
                # o un flash event que no debe cristalizarse en el HWM de inmediato.
                if _jump > 0.02:
                    _is_ghost_tick = True
                    logger.warning(f"👻 [GHOST TICK DETECTED] {symbol}: Jump {_jump*100:.2f}% ({_last_p} → {price}). Protecting HWM/LWM.")

            self._last_prices[symbol] = price  # Meritocracy Bridge (Phase 3.13)
                    
            # --- 🛡️ PHASE 15: HEDGE MODE SYNC ---
            # 🛡️ VIRTUAL LEDGER SYNC: Propagate real-time price to all horizon sub-positions
            
            # [FASE 4: NANO OPTIMIZATION] - Zero-Copy C++ fast loop
            if self._nano_ledger is not None:
                active_v_keys = self._nano_ledger.update_market_price(symbol, float(price), bool(_is_ghost_tick))
            else:
                active_v_keys = []
                for v_key, vpos in self.virtual_ledger.items():
                    if v_key.startswith(f"{symbol}_"):
                        vpos['current_price'] = price
                        if 'high_water_mark' not in vpos: vpos['high_water_mark'] = price
                        if 'low_water_mark' not in vpos: vpos['low_water_mark'] = price
                        
                        if not _is_ghost_tick:
                            if update_hwm_lwm is not None:
                                new_hwm, new_lwm = update_hwm_lwm(
                                    float(price), 
                                    float(vpos['high_water_mark']), 
                                    float(vpos['low_water_mark'])
                                )
                                vpos['high_water_mark'] = new_hwm
                                vpos['low_water_mark'] = new_lwm
                            else:
                                if price > vpos['high_water_mark']:
                                    vpos['high_water_mark'] = price
                                if vpos['low_water_mark'] == 0 or price < vpos['low_water_mark']:
                                    vpos['low_water_mark'] = price
                                
                        # Mirror changes to self.positions
                        if v_key in self.positions:
                            self.positions[v_key]['current_price'] = vpos['current_price']
                            self.positions[v_key]['high_water_mark'] = vpos['high_water_mark']
                            self.positions[v_key]['low_water_mark'] = vpos['low_water_mark']
                        active_v_keys.append(v_key)
            
            # DB Snapshot prep (Copy inside lock)
            snapshot_positions = []
            for v_key in active_v_keys:
                if v_key in self.positions:
                    snapshot_positions.append((v_key, self.positions[v_key].copy()))
            should_update_db = len(snapshot_positions) > 0
        finally:
            self.guard.release()

        # PHOENIX FIX: Restored from orphaned dead code after get_kelly_metrics().
        # This code was disconnected from update_market_price during refactoring.
        # Without it: equity cache goes stale, crash recovery DB writes don't happen.
        now = self._get_current_time()
        if not hasattr(self, '_last_save_time'): self._last_save_time = datetime.min.replace(tzinfo=timezone.utc)
        
        if self.auto_save and (now - self._last_save_time).total_seconds() > 1.0:
            self._refresh_equity_cache()
            self.save_status()
            self._last_save_time = now
        else:
            self._refresh_equity_cache()
            
        # Update DB (Snapshot for crash recovery)
        if should_update_db:
            for v_key, pos in snapshot_positions:
                qty = pos['quantity']
                avg = pos['avg_price']
                pnl = (price - avg) * qty if qty != 0 else 0
                self.io_executor.submit(self.db.update_position, v_key, qty, avg, price, pnl)

        # ═══════════════════════════════════════════════════════════════
        # CTOS PHASE 4: TRADE CHRONICLE — TICK-BY-TICK HISTORY
        # QUÉ: Cada 5 ticks, graba el estado de CADA posición abierta.
        # POR QUÉ: La tabla trade_chronicle existía pero NUNCA se llenaba.
        #   Sin historia, no podemos identificar el punto óptimo de cierre.
        # PARA QUÉ: Reconstrucción forense completa de cada trade.
        # CÓMO: Itera virtual_ledger, calcula métricas, llama log_trade_chronicle().
        # CUÁNDO: Cada 5 actualizaciones de precio por posición.
        # DÓNDE: core/portfolio.py → update_market_price()
        # QUIÉN: Portfolio → DatabaseHandler.log_trade_chronicle()
        # ═══════════════════════════════════════════════════════════════
        try:
            for v_key, vpos in list(self.virtual_ledger.items()):
                if vpos['quantity'] == 0:
                    continue
                if not v_key.startswith(f"{symbol}_"):
                    continue
                    
                trade_id = vpos['trade_id']
                if not trade_id:
                    continue
                    
                # Increment tick counter
                self._chronicle_tick_counters[trade_id] = self._chronicle_tick_counters[trade_id] + 1
                tick_num = self._chronicle_tick_counters[trade_id]
                
                # Log every 5 ticks
                if tick_num % 5 != 0:
                    continue
                    
                entry_p = vpos['avg_price']
                if entry_p <= 0:
                    continue
                    
                qty = vpos['quantity']
                horizon = vpos['horizon']
                direction = 'LONG' if qty > 0 else 'SHORT'
                
                # Calculate unrealized PnL %
                if direction == 'LONG':
                    unrealized_pct = ((price - entry_p) / entry_p) * 100
                    mfe = ((vpos['high_water_mark'] - entry_p) / entry_p) * 100
                    mae = ((entry_p - vpos['low_water_mark']) / entry_p) * 100
                else:
                    unrealized_pct = ((entry_p - price) / entry_p) * 100
                    mfe = ((entry_p - vpos['low_water_mark']) / entry_p) * 100
                    mae = ((vpos['high_water_mark'] - entry_p) / entry_p) * 100
                
                # Distance to SL/TP
                sl_pct = vpos['sl_pct'] or 0
                tp_pct = vpos['tp_pct'] or 0
                dist_tp = (tp_pct * 100 - unrealized_pct) if tp_pct > 0 else 0
                dist_sl = (unrealized_pct - (-sl_pct * 100)) if sl_pct > 0 else 0
                
                # Oracle predictions from entry metadata
                _meta = vpos['metadata'] or {}
                _traj = vpos['trajectory_prediction'] or _meta['trajectory_prediction'] or {}
                pred_mag = _traj['magnitude_pct'] if isinstance(_traj, dict) else None
                pred_target = _traj['target_price'] if isinstance(_traj, dict) else None
                pred_bars = _traj['duration_bars'] if isinstance(_traj, dict) else None
                
                entry_size = abs(qty) * entry_p
                
                self.io_executor.submit(
                    self.db.log_trade_chronicle,
                    trade_id=trade_id,
                    symbol=symbol,
                    horizon=horizon,
                    tick_number=tick_num,
                    current_price=price,
                    entry_price=entry_p,
                    unrealized_pnl_pct=unrealized_pct,
                    distance_to_tp_pct=dist_tp,
                    distance_to_sl_pct=dist_sl,
                    mfe_so_far=mfe,
                    mae_so_far=mae,
                    oracle_prediction_magnitude=pred_mag,
                    oracle_prediction_target_price=pred_target,
                    oracle_prediction_time_bars=pred_bars,
                    direction=direction,
                    entry_size_usd=entry_size,
                )
        except Exception as _chron_e:
            logger.debug(f"[CHRONICLE] Tick logging skipped: {_chron_e}")

    def _update_kelly_stats(self, pnl: float):
        """
        Phase 14: Updates rolling statistics for Dynamic Kelly computation.
        Uses a rolling window of recent trades.
        """
        is_win = pnl > 0
        self.kelly_trades_history.append({'pnl': pnl, 'is_win': is_win})
        
        # Enforce rolling window
        if len(self.kelly_trades_history) > self.KELLY_WINDOW:
            self.kelly_trades_history.pop(0)
            
        # Recompute stats
        wins = 0
        gross_profit = 0.0
        gross_loss = 0.0
        loss_count = 0
        
        for t in self.kelly_trades_history:
            if t['is_win']:
                wins += 1
                gross_profit += t['pnl']
            else:
                loss_count += 1
                gross_loss += abs(t['pnl'])
                
        total_trades = len(self.kelly_trades_history)
        if total_trades > 0:
            self.kelly_winrate = wins / total_trades
            
        avg_win = (gross_profit / wins) if wins > 0 else 0.0
        avg_loss = (gross_loss / loss_count) if loss_count > 0 else 1.0 # Prevent div by zero
        
        if avg_loss > 0:
            self.kelly_payoff_ratio = avg_win / avg_loss
        else:
            self.kelly_payoff_ratio = avg_win # If no losses, payoff is basically infinite, but cap at average win
            
    def get_kelly_metrics(self) -> tuple[float, float]:
        """Returns (winrate, payoff_ratio) for Kelly computation"""
        return self.kelly_winrate, self.kelly_payoff_ratio

    @trace_execution
    def update_signal(self, event):
        if event.type == EventType.SIGNAL:
            if event.metadata:
                self.guard.acquire()
                try:
                    if not hasattr(self, '_pending_metadata'):
                        self._pending_metadata = {}
                    
                    # Store full metadata by symbol (or v_key if horizon is available)
                    meta_key = f"{event.symbol}_{getattr(event, 'horizon', 'SCALPING')}"
                    self._pending_metadata[meta_key] = event.metadata
                finally:
                    self.guard.release()
                
                # SOPHIA-INTELLIGENCE: Store trade intent for Post-Mortem
                sophia_data = event.metadata['sophia']
                if sophia_data and hasattr(event, 'trade_id') and event.trade_id:
                    try:
                        self.sophia_post_mortem.store_intent(
                            trade_id=event.trade_id,
                            symbol=event.symbol,
                            direction=event.signal_type.name if hasattr(event.signal_type, 'name') else str(event.signal_type),
                            sophia_report=sophia_data,
                            narrative=event.metadata['sophia_narrative'],
                            trigger_price=getattr(event, 'current_price', 0.0) or 0.0,
                        )
                    except Exception as e:
                        logger.debug(f"[SOPHIA] Intent store skipped: {e}")

    @omniscient_trace(layer="MEMORY")
    def _update_virtual_ledger(self, event) -> float:
        """🛡️ Phase 1 (Virtual Ledger): Isolates Avg Entry Price for Scalping vs Swing safely. Returns isolated PnL."""
        horizon = getattr(event, 'horizon', 'SCALPING')
        
        # 🛡️ PHOENIX V3: HEDGE MODE ENFORCEMENT
        # Aislamiento de LONG y SHORT simultáneos para el mismo símbolo/horizonte
        meta = getattr(event, 'metadata', {}) or {}
        is_close = getattr(event, 'is_close', False) or getattr(event, 'is_exit', False) or meta['is_close'] or meta['is_exit']
        
        # FORENSIC FIX: Force is_close from FillEvent top-level if present (from UserDataStream)
        if getattr(event, 'is_closed', False) and (meta['is_close'] or meta['is_exit']):
            is_close = True
            
        direction_val = event.direction.name if hasattr(event.direction, 'name') else str(event.direction)
        
        # ═══════════════════════════════════════════════════════════════
        # NATIVE BINANCE HEDGE ROUTING (Zombie Trade Prevention)
        # QUÉ: Usa la información nativa de Binance para dirigir el fill.
        # ═══════════════════════════════════════════════════════════════
        binance_ps = meta['binance_position_side']
        
        if binance_ps in ('LONG', 'SHORT'):
            # El Websocket nos dijo EXACTAMENTE a qué lado pertenece
            pos_side = binance_ps
        else:
            # Fallback legacy si viene de REST
            if is_close:
                pos_side = 'LONG' if direction_val == 'SELL' else 'SHORT'
            else:
                pos_side = 'LONG' if direction_val == 'BUY' else 'SHORT'
            
        v_key = f"{event.symbol}_{horizon}_{pos_side}"
        existing_qty = self.virtual_ledger[v_key]['quantity'] if v_key in self.virtual_ledger else 0.0
        
        if v_key not in self.virtual_ledger:
            # Initialize specialized ledger for this horizon and side
            self.virtual_ledger[v_key] = {
                'quantity': 0.0,
                'avg_price': 0.0,
                'horizon': horizon,
                'pos_side': pos_side,
                'current_price': 0.0,
                'high_water_mark': 0.0,
                'low_water_mark': 0.0,
                'entry_time': self._get_current_time().timestamp() if hasattr(self._get_current_time(), 'timestamp') else time.time(),
                'sl_pct': getattr(event, 'sl_pct', None),
                'tp_pct': getattr(event, 'tp_pct', None),
                'opener_strategy_id': getattr(event, 'strategy_id', None) or 'Unknown',
                'cognitive_anchor': None,
                'setup_type': getattr(event, 'setup_type', 'UNKNOWN'),
                'strategy_version': getattr(event, 'strategy_version', '1.0.0'),
                'ml_confidence': (getattr(event, 'metadata', {}) or {}).get('ml_confidence', 0.0),
                'tp_limit_placed': False,
                'tp_order_id': None,
                'trade_id': getattr(event, 'trade_id', None),
                'predicted_duration': getattr(event, 'predicted_duration', None),
                'predicted_magnitude': getattr(event, 'predicted_magnitude', None),
                'metadata': getattr(event, 'metadata', {}) or {},
                'exit_pending_time': 0, # FORENSIC FIX: Initialize exit race condition lock
                'state': 'OPENING',
                # === TRAILING ENGINE V7 FIELDS ===
                'mfe_atr': 0.0,
                'fase_actual': 'FASE_0_RIESGO_INICIAL',
                'ratio_captura': 0.0,
                'trail_stop_price': 0.0,
                'strategy_family': 'DEFAULT',
                'leverage': getattr(event, 'leverage', getattr(Config, 'BINANCE_LEVERAGE', 1)),
                'trajectory_prediction': (getattr(event, 'metadata', {}) or {}).get('trajectory_prediction', 'UNKNOWN')
            }
            if getattr(self, '_nano_ledger', None) is not None:
                self._nano_ledger.register_vkey(event.symbol, v_key)
            
        pos = self.virtual_ledger[v_key]
        
        # Enforce strict ownership: verify that the exit signal's strategy_id or horizon matches the position's opener_strategy_id before closing.
        existing_qty = pos['quantity']
        is_exit_fill = False
        if existing_qty > 0 and event.direction == OrderSide.SELL:
            is_exit_fill = True
        elif existing_qty < 0 and event.direction == OrderSide.BUY:
            is_exit_fill = True
            
        if is_exit_fill:
            opener_strat = pos['opener_strategy_id']
            evt_strat = getattr(event, 'strategy_id', 'Unknown')
            system_exits = {'99', 'EXIT', 'EMERGENCY_EXIT', 'KILL_SWITCH', 'risk_manager', 'RiskManager', 'Unknown', 'HARD_SL', 'TIME_STOP_ZOMBIE', 'HARD_SCALP_TIMEOUT', 'ZOMBIE_FLAT_MARKET', 'ZOMBIE', 'BACKTEST_CLOSE', 'DIAG_CLOSE', 'LIFECYCLE_EXIT', 'TURBO_BE', 'DRAWDOWN_LIMIT', 'ZOMBIE_CHASER_EXIT', 'ML_PREDICTED_TP', 'PLACE_TP_LIMIT', 'AUTO_HEALING_HEDGE', 'MOMENT_MGR', 'FORCE_CLOSE'}
            
            is_sys_exit = (
                evt_strat in system_exits or
                any(evt_strat.startswith(p) for p in ("HARD_", "SPAP_", "TRAIL_", "WEAK_", "V7_", "CLOSE_", "TIME_", "MOMENT_", "LONG_", "SHORT_", "T1_", "T2_", "T3_", "MACRO_"))
            )

            
            if not is_sys_exit and opener_strat != 'Unknown' and evt_strat != opener_strat:
                logger.error(
                    f"🛡️ [OWNERSHIP VIOLATION] Strategy {evt_strat} attempted to close position "
                    f"owned by {opener_strat} for {event.symbol} ({horizon}). Blocking ledger update!"
                )
                return 0.0

        # ═══════════════════════════════════════════════════════════════
        # FORENSIC FIX: None-coalescing price derivation
        # QUÉ: getattr(event, 'fill_price', default) returns None when
        #   the attribute EXISTS but is None — NOT the default value.
        # POR QUÉ: This caused `float * None` → TypeError, silently
        #   leaving quantity=0 in the ledger → exits never fire.
        # CÓMO: Explicit None-check chain: fill_price → fill_cost/qty → price
        # ═══════════════════════════════════════════════════════════════
        price = getattr(event, 'fill_price', None)
        if price is None or price == 0:
            # Derive from fill_cost / quantity (the canonical source)
            qty = getattr(event, 'quantity', 0)
            fc = getattr(event, 'fill_cost', 0)
            if qty > 0 and fc > 0:
                price = fc / qty
            else:
                price = getattr(event, 'price', None) or 0.0
        if not price or price == 0: return 0.0 # Skip invalid physics
        
        # Update SL/TP if event provides new targets (Dynamic Calibration)
        if getattr(event, 'sl_pct', None): pos['sl_pct'] = event.sl_pct
        if getattr(event, 'tp_pct', None): pos['tp_pct'] = event.tp_pct
        
        fill_cost = event.quantity * price
        isolated_pnl = 0.0
        closed = 0.0 # FASE 2: Track closed quantity explicitly
        
        # Calculate new average price isolating strategies
        if event.direction == OrderSide.BUY:
            if pos['quantity'] < 0: # Closing Short
                closed = min(abs(pos['quantity']), event.quantity)
                isolated_pnl = (pos['avg_price'] - price) * closed
                self._record_closed_trade(event, pos, closed, pos['avg_price'], price, isolated_pnl)
                pos['quantity'] += closed
                if abs(pos['quantity']) < 1e-8:
                    pos['quantity'] = 0.0
                    pos['exit_pending_time'] = 0  # FORENSIC FIX #19: Clear lock on successful close
                if event.quantity > closed:
                    remain = event.quantity - closed
                    pos['quantity'] = remain
                    pos['avg_price'] = price
                    pos['entry_time'] = self._get_current_time()
                    pos['opener_strategy_id'] = getattr(event, 'strategy_id', 'Unknown')
                    pos['ml_confidence'] = (getattr(event, 'metadata', {}) or {})['ml_confidence']
                    pos['trajectory_prediction'] = (getattr(event, 'metadata', {}) or {})['trajectory_prediction']
                    pos['tp_limit_placed'] = False
                    pos['tp_order_id'] = None
                    pos['metadata'] = getattr(event, 'metadata', {}) or {}
                    self._bind_cognitive_anchor(event.symbol, pos)
                else:
                    # If just added to existing, keep original opener_id but could log the add
                    pos['scale_count'] = pos['scale_count'] + 1
                    logger.info(f"📈 [PYRAMID RECORDED] {event.symbol} added to LONG. Scale count: {pos['scale_count']}")
            else: # Adding Short
                total_cost = (abs(pos['quantity']) * pos['avg_price']) + fill_cost
                pos['quantity'] += event.quantity
                pos['avg_price'] = total_cost / pos['quantity']
                if pos['quantity'] == event.quantity: # New entry
                    pos['entry_time'] = self._get_current_time()
                    pos['opener_strategy_id'] = getattr(event, 'strategy_id', None) or 'Unknown'
                    pos['ml_confidence'] = (getattr(event, 'metadata', {}) or {}).get('ml_confidence', 0.0)
                    pos['trajectory_prediction'] = (getattr(event, 'metadata', {}) or {}).get('trajectory_prediction', 'UNKNOWN')
                    pos['tp_limit_placed'] = False
                    pos['tp_order_id'] = None
                    # FORENSIC FIX: Reset watermarks on new trade to prevent instant TURBO_BE
                    pos['high_water_mark'] = price
                    pos['low_water_mark'] = price
                    pos['metadata'] = getattr(event, 'metadata', {}) or {}
                    pos['exit_pending_time'] = 0 # FORENSIC FIX: Reset lock on new DCA/add
                    self._bind_cognitive_anchor(event.symbol, pos)
                    
                    # 🚀 TELEGRAM NOTIFICATION: Handled by log_trade_report() → send_trade_open()
                    # FORENSIC-V21 FIX #3: Removed duplicate raw notification.
                    # The enhanced notification in log_trade_report() includes ALL context.
                    pass
        else: # OrderSide.SELL
            if pos['quantity'] > 0: # Closing Long
                closed = min(pos['quantity'], event.quantity)
                isolated_pnl = (price - pos['avg_price']) * closed
                self._record_closed_trade(event, pos, closed, pos['avg_price'], price, isolated_pnl)
                pos['quantity'] -= closed
                if abs(pos['quantity']) < 1e-8:
                    pos['quantity'] = 0.0
                    pos['exit_pending_time'] = 0  # FORENSIC FIX #19: Clear lock on successful close
                if event.quantity > closed:
                    remain = event.quantity - closed
                    pos['quantity'] = -remain
                    pos['avg_price'] = price
                    pos['entry_time'] = self._get_current_time()
                    pos['opener_strategy_id'] = getattr(event, 'strategy_id', None) or 'Unknown'
                    pos['ml_confidence'] = (getattr(event, 'metadata', {}) or {})['ml_confidence']
                    pos['trajectory_prediction'] = (getattr(event, 'metadata', {}) or {})['trajectory_prediction']
                    pos['tp_limit_placed'] = False
                    pos['tp_order_id'] = None
                    pos['metadata'] = getattr(event, 'metadata', {}) or {}
                    self._bind_cognitive_anchor(event.symbol, pos)
            else: # Adding Short
                total_cost = (abs(pos['quantity']) * pos['avg_price']) + fill_cost
                pos['quantity'] -= event.quantity
                pos['avg_price'] = total_cost / abs(pos['quantity'])
                if pos['quantity'] == -event.quantity: # New entry
                    pos['entry_time'] = self._get_current_time()
                    pos['opener_strategy_id'] = getattr(event, 'strategy_id', None) or 'Unknown'
                    pos['ml_confidence'] = (getattr(event, 'metadata', {}) or {})['ml_confidence']
                    pos['trajectory_prediction'] = (getattr(event, 'metadata', {}) or {})['trajectory_prediction']
                    pos['tp_limit_placed'] = False
                    pos['tp_order_id'] = None
                    # FORENSIC FIX: Reset watermarks on new trade to prevent instant TURBO_BE
                    pos['high_water_mark'] = price
                    pos['low_water_mark'] = price
                    pos['metadata'] = getattr(event, 'metadata', {}) or {}
                    pos['exit_pending_time'] = 0 # FORENSIC FIX: Reset lock on new DCA/add
                    self._bind_cognitive_anchor(event.symbol, pos)
                    
                    # 🚀 TELEGRAM NOTIFICATION: Handled by log_trade_report() → send_trade_open()
                    # FORENSIC-V21 FIX #3: Removed duplicate raw notification.
                    pass
                else:
                    pos['scale_count'] = pos['scale_count'] + 1
                    logger.info(f"📈 [PYRAMID RECORDED] {event.symbol} added to SHORT. Scale count: {pos['scale_count']}")


        # Re-evaluar cognitive anchor si the direction se volteó
        if getattr(pos, 'just_flipped', False):
             self._bind_cognitive_anchor(event.symbol, pos)
                
        pos['current_price'] = price
        pos['high_water_mark'] = max(pos['high_water_mark'], price)
        if pos['low_water_mark'] == 0 or price < pos['low_water_mark']:
            pos['low_water_mark'] = price
            
        # Determine the final state of the position
        if abs(pos['quantity']) < 1e-8:
            pos['state'] = 'CLOSED'
        elif pos['exit_pending_time'] > 0:
            pos['state'] = 'CLOSING'
        elif pos['high_water_mark'] > pos['avg_price'] * 1.002:
            pos['state'] = 'TRAILING'
        else:
            pos['state'] = 'ACTIVE'
            
        new_qty = pos['quantity']
        
        # Did the position close or flip?
        if (existing_qty != 0.0 and new_qty == 0.0) or (existing_qty * new_qty < 0):
            from core.senior_auditor import SeniorAuditor
            self.io_executor.submit(
                SeniorAuditor().log_trade_lifecycle,
                trade_id=pos['trade_id'] or getattr(event, 'trade_id', None),
                action="EXIT",
                details={
                    "symbol": event.symbol,
                    "horizon": horizon,
                    "pos_side": pos_side,
                    "exit_price": price,
                    "gross_pnl": isolated_pnl,
                    "exit_reason": getattr(event, 'exit_reason', 'FLIP_EXIT' if (existing_qty * new_qty < 0) else 'NORMAL_CLOSE')
                }
            )

        # Did a new position open or flip?
        if (existing_qty == 0.0 and new_qty != 0.0) or (existing_qty * new_qty < 0):
            from core.senior_auditor import SeniorAuditor, STRATEGY_DNA
            from core.asset_intelligence import get_asset_intelligence
            strat_key = SeniorAuditor()._map_strategy_name(pos['opener_strategy_id'])
            dna = STRATEGY_DNA[strat_key]
            profile = get_asset_intelligence().get_profile(event.symbol)
            
            dna_snapshot = dict(dna)
            dna_snapshot["ASIMETRIAS_DEL_ACTIVO_SNAP"] = {
                "min_signal_threshold": profile.min_signal_threshold,
                "factor_sizing": profile.factor_sizing,
                "stop_atr_mult": profile.stop_atr_mult,
                "kelly_fraction": profile.kelly_fraction,
                "tier": str(profile.tier),
                "volatility": str(profile.volatility)
            }
            pos['strategy_dna'] = dna_snapshot
            
            self.io_executor.submit(
                SeniorAuditor().log_trade_lifecycle,
                trade_id=pos['trade_id'] or getattr(event, 'trade_id', None),
                action="ENTRY",
                details={
                    "symbol": event.symbol,
                    "horizon": horizon,
                    "pos_side": pos_side,
                    "entry_price": price,
                    "quantity": abs(new_qty),
                    "sl_pct": pos['sl_pct'],
                    "tp_pct": pos['tp_pct'],
                    "opener_strategy_id": pos['opener_strategy_id'],
                    "strategy_dna": dna_snapshot
                }
            )
            
        logger.info(f"📓 [LEDGER] {v_key} | State: {pos['state']} | Qty: {pos['quantity']:.4f} | Avg: ${pos['avg_price']:.2f} | Unrealized PnL: ${isolated_pnl:.4f} | Target SL: {pos['sl_pct']}")
        
        # FORENSIC FIX FASE 2: Evitar Phantom Losses en RL (Solo retorna PnL si cerramos algo)
        return isolated_pnl if closed > 0 else None

    def _record_closed_trade(self, event, pos, closed_qty, entry_price, exit_price, gross_pnl):
        """Generates the Forensic Dictionary for closed operations and routes it."""
        import uuid
        
        # ═══════════════════════════════════════════════════════════════
        # FORENSIC FIX #5: DYNAMIC FEE PER LEG (Entry=Maker, Exit=Dynamic)
        # QUÉ: El fee de salida depende del tipo de orden real (LIMIT vs MARKET).
        # POR QUÉ: Antes hardcodeaba Maker para AMBAS patas. Pero Kill Switch
        #   exits y BBO fallbacks usan MARKET (Taker 0.0375% vs Maker 0.02%).
        # PARA QUÉ: Net PnL preciso en trades cerrados por emergencia.
        # CÓMO: Lee 'actual_order_type' del metadata del FillEvent.
        # ═══════════════════════════════════════════════════════════════
        real_commission = getattr(event, 'commission', None)
        _meta = getattr(event, 'metadata', {}) or {}
        exit_order_type = _meta['actual_order_type']
        
        # Entry fee: always Maker (BBO architecture → entries are LIMIT)
        entry_fee_rate = getattr(Config, 'BINANCE_MAKER_FEE_BNB', 0.0002)
        # Exit fee: depends on actual execution type
        if exit_order_type == 'market':
            exit_fee_rate = getattr(Config, 'BINANCE_TAKER_FEE_BNB', 0.000375)
        else:
            exit_fee_rate = getattr(Config, 'BINANCE_MAKER_FEE_BNB', 0.0002)
        
        fees_entry = (closed_qty * entry_price) * entry_fee_rate
        fees_exit = (closed_qty * exit_price) * exit_fee_rate
        
        if real_commission is not None and real_commission > 0:
            # FORENSIC-V81: real_commission from the EXIT event is ONLY the exit fee.
            # We MUST add it to the estimated entry fee (which was already paid on entry).
            fees_exit = real_commission
            total_fees = fees_entry + fees_exit
        else:
            total_fees = fees_entry + fees_exit
        
        net_pnl = gross_pnl - total_fees
        net_pnl_percent = net_pnl / (closed_qty * entry_price) if (closed_qty * entry_price) > 0 else 0
        
        # [PHASE 6] Kelly Hot-Hand Tracker Injection
        try:
            from core.compounding_engine import get_compounding_engine
            get_compounding_engine().record_trade_result(net_pnl)
        except Exception as e:
            pass
        
        now_ts = self._get_current_time()
        duration = int((now_ts - pos['entry_time']).total_seconds()) if pos['entry_time'] else 0
        
        exit_side = getattr(event, 'direction', OrderSide.SELL)
        if exit_side == OrderSide.SELL:
            closed_direction = "LONG"   # We SELL to close LONG
        else:
            closed_direction = "SHORT"  # We BUY to close SHORT
            
        size_usd = closed_qty * entry_price
        leverage = getattr(Config, 'BINANCE_LEVERAGE', 10.0) if getattr(Config, 'BINANCE_USE_FUTURES', False) else 1.0
        margin_usd = float(size_usd) / leverage
        size_percent = margin_usd / float(self.current_cash) if self.current_cash > 0 else 0.0

        opener_strat = pos['opener_strategy_id']
        evt_strat = getattr(event, 'strategy_id', "")
        
        # Determine exit_reason. Priority: event.exit_reason > event.strategy_id > "NORMAL_CLOSE"
        exit_reason = getattr(event, 'exit_reason', None)
        if not exit_reason:
            exit_reason = evt_strat if evt_strat and evt_strat != opener_strat else "NORMAL_CLOSE"
        
        # Compute MFE and MAE
        hwm = pos['high_water_mark']
        lwm = pos['low_water_mark']
        if closed_direction == "LONG":
            mfe_pct = (hwm - entry_price) / entry_price if entry_price > 0 else 0.0
            mae_pct = (entry_price - lwm) / entry_price if entry_price > 0 else 0.0
        else: # SHORT
            mfe_pct = (entry_price - lwm) / entry_price if entry_price > 0 else 0.0
            mae_pct = (hwm - entry_price) / entry_price if entry_price > 0 else 0.0

        trade_data = {
            "trade_id": getattr(event, 'trade_id', None) or pos['trade_id'] or str(uuid.uuid4()),
            "exit_reason": exit_reason,
            "symbol": event.symbol,
            "strategy_id": opener_strat,
            "horizon": getattr(event, 'horizon', 'SCALPING'),
            "direction": closed_direction,
            "quantity": closed_qty,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size_usd": size_usd,
            "margin_usd": margin_usd,
            "size_percent": size_percent,
            "fees_entry": fees_entry,
            "fees_exit": fees_exit,
            "slippage_entry": 0.0,
            "slippage_exit": 0.0,
            "fees_paid": total_fees,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "net_pnl_percent": net_pnl_percent,
            "duration_seconds": duration,
            "exit_reason": exit_reason,
            "closed_at": now_ts.isoformat(),
            "exit_type": exit_order_type,
            "oracle_certainty": pos["ml_confidence"],
            "setup_type": pos['setup_type'],
            "strategy_version": pos['strategy_version'],
            "mfe_pct": mfe_pct,
            "mae_pct": mae_pct,
            # CTOS Phase 3: Exit attribution
            "closer_strategy_id": evt_strat if evt_strat else "UNKNOWN",
            "opener_strategy_id": opener_strat,
            # CTOS Phase 4: Size tracking (Open → Close)
            "open_size_usd": closed_qty * entry_price,
            "close_size_usd": closed_qty * exit_price,
            "size_delta_usd": (closed_qty * exit_price) - (closed_qty * entry_price),
            # ═══════════════════════════════════════════════════════════════
            # FORENSIC FIX #2: EXIT BALLOT PROPAGATION
            # QUÉ: Captura los votos de las estrategias de cierre al momento
            #   del cierre del trade y los almacena en trade_data.
            # POR QUÉ: RiskManager.check_stops() almacena votos en vpos['_exit_votes']
            #   y vpos['_hold_votes'], pero estos se perdían al cerrar.
            # PARA QUÉ: El Telegram muestra qué estrategias votaron EXIT vs HOLD.
            # ═══════════════════════════════════════════════════════════════
            "exit_ballot": {
                'exit_voters': pos['_exit_votes'],
                'hold_voters': pos['_hold_votes'],
            } if pos['_exit_votes'] or pos['_hold_votes'] else None,
        }
        
        # ═══════════════════════════════════════════════════════════════
        # FORENSIC-V31: Store reference for log_trade_report() enrichment
        # QUÉ: Guarda los datos del trade cerrado para que log_trade_report()
        #   pueda usar net_pnl real, fees breakdown, y duración exacta.
        # POR QUÉ: Sin esto, log_trade_report() tiene que recalcular todo.
        # ═══════════════════════════════════════════════════════════════
        
        # --- FORENSIC-V36: LOSS PACING (Dynamic Cooldown) ---
        if not hasattr(self, 'consecutive_losses'):
            self.consecutive_losses = {}
            
        pacing_key = f"{event.symbol}_{trade_data['horizon']}"
        if net_pnl < 0:
            self.consecutive_losses[pacing_key] = self.consecutive_losses[pacing_key] + 1
            if self.consecutive_losses[pacing_key] >= 2:
                logger.warning(f"❄️ [PACING COOLDOWN] {pacing_key} suffered 2 consecutive losses. Freezing entries for 15 mins.")
                from utils.cooldown_manager import cooldown_manager
                if not hasattr(cooldown_manager, 'custom_cooldowns'):
                    cooldown_manager.custom_cooldowns = {}
                frozen_key = f"SHOCK_FREEZE_{event.symbol}"
                cooldown_manager.custom_cooldowns[frozen_key] = datetime.now(timezone.utc)
                self.consecutive_losses[pacing_key] = 0  # Reset after triggering
        else:
            self.consecutive_losses[pacing_key] = 0  # Reset on win
            
        self._last_closed_trade_data = trade_data
        
        # Route to respective ledger
        # ═══════════════════════════════════════════════════════════════
        # PHASE 4 FIX: RAM MEMORY LEAK PREVENTION (LEDGERS)
        # QUÉ: Mantener eficiencia de memoria limitando historiales en RAM.
        # POR QUÉ: Evitar consumo infinito durante backtests o runs de varios días.
        # ═══════════════════════════════════════════════════════════════
        if trade_data['horizon'] in ('SCALPING', 'MICROSCALPING'):
            self.scalping_ledger.append(trade_data)
            if len(self.scalping_ledger) > 50:
                self.scalping_ledger.pop(0)
        else:
            self.swing_ledger.append(trade_data)
            if len(self.swing_ledger) > 50:
                self.swing_ledger.pop(0)
            
        logger.info(f"📓 [TRADE CLOSED] {trade_data['horizon']} | {event.symbol} Neto: ${net_pnl:.4f} | Gross: ${gross_pnl:.4f} | Fees: ${total_fees:.4f} | T: {duration}s | Reason: {exit_reason}")
        
        # Meritocracy Central (Phase 3.9): Record for setup-based sizing
        self.trade_history.append(trade_data)
        if len(self.trade_history) > 50: # Maintain memory efficiency (Phase 6 RAM OFF-LOAD)
            self.trade_history.pop(0)
        
        # ═══════════════════════════════════════════════════════════════
        # FORENSIC-V16: STRATEGY TRACKER INTEGRATION
        # QUÉ: Registra cada trade completado en el tracker para métricas
        #   granulares por estrategia/símbolo/horizonte.
        # POR QUÉ: El StrategySelector anterior tenía IDs hardcoded que no
        #   coincidían con los reales → pesos siempre neutrales.
        # PARA QUÉ: Ranking evolutivo real + sizing basado en performance.
        # ═══════════════════════════════════════════════════════════════
        try:
            from utils.strategy_tracker import strategy_tracker
            entry_ts = pos['entry_time']
            if hasattr(entry_ts, 'timestamp'):
                entry_unix = entry_ts.timestamp()
            else:
                entry_unix = time.time() - duration
            
            strategy_tracker.record_trade(
                strategy_id=opener_strat,
                symbol=event.symbol,
                horizon=trade_data['horizon'],
                direction=closed_direction,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=closed_qty,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                fees=total_fees,
                entry_time=entry_unix,
                exit_time=self._get_current_time().timestamp(),
                exit_reason=exit_reason,
                setup_type=trade_data['setup_type'],
                strategy_version=trade_data['strategy_version']
            )
        except Exception as e:
            logger.debug(f"[StrategyTracker] Record skipped: {e}")
        
        # --- SISTEMA DE AUTO-DIAGNÓSTICO ---
        try:
            from utils.loss_analyzer import get_loss_analyzer
            from utils.auto_correction_engine import get_auto_correction_engine
            
            diag_data = {
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "fees": total_fees,
                "slippage_pct": trade_data["slippage_entry"] + trade_data["slippage_exit"],
                "duration_sec": duration
            }
            issues = get_loss_analyzer().analyze_trade(diag_data)
            if issues:
                get_auto_correction_engine().apply_corrections(issues)
        except Exception as e:
            logger.error(f"⚠️ Error ejecutando Auto-Diagnóstico Post-Trade: {e}")
        
        # ═══════════════════════════════════════════════════════════════
        # CTOS PHASE 3: PREDICTION AUDIT — What was predicted vs reality
        # QUÉ: Al cerrar un trade, registra en prediction_audit la predicción
        #   original de la estrategia que abrió vs el resultado real.
        # POR QUÉ: Para saber qué estrategias predicen bien y cuáles no.
        # PARA QUÉ: Feedback loop → mejorar o desactivar estrategias malas.
        # CÓMO: Lee predicciones del metadata de la posición (trajectory_prediction,
        #   ml_confidence) y compara con MFE/MAE reales.
        # ═══════════════════════════════════════════════════════════════
        try:
            _pos_meta = pos['metadata'] or {}
            _trajectory = pos['trajectory_prediction']
            _predicted_mag = None
            _predicted_dur = None
            _predicted_target = None
            _confidence = pos['ml_confidence'] or 0.0
            
            if _trajectory and isinstance(_trajectory, dict):
                _predicted_mag = _trajectory['magnitude_pct']
                _predicted_dur = _trajectory['duration_bars']
                _predicted_target = _trajectory['target_price']
            elif _pos_meta['predicted_magnitude']:
                _predicted_mag = _pos_meta['predicted_magnitude']
                _predicted_dur = _pos_meta['predicted_duration']
                _predicted_target = _pos_meta['predicted_target_price']
            
            # Calculate optimal exit (MFE point)
            optimal_exit_price = hwm if closed_direction == "LONG" else lwm
            missed_profit = max(0.0, mfe_pct - max(0.0, net_pnl_percent))
            was_correct = net_pnl > 0
            
            # Store in trade_data for notification enrichment
            trade_data['predicted_magnitude'] = _predicted_mag
            trade_data['predicted_duration_bars'] = _predicted_dur
            trade_data['predicted_target_price'] = _predicted_target
            trade_data['prediction_confidence'] = _confidence
            trade_data['optimal_exit_price'] = optimal_exit_price
            trade_data['missed_profit_pct'] = missed_profit
            trade_data['was_prediction_correct'] = was_correct
            
            # Write to DB
            # FORENSIC FIX: Include position sizing data for capital-at-risk traceability
            _open_size_usd = closed_qty * entry_price
            _close_size_usd = closed_qty * exit_price
            self.io_executor.submit(
                self.db.log_prediction_audit,
                trade_id=trade_data['trade_id'],
                thought_id=_pos_meta['thought_id'],
                strategy_id=opener_strat,
                symbol=event.symbol,
                horizon=trade_data['horizon'],
                direction=closed_direction,
                predicted_magnitude_pct=_predicted_mag,
                predicted_duration_bars=_predicted_dur,
                predicted_target_price=_predicted_target,
                confidence=_confidence,
                actual_magnitude_pct=net_pnl_percent * 100,
                actual_duration_bars=int(duration / 60) if duration else 0,  # Convert seconds to bars (~1min)
                actual_exit_price=exit_price,
                was_correct=was_correct,
                optimal_exit_price=optimal_exit_price,
                optimal_exit_bar=0,  # Will be enriched by PredictionTracker
                missed_profit_pct=missed_profit * 100,
                entry_time=pos['entry_time'],
                open_size_usd=_open_size_usd,
                close_size_usd=_close_size_usd,
                size_delta_usd=_close_size_usd - _open_size_usd,
                open_price_at_prediction=entry_price,
            )
            
            # FORENSIC FIX: Update Strategy Report Card for governance
            self.io_executor.submit(
                self.db.update_strategy_report_card,
                strategy_id=opener_strat,
                pnl=net_pnl,
                is_win=was_correct
            )
            
            # Update PredictionTracker with outcome
            if hasattr(self, '_engine') and self._engine:
                # FORENSIC FIX: prediction_tracker lives on risk_manager, not engine
                rm = getattr(self._engine, 'risk_manager', None)
                pt = getattr(rm, 'prediction_tracker', None) if rm else None
                if pt:
                    audit = pt.record_trade_outcome(
                        symbol=event.symbol,
                        is_win=net_pnl > 0,
                        pnl_pct=net_pnl_percent,
                        strategy_id=opener_strat,
                        trade_id=trade_data['trade_id']
                    )
                    if audit and audit['optimal_exit_bar']:
                        trade_data['optimal_exit_bar'] = audit['optimal_exit_bar']
                        trade_data['prediction_audit'] = audit
        except Exception as e:
            logger.debug(f"[CTOS-P3] Prediction audit skipped: {e}")
        
        # Update session PnL tracking
        self._session_net_pnl += net_pnl
        
        # [FASE 4: SWEEP DE COLATERAL] Si es ganancia de Microscalping, la reservamos para Swing
        if trade_data['horizon'] == 'MICROSCALPING' and net_pnl > 0:
            self.swept_micro_profits += net_pnl
            logger.info(f"🧹 [COLLATERAL SWEEP] ${net_pnl:.4f} barrido de MICRO hacia SWING. Total Barrido: ${self.swept_micro_profits:.4f}")
            
        # 🚀 TELEGRAM NOTIFICATION: Handled by log_trade_report() → send_trade_close()
        # FORENSIC-V21 FIX #3: Removed duplicate raw notification.
        # The enhanced notification in log_trade_report() includes ALL context.
        
        # FORENSIC-V21 FIX #5: Store computed trade data for log_trade_report enrichment
        # This dict is used by log_trade_report() to populate the enhanced notification
        # with real net PnL, fees breakdown, and duration from _record_closed_trade.
        self._last_closed_trade_data = trade_data

        # ═══════════════════════════════════════════════════════════════
        # PHASE 2 POWER: HOT ADAPTER RL FEEDBACK LOOP
        # QUÉ: Notifica a TODAS las estrategias ML que tienen HotAdapterRL
        #   sobre el resultado del trade para que actualicen su bias.
        # POR QUÉ: Sin esto, el Hot Adapter nunca aprende → bias=1.0 siempre.
        # PARA QUÉ: Penalizar direcciones perdedoras en tiempo real.
        # CÓMO: Itera por las estrategias registradas en el engine y llama
        #   a hot_adapter.update_weights() con los datos del trade.
        # CUÁNDO: Inmediatamente después de registrar el trade cerrado.
        # DÓNDE: core/portfolio.py → _record_closed_trade()
        # QUIÉN: Portfolio → Engine.strategies → MLStrategy.hot_adapter
        # ═══════════════════════════════════════════════════════════════
        try:
            if hasattr(self, '_engine') and self._engine:
                for strat in getattr(self._engine, 'strategies', []):
                    adapter = getattr(strat, 'hot_adapter', None)
                    if adapter and hasattr(adapter, 'update_weights'):
                        # Only send feedback for matching symbol
                        if getattr(strat, 'symbol', None) == event.symbol:
                            adapter.update_weights(
                                symbol=event.symbol,
                                is_win=net_pnl > 0,
                                pnl_pct=net_pnl_percent,
                                direction=closed_direction
                            )
        except Exception as e:
            logger.debug(f"[HOT-RL] Feedback dispatch skipped: {e}")

    def _bind_cognitive_anchor(self, symbol: str, entry_pos: dict):
        """Asocia metadata pre-computada al momento de abrirse un ledger virtual."""
        meta = None
        if hasattr(self, '_pending_metadata'):
            meta_key = f"{symbol}_{entry_pos['horizon']}"
            meta = self._pending_metadata[meta_key]
            
        if meta:
            entry_pos['cognitive_anchor'] = {
                'initial_strength': meta['signal_strength'],
                'initial_prob': meta['sophia']['win_probability'],
                'ttl_seconds': meta['ttl']
            }
        else:
            entry_pos['cognitive_anchor'] = {
                'initial_strength': 0.8,
                'initial_prob': 0.5,
                'ttl_seconds': 180.0 if entry_pos['horizon'] in ('SCALPING', 'MICROSCALPING') else 3600.0
            }

    @omniscient_trace(layer="MEMORY")
    def update_fill(self, event) -> Optional[Tuple[float, TradeOutcome]]:
        """Atomically update portfolio state. Returns (realized PnL, TradeOutcome) if closed."""
        if event.type == EventType.FILL:
            
            # Subsystem Hook: Update the independent Virtual Ledger for this specific Horizon
            isolated_pnl = 0.0
            try:
                ledger_pnl = self._update_virtual_ledger(event)
                if ledger_pnl is not None:
                    isolated_pnl = ledger_pnl
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Failed to update virtual ledger for {event.symbol}: {e}")
                
            pnl_realized = None
            outcome_obj = None # Neural Fortress Object
            # Update Cash and Positions
            fill_cost = event.fill_cost # Total notional value (price * quantity)
            fill_price = event.fill_cost / event.quantity if event.quantity > 0 else 0
            
            # Calculate Margin Impact (Futures Only)
            margin_impact = 0.0
            if Config.BINANCE_USE_FUTURES:
                leverage = getattr(event, 'leverage', Config.BINANCE_LEVERAGE) or Config.BINANCE_LEVERAGE
                margin_impact = fill_cost / leverage
            
            
            # ═══════════════════════════════════════════════════════════════
            # FORENSIC FIX #1: DYNAMIC FEE RATE (was hardcoded 0.0006 = 0.06%)
            # QUÉ: Selecciona Maker (0.02%) o Taker (0.0375%) según tipo de orden real.
            # POR QUÉ: Con BBO architecture, ~90% de órdenes son LIMIT (Maker).
            #   El rate anterior (0.06%) sobredescontaba fees 3x → capital fantasma perdido.
            # PARA QUÉ: Precisión contable exacta en micro-cuenta.
            # CÓMO: Lee 'actual_order_type' del metadata inyectado por BinanceExecutor.
            # ═══════════════════════════════════════════════════════════════
            _meta = getattr(event, 'metadata', {}) or {}
            actual_order_type = _meta['actual_order_type']  # Default Maker (BBO)
            
            if Config.BINANCE_USE_FUTURES:
                if actual_order_type == 'market':
                    fee_rate = getattr(Config, 'BINANCE_TAKER_FEE_BNB', 0.000375)  # 0.0375%
                else:
                    fee_rate = getattr(Config, 'BINANCE_MAKER_FEE_BNB', 0.0002)    # 0.02%
            else:
                fee_rate = 0.001  # Spot default
            
            if event.commission is not None:
                estimated_fee = event.commission
            else:
                # Fallback to estimate with CORRECT rate
                estimated_fee = fill_cost * fee_rate
            
            
            # BEGIN ATOMIC UPDATE
            self.guard.acquire()
            try:
                # Capture Pre-State for Accounting Audit
                pre_balance = Decimal(str(self.current_cash))
                
                # 🚀 AEGIS-V15: ORDER-AWARE PENDING RELEASE
                # QUÉ: Liberamos el capital reservado usando el client_order_id.
                # POR QUÉ: Evita fugas si el dollar_size no coincide o si hay rellenos parciales.
                _meta = getattr(event, 'metadata', {}) or {}
                order_id = _meta['client_order_id']
                reserved_amt = _meta['dollar_size']
                
                # Liberación atómica: Primero por ID, luego por monto si no hay ID.
                self.release_order_margin(amount=reserved_amt, order_id=order_id, skip_lock=True)
                
                # Deduct fee from Cash immediately (Atomic & Single Deduction)
                self.current_cash -= estimated_fee
                self.total_fees_paid += estimated_fee  # CRITERIO-AXIOMA: explicit fee tracking
                
                logger.info(f"  💸 Fee Paid: ${estimated_fee:.4f} ({fee_rate*100}%)")
                
                # 🚀 FORENSIC-V30: HEDGE MODE NATIVE REFACTOR
                # En lugar de usar self.positions (One-Way) que colisiona LONG/SHORT,
                # confiamos en _update_virtual_ledger (aislado por horizonte) que ya se ejecutó.
                
                # 1. Update Cash & Realized PnL
                if isolated_pnl is not None and isolated_pnl != 0.0:
                    self.realized_pnl += isolated_pnl
                    if Config.BINANCE_USE_FUTURES:
                        self.current_cash += isolated_pnl
                    else:
                        # For Spot: Cash was deducted on entry, now we add the full sold value
                        self.current_cash += (event.quantity * fill_price) + isolated_pnl
                elif not Config.BINANCE_USE_FUTURES:
                    # In Spot, opening a position costs cash
                    self.current_cash -= fill_cost

                # 2. Recalculate Margin perfectly from Virtual Ledger
                if Config.BINANCE_USE_FUTURES:
                    new_used_margin = 0.0
                    for v_pos in self.virtual_ledger.values():
                        if v_pos['quantity'] != 0:
                            lev = v_pos['leverage'] or Config.BINANCE_LEVERAGE
                            new_used_margin += (abs(v_pos['quantity']) * v_pos['avg_price']) / lev
                    self.used_margin = new_used_margin
                
                # 3. Synchronize 'self.positions' perfectly with Virtual Ledger (Isolated by Horizon)
                # Remove obsolete symbol-only or stale keys
                keys_to_remove = [k for k in self.positions.keys() if k.startswith(f"{event.symbol}_") or k == event.symbol]
                for k in keys_to_remove:
                    self.positions.pop(k, None)
                
                active_virtual_items = [(k, v) for k, v in self.virtual_ledger.items() if k.startswith(f"{event.symbol}_") and abs(v['quantity']) > 1e-8]
                for v_key, v_pos in active_virtual_items:
                    self.positions[v_key] = v_pos.copy()
                    self.positions[v_key]['current_price'] = fill_price
                    
                    if 'high_water_mark' not in self.positions[v_key] or self.positions[v_key]['high_water_mark'] == 0:
                        self.positions[v_key]['high_water_mark'] = fill_price
                    if 'low_water_mark' not in self.positions[v_key] or self.positions[v_key]['low_water_mark'] == 0:
                        self.positions[v_key]['low_water_mark'] = fill_price
                        
                    _seg = (getattr(event, 'metadata', {}) or {}).get('segment_policy', None)
                    self.positions[v_key]['exec_policy'] = getattr(_seg, 'execution_type', 'MAKER_ONLY') if _seg else 'MAKER_ONLY'
                    self.positions[v_key]['trail_policy'] = getattr(_seg, 'trailing_aggression', 'STRUCTURED') if _seg else 'STRUCTURED'

                    # 🪐 [OMEGA PHASE 9] SYNC TO RUST PORTFOLIO
                    try:
                        from core.rust_execution_bridge import ffi_set_position_bridge
                        _horizon = v_pos.get('horizon', 'SCALPING')
                        rust_horiz = 0 if _horizon == "SCALPING" else 1
                        rust_side = 1 if v_pos['quantity'] > 0 else -1
                        ffi_set_position_bridge(rust_horiz, rust_side, v_pos['avg_price'], abs(v_pos['quantity']))
                    except Exception as e:
                        pass
                
                # Check for cleared positions to sync to Rust
                try:
                    from core.rust_execution_bridge import ffi_clear_position_bridge
                    _ev_horizon = getattr(event, 'horizon', 'SCALPING')
                    rust_horiz = 0 if _ev_horizon == "SCALPING" else 1
                    if len(active_virtual_items) == 0:
                        ffi_clear_position_bridge(rust_horiz)
                except Exception as e:
                    pass

                # 4. Strategy Performance, Sophia Post-Mortem & Reporting
                if isolated_pnl != 0.0:
                    strat_id = getattr(event, 'strategy_id', None) or 'UNTAGGED_STRAT'
                    # ═══════════════════════════════════════════════════════════════
                    # FORENSIC-V31 FIX: USE NET PNL FOR WIN/LOSS DETERMINATION
                    # QUÉ: Pasamos el net_pnl (después de fees) para clasificar
                    #   correctamente wins vs losses.
                    # POR QUÉ: Con micro-cuenta $13, muchos trades tienen
                    #   gross_pnl > 0 pero net_pnl < 0 (fees > ganancia bruta).
                    #   Esto inflaba el WR a 90-100% cuando realmente era 0%.
                    # PARA QUÉ: WR correcto en Telegram, Kelly sizing preciso.
                    # ═══════════════════════════════════════════════════════════════
                    closed_trade = getattr(self, '_last_closed_trade_data', None)
                    if closed_trade and closed_trade['symbol'] == event.symbol:
                        net_pnl_for_perf = closed_trade['net_pnl']
                    else:
                        net_pnl_for_perf = isolated_pnl - estimated_fee
                    self._update_strategy_performance(strat_id, net_pnl_for_perf)
                    self._update_kelly_stats(net_pnl_for_perf) # Phase 14: Dynamic Kelly tracking
                    
                    trade_data = getattr(self, '_last_closed_trade_data', None)
                    duration = trade_data['duration_seconds'] if trade_data else 0.0
                    exit_reason = trade_data['exit_reason'] if trade_data else getattr(event, 'exit_reason', 'NORMAL_CLOSE')
                    self._sophia_post_mortem_check(event, isolated_pnl, duration, exit_reason)
                    
                    logger.info(f"📈 Trade Closed: {event.symbol} (Isolated Horizon PnL: ${isolated_pnl:.2f})")
                    self.log_trade_report(event, pnl=isolated_pnl, fill_price=fill_price)
                    pnl_realized = isolated_pnl
                else:
                    self.log_trade_report(event, pnl=None, fill_price=fill_price)
                    pnl_realized = None

                # CRITERIO-AXIOMA Accounting Audit
                # 🪐 [OMEGA PHASE 9] SYNC CASH TO RUST PORTFOLIO
                try:
                    from core.rust_execution_bridge import ffi_update_portfolio_bridge
                    ffi_update_portfolio_bridge(float(self.current_cash))
                except Exception as e:
                    pass

                post_balance = Decimal(str(self.current_cash))
                expected_balance = pre_balance - Decimal(str(estimated_fee)) + Decimal(str(isolated_pnl))
                
                if Config.BINANCE_USE_FUTURES is False:
                    if isolated_pnl != 0.0:
                        sold_cash_returned = Decimal(str((event.quantity * fill_price) + isolated_pnl))
                        expected_balance = pre_balance - Decimal(str(estimated_fee)) + sold_cash_returned
                    else:
                        from_cash_spent = Decimal(str(fill_cost))
                        expected_balance = pre_balance - Decimal(str(estimated_fee)) - from_cash_spent
                        
                drift = abs(post_balance - expected_balance)
                self.precision_drift_accumulated += drift
                
                if drift > Decimal('1e-8'):
                     logger.warning(f"⚠️ [AXIOMA-LOG] Precision Drift Detected: Max Deviation {drift:.4e}")
                
                if self.precision_drift_accumulated > (Decimal(str(self.initial_capital)) * Decimal('0.00001')):
                     logger.critical(f"🛑 [AXIOMA-LOG] Accumulated Drift Exceeds Tolerance: {self.precision_drift_accumulated:.6e}. Force Sync required.")
                
                self.verify_accounting_equation()
                
            finally:
                self.guard.release()
            
            # ═══════════════════════════════════════════════════════════════
            # FORENSIC-V29 FIX #3: IMMEDIATE EQUITY CACHE REFRESH
            # QUÉ: Refresca _equity_cache inmediatamente después de cada fill.
            # POR QUÉ: Sin esto, el equity cache queda desactualizado entre
            #   fills, causando que _get_available_cash_internal() calcule
            #   horizon silos con equity stale → sizing inconsistente.
            # PARA QUÉ: Precisión atómica en partición 60/40 para micro-cuenta.
            # CUÁNDO: Después de cada fill procesado.
            # ═══════════════════════════════════════════════════════════════
            self._refresh_equity_cache()
            
            # 🚀 FASE 12: HYPER-FREQUENCY CAPITAL RECYCLING
            if hasattr(self, 'compounding_engine') and self.compounding_engine:
                self.compounding_engine.force_recalc()
            
            # Phase 5: Trigger Secuestro Si Corresponde
            self._check_auto_secuestro()
            
            # ════════════════════════════════════════════════════════════════
            # CTOS PHASE 2: PORTFOLIO → SSOT SYNC
            # QUÉ: Publica equity, margin y posiciones activas al SSOT global.
            # POR QUÉ: Sin esto, global_state.portfolio está vacío y todos los
            #   consumidores (MetaCoordinator, Invariants, FeedbackProcessor)
            #   operan con datos ficticios (equity=0, positions=0).
            # PARA QUÉ: Coherencia total → el sistema sabe cuánto tiene y
            #   dónde está invertido en todo momento.
            # CÓMO: Llama update_portfolio_snapshot() + sincroniza active_positions.
            # CUÁNDO: Después de cada fill procesado (post _refresh_equity_cache).
            # DÓNDE: core/portfolio.py → update_fill()
            # QUIÉN: Portfolio (escritor) → GlobalMarketState (receptor)
            # ════════════════════════════════════════════════════════════════
            try:
                from core.global_state import global_state
                # O(1) Fast path logic instead of sum(1 for ...)
                _open_count = sum(1 for v in self.virtual_ledger.values() if v['quantity'] != 0) if self.virtual_ledger else 0
                _equity = self.get_total_equity()
                global_state.update_portfolio_snapshot(
                    equity=_equity,
                    margin_used=self.used_margin,
                    positions_count=_open_count,
                    unrealized_pnl=_equity - self.initial_capital - self.realized_pnl,
                    realized_pnl=self.realized_pnl
                )
                # Sync active positions to SSOT for downstream consumers
                _active = {}
                for _vk, _vp in self.virtual_ledger.items():
                    if _vp['quantity'] != 0:
                        _parts = _vk.split('_')
                        _sym = _parts[0] if _parts else event.symbol
                        _dir = 'LONG' if _vp['quantity'] > 0 else 'SHORT'
                        from core.structs import PositionState
                        _active[_vk] = PositionState(
                            symbol=_sym, direction=_dir,
                            quantity=abs(_vp['quantity']),
                            entry_price=_vp['avg_price'],
                            current_price=_vp['current_price'],
                            horizon=_vp.get('horizon', 'UNKNOWN')
                        )
                global_state.active_positions = _active
            except Exception as _ssot_e:
                logger.debug(f"[SSOT] Portfolio sync skipped: {_ssot_e}")
            
            meta = getattr(event, 'metadata', {}) or {}
            is_close = getattr(event, 'is_close', False) or getattr(event, 'is_exit', False) or meta['is_close'] or meta['is_exit']
            
            # Log Trade
            details_dict = {
                'margin': margin_impact,
                'setup': getattr(event, 'setup_type', 'UNKNOWN'),
                'is_close': is_close
            }
            if is_close and hasattr(self, '_last_closed_trade_data') and self._last_closed_trade_data:
                td = self._last_closed_trade_data
                details_dict.update({
                    'pnl': td['net_pnl'],
                    'fees': td['fees_paid'],
                    'mfe_pct': td['mfe_pct'],
                    'mae_pct': td['mae_pct'],
                    'duration_s': td['duration_seconds'],
                    'exit_reason': td['exit_reason'],
                    'ml_confidence': td['oracle_certainty']
                })

            trade_id_val = getattr(event, 'trade_id', None)

            self.log_to_csv({
                'datetime': self._get_current_time(),
                'symbol': event.symbol,
                'type': "FILL_CLOSE" if is_close else "FILL_OPEN",
                'direction': event.direction,
                'quantity': event.quantity,
                'price': fill_price,
                'fill_cost': event.quantity * fill_price,
                'strategy_id': getattr(event, 'strategy_id', 'Unknown'),
                'setup_type': getattr(event, 'setup_type', 'UNKNOWN'),
                'strategy_version': getattr(event, 'strategy_version', '1.0.0'),
                'trade_id': trade_id_val,
                'details': json.dumps(details_dict)
            })

            # ATOMIC DB UPDATE (Rule 5.2) - Done outside spinlock to avoid blocking loop
            trade_payload = {
                'symbol': event.symbol,
                'side': event.direction,
                'quantity': event.quantity,
                'price': fill_price,
                'timestamp': self._get_current_time(),
                'order_type': OrderType.MARKET,
                'strategy_id': getattr(event, 'strategy_id', 'Unknown'),
                'pnl': pnl_realized if pnl_realized is not None else 0.0,  # FORENSIC-V21 FIX #1
                'commission': estimated_fee
            }
            direction_val = event.direction.name if hasattr(event.direction, 'name') else str(event.direction)
            
            if is_close:
                pos_side = 'LONG' if direction_val == 'SELL' else 'SHORT'
            else:
                pos_side = 'LONG' if direction_val == 'BUY' else 'SHORT'

            v_key = f"{event.symbol}_{getattr(event, 'horizon', 'SCALPING')}_{pos_side}"
            v_pos = self.virtual_ledger[v_key]
            position_payload = {
                'symbol': event.symbol,
                'quantity': v_pos['quantity'],
                'entry_price': v_pos['avg_price'],
                'current_price': fill_price,
                'pnl': pnl_realized if pnl_realized is not None else 0.0,  # FORENSIC-V21 FIX #1
                'sl_pct': v_pos['sl_pct'],
                'tp_pct': v_pos['tp_pct'],
                'horizon': getattr(event, 'horizon', 'SCALPING'),
                'strategy_id': getattr(event, 'strategy_id', 'Unknown')
            }
            
            self.io_executor.submit(self.db.log_fill_event_atomic, trade_payload, position_payload)
            
            if self.auto_save:
                self.save_status()
                
            if pnl_realized is not None:
                return (pnl_realized, outcome_obj)
            return None

    def verify_accounting_equation(self):
        """
        CRITERIO-AXIOMA Protocol: Ley de Conservación
        Verifica que el Dinero no aparece ni desaparece de la nada.
        """
        import math
        # Calculate theoretical settled balance
        theoretical_settled = self.initial_capital + self.realized_pnl - self.total_fees_paid
        
        if Config.BINANCE_USE_FUTURES:
            # In Futures, current_cash tracks total wallet equity (Initial + PnL - Fees)
            actual_settled = self.current_cash
        else:
            # In Spot, current_cash is free margin, so we add back the cost basis of open positions
            open_cost = sum(abs(p['quantity']) * p['avg_price'] for p in self.virtual_ledger.values())
            actual_settled = self.current_cash + open_cost
            
        try:
            if math.isnan(theoretical_settled) or math.isnan(actual_settled):
                return # Avoid cascading decimal Parse exceptions
                
            from decimal import Decimal
            d_theoretical = Decimal(f"{theoretical_settled:.8f}")
            d_actual = Decimal(f"{actual_settled:.8f}")
            delta = abs(d_theoretical - d_actual)
            
            if delta > PrecisionAuditor.STRICT_EPSILON:
                logger.error(f"🚨 [AXIOMA-FATAL] CORRUPCIÓN CONTABLE DETECTADA!")
                logger.error(f"   Teórico:  ${theoretical_settled:.8f}")
                logger.error(f"   Real:     ${actual_settled:.8f}")
                logger.error(f"   Delta:    ${float(delta):f}")
                logger.error(f"   Initial={self.initial_capital}, PnL={self.realized_pnl}, Fees={self.total_fees_paid}")
                
                # Soft-kill the engine by poisoning PnL
                logger.error("☠️ SISTEMA COMPROMETIDO. CÁLCULOS INVÁLIDOS.")
                self.realized_pnl = float('nan')
        except Exception as e:
            logger.error(f"⚠️ [AXIOMA] Falló la auditoría contable: {e}")

    def _sophia_post_mortem_check(self, event, pnl: float, duration_seconds: float = 0.0, exit_reason: str = ""):
        """
        SOPHIA-INTELLIGENCE + NÉMESIS-RETROSPECCIÓN + XAI AUTOPSY:
        1. Compute SOPHIA Brier Score (basic post-mortem)
        2. Run NÉMESIS full autopsy (deep diagnosis)
        3. Inject XAI forensic summary into notification payload
        Called from update_fill() when a position is closed (SHORT or LONG).
        """
        try:
            trade_id = getattr(event, 'trade_id', None)
            if not trade_id:
                # Try to find by symbol match
                for tid in list(self.sophia_post_mortem.pending_intents.keys()):
                    intent = self.sophia_post_mortem.pending_intents[tid]
                    if intent.symbol == event.symbol:
                        trade_id = tid
                        break
            
            if trade_id:
                # Retrieve intent BEFORE compute_post_mortem pops it
                intent = self.sophia_post_mortem.pending_intents[trade_id]
                
                result = self.sophia_post_mortem.compute_post_mortem(
                    trade_id=trade_id,
                    actual_pnl=pnl,
                    duration_seconds=duration_seconds,
                    exit_reason=exit_reason,
                )
                if result:
                    # Phase 46: Trigger Meta-Optimizer for Sovereign Evolution
                    try:
                        meta_optimizer.process_trade_result(result)
                    except Exception as e:
                        logger.debug(f"[META] Optimization skipped: {e}")

                    # ═══════════════════════════════════════════════════════════════
                    # XAI AUTOPSY INJECTION (Phase Omega)
                    # QUÉ: Inyecta un resumen forense legible en el payload de
                    #   notificación para que Telegram muestre POR QUÉ perdimos.
                    # POR QUÉ: Sin esto, solo vemos "$-0.01" pero no sabemos si fue
                    #   culpa del RSI, del régimen, o de la calibración Sophia.
                    # PARA QUÉ: Feedback loop humano + auto-aprendizaje evolutivo.
                    # CÓMO: Construye un mini-reporte con Brier Score, top features,
                    #   y la narrativa de Sophia, y lo pega en _last_closed_trade_data.
                    # CUÁNDO: En cada cierre de trade que tenga intent registrado.
                    # DÓNDE: core/portfolio.py → _sophia_post_mortem_check
                    # QUIÉN: PostMortemComparator + NemesisEngine + XAIEngine
                    # ═══════════════════════════════════════════════════════════════
                    try:
                        xai_lines = []
                        xai_lines.append(f"🔬 Brier: {result.brier_score:.3f}")
                        
                        # Calibration quality label
                        if result.brier_score < 0.05:
                            xai_lines.append("Calibración: 🎯 EXCELENTE")
                        elif result.brier_score < 0.15:
                            xai_lines.append("Calibración: ✅ BUENA")
                        elif result.brier_score < 0.25:
                            xai_lines.append("Calibración: ⚠️ DEGRADADA")
                        else:
                            xai_lines.append("Calibración: ❌ CRÍTICA")
                        
                        # Predicted vs Actual
                        xai_lines.append(f"Predicho: WP={result.predicted_prob*100:.0f}% | Duración={result.predicted_exit_mins:.1f}min")
                        xai_lines.append(f"Real: {result.actual_outcome} | Duración={result.actual_duration_mins:.1f}min")
                        
                        # Time error
                        if result.time_error_mins > 5.0:
                            xai_lines.append(f"⏱️ Error temporal: {result.time_error_mins:.1f}min (modelo desincronizado)")
                        
                        # Top features from intent
                        if intent and intent.top_features:
                            top_3 = intent.top_features[:3]
                            feat_str = ", ".join(
                                f"{f['name']}={f['value']:.2f}" 
                                for f in top_3 if isinstance(f, dict)
                            )
                            if feat_str:
                                xai_lines.append(f"Drivers: {feat_str}")
                        
                        xai_summary = "\n".join(xai_lines)
                        
                        # Inject into notification payload
                        closed_data = getattr(self, '_last_closed_trade_data', None)
                        if closed_data and closed_data['symbol'] == event.symbol:
                            closed_data['xai_autopsy'] = xai_summary
                            closed_data['brier_score'] = result.brier_score
                            closed_data['sophia_narrative'] = result.narrative
                    except Exception as xai_e:
                        logger.debug(f"[XAI] Autopsy injection skipped: {xai_e}")

                    # Update SOPHIA calibrator if available
                    won = pnl > 0
                    # Log calibration status periodically
                    if self.sophia_post_mortem.total_trades % 10 == 0:
                        cal_log = self.sophia_post_mortem.get_summary_log()
                        logger.info(f"   📊 {cal_log}")
                    
                    # ── NÉMESIS-RETROSPECCIÓN: Full Autopsy ──
                    if intent:
                        try:
                            fill_price = getattr(event, 'fill_price', 0.0) or 0.0
                            sophia_data = intent.sophia_report or {}
                            
                            # CRITICAL FIX: Inject real execution horizon so Nemesis doesn't fallback to SHORT_TERM
                            if 'horizon_profile' not in sophia_data:
                                sophia_data['horizon_profile'] = getattr(event, 'horizon', 'SCALPING')
                            
                            nemesis_report = self.nemesis_engine.full_autopsy(
                                trade_id=trade_id,
                                symbol=event.symbol,
                                direction=intent.direction,
                                predicted_prob=intent.win_probability,
                                predicted_exit_mins=intent.expected_exit_mins,
                                predicted_tp_mins=sophia_data['time_to_tp_mins'],
                                predicted_sl_mins=sophia_data['time_to_sl_mins'],
                                actual_pnl=pnl,
                                actual_duration_mins=duration_seconds / 60.0,
                                brier_score=result.brier_score,
                                sophia_report=sophia_data,
                                top_features=intent.top_features,
                                trigger_price=intent.trigger_price,
                                fill_price=fill_price,
                            )
                            
                            # Apply overconfidence penalty to next signals
                            if nemesis_report.overconfidence_active:
                                logger.info(
                                    f"   ⚠️ [NÉMESIS] Confidence penalty active: "
                                    f"{nemesis_report.overconfidence_penalty_factor:.2f}x"
                                )
                            
                            # Log NÉMESIS health every 10 trades
                            if self.sophia_post_mortem.total_trades % 10 == 0:
                                health = self.nemesis_engine.get_calibration_health()
                                logger.info(f"   ⚔️ [NÉMESIS] Health: {health}")
                        except Exception as e:
                            logger.debug(f"[NÉMESIS] Autopsy skipped: {e}")
        except Exception as e:
            logger.debug(f"[SOPHIA] Post-mortem skipped: {e}")

    def close(self):
        """
        Graceful shutdown for portfolio resources.
        """
        # FORENSIC FIX: Log session telemetry before closing DB
        try:
            if hasattr(self, 'db') and self.db and hasattr(self, '_session_stats'):
                session_id = f"SES_{self._session_stats['start_time'].replace(':', '').replace('-', '').split('.')[0]}"
                end_equity = self.current_cash + self.realized_pnl
                for pos in self.virtual_ledger.values():
                    if 'unrealized_pnl' in pos:
                        end_equity += pos['unrealized_pnl']
                
                # Derive symbols traded
                symbols = list(set([k.split('_')[0] for k in self.virtual_ledger.keys()]))
                
                self.db.log_session(
                    session_id=session_id,
                    start_equity=self._session_start_equity,
                    end_equity=end_equity,
                    total_trades=self._session_stats['total'],
                    wins=self._session_stats['wins'],
                    losses=self._session_stats['losses'],
                    gross_pnl=self._session_stats['gross_pnl'],
                    total_fees=getattr(self, 'total_fees_paid', 0.0),
                    net_pnl=self._session_stats['net_pnl'],
                    best_trade_pnl=0.0,  # Could be tracked individually
                    worst_trade_pnl=0.0,
                    avg_trade_duration_sec=0.0,
                    symbols_traded=",".join(symbols) if symbols else "NONE",
                    start_time=self._session_stats['start_time']
                )
        except Exception as e:
            logger.debug(f"[CTOS] Session log skipped: {e}")

        if hasattr(self, 'io_executor') and self.io_executor:
            self.io_executor.shutdown(wait=True)
        if hasattr(self, 'db') and self.db:
            self.db.close()
        logger.info("✅ Portfolio: Shutdown complete.")

    def save_status(self):
        """
        Save current portfolio state.
        Uses ThreadPoolExecutor to prevent I/O blocking.
        """
        # Snapshot state safely
        self.guard.acquire()
        try:
            cash_snapshot = self.current_cash
            realized_snapshot = self.realized_pnl
            positions_snapshot = self.positions.copy()
            # Serialize virtual ledger (remove non-serializable objects like datetime if any, though it should be primitive)
            vl_snapshot = {}
            for k, v in self.virtual_ledger.items():
                vl_copy = v.copy()
                if 'entry_time' in vl_copy and hasattr(vl_copy['entry_time'], 'isoformat'):
                    vl_copy['entry_time'] = vl_copy['entry_time'].isoformat()
                vl_snapshot[k] = vl_copy
        finally:
            self.guard.release()
            
        equity = self.get_total_equity()
        unrealized_pnl = equity - self.initial_capital - realized_snapshot
        
        # Submit to Executor
        self.io_executor.submit(self._do_save_status, cash_snapshot, realized_snapshot, positions_snapshot, equity, unrealized_pnl, vl_snapshot)

    def _do_save_status(self, cash, realized, positions, equity, unrealized, vl_snapshot):
        """Worker method for save_status: Executed in background thread"""
        try:
            # Get Session Info (Disabled: Decoupled for tests)
            session_id = None
            
            metrics = {} 
            
            # SPECTACULAR OPTIMIZATION: Heavy Math is now here, off the main Event Loop!
            try:
                # Load trades safely for Expectancy
                trades_path = os.path.join(os.path.dirname(self.status_path), "trades.csv")
                # Use thread-safe read (F43 Fix)
                trades_df = safe_read_csv(trades_path)
                
                if trades_df is not None:
                    
                    # Lazy Import to avoid circular dependencies
                    from utils.analytics import AnalyticsEngine 
                    
                    # Complex generic calculations
                    exp_stats = AnalyticsEngine.calculate_expectancy(trades_df)
                    metrics.update(exp_stats)
                    
                    # Also calculate Sharpe/Sortino if enough data (approximate)
                    if len(trades_df) > 30:
                        # Assume 1m signals for simplicity or use PnL curve
                        pass # Keep it simple for now to avoid crashes
            except Exception as e:
                # Don't fail the save if analytics fail
                logger.debug(f"Async Analytics Calc Skipped: {e}")

            def _calculate_kelly_fraction(self) -> float:
                """
                [FASE 8: GESTIÓN DE RIESGO AVANZADA]
                Calcula la fracción de Kelly usando un C-optimized Nano Core si está disponible.
                """
                # Minimum trades required for statistically significant Kelly calculation
                if len(self.kelly_trades_history) < 5:
                    return 0.01  # Use base minimal risk
                    
                wins = sum(1 for t in self.kelly_trades_history if t['is_win'])
                total = len(self.kelly_trades_history)
                
                self.kelly_winrate = wins / total
                
                # Calculate Payoff Ratio (Avg Win / Avg Loss)
                gross_profits = sum(t['pnl'] for t in self.kelly_trades_history if t['is_win'])
                gross_losses = abs(sum(t['pnl'] for t in self.kelly_trades_history if not t['is_win']))
                
                avg_win = gross_profits / wins if wins > 0 else 0
                avg_loss = gross_losses / (total - wins) if (total - wins) > 0 else 0
                
                self.kelly_payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
                
                if calculate_kelly_fraction is not None:
                    try:
                        return calculate_kelly_fraction(
                            self._win_streak,
                            self._loss_streak,
                            float(self.kelly_winrate),
                            float(self.kelly_payoff_ratio),
                            0.5, # max_kelly
                            float(getattr(self, 'current_brier_score', 100.0)), # using brier score as stress approx
                            True
                        )
                    except Exception as e:
                        from core.exceptions import SystemIntegrityError
                        raise SystemIntegrityError(f"Cálculo de Kelly fallido de forma silenciosa para {self._win_streak}/{self._loss_streak} con winrate {self.kelly_winrate}") from e
                if self.kelly_winrate <= 0.0 or self.kelly_payoff_ratio <= 0.0:
                    return 0.01
                    
                kelly_pct = self.kelly_winrate - ((1.0 - self.kelly_winrate) / self.kelly_payoff_ratio)
                
                # Anti-Martingale / Streak Logic Fallback
                if self._win_streak > 0:
                    multiplier = 1.0 + (min(self._win_streak, 5) * 0.1)  # Up to 50% boost on win streaks
                    kelly_pct *= multiplier
                elif self._loss_streak > 0:
                    divisor = 1.0 + (min(self._loss_streak, 5) * 0.2)    # Cut risk significantly on losing streaks
                    kelly_pct /= divisor
                    
                if kelly_pct <= 0.0:
                    return 0.01  # Minimum risk floor
                    
                # ═══════════════════════════════════════════════════════════════
                # TECHO TERMODINÁMICO DURO: 25% MAXIMUM (ANTI-RUINA)
                # ═══════════════════════════════════════════════════════════════
                return min(kelly_pct, 0.25)

            json_data = {
                'timestamp': self._get_current_time().isoformat(),
                'session_id': session_id,
                'total_equity': cash + realized, # nano_core.calculate_total_equity does not exist
                'cash': cash,
                'realized_pnl': realized,
                'unrealized_pnl': 0.0, # Will be handled by the real portfolio update
                'positions': positions,
                'virtual_ledger': vl_snapshot,
                'performance_metrics': metrics, # Now populated!
                'balance': cash, 
                'precision_drift': float(getattr(self, 'precision_drift_accumulated', 0.0)),
                'brier_score_active': float(getattr(self, 'current_brier_score', 0.0)),
                'ppo_entropy_active': float(getattr(self, 'current_ppo_entropy', 0.0)),
                'last_heartbeat': self._get_current_time().isoformat()
            }
            
            # 🛡️ PHASE 27: ATOMIC PERSISTENCE (Nadir-Soberano)
            # Replaces legacy handler with AtomicStateManager
            json_path = os.path.join(os.path.dirname(self.status_path), "live_status.json")
            AtomicStateManager.save_json_atomic(json_path, json_data)
        except Exception as e:
            logger.error(f"Async Status Save Failed: {e}")

    def _sync_log_to_csv(self, data):
        # Internal synchronous method for CSV writing
        try:
            # Use thread-safe append (F43 Fix)
            safe_append_csv(self.csv_path, data)
        except Exception as e:
            logger.error(f"CSV Log Failed: {e}")


    @trace_execution
    def check_exits(self, data_provider, events_queue):
        """
        Legacy Portfolio-based exit monitoring.
        DEPRECATED: Risk management has been centralized in RiskManager.check_stops().
        This method is now a no-op to prevent duplicate exit signal emissions with conflicting thresholds.
        """
        pass


    @trace_execution
    def validate_notional_physics(self, margin: float, leverage: int, min_notional: float = 5.05) -> bool:
        """
        [FASE 4 SUPREMACÍA NATIVA] Restricción dura de capital físico.
        Si el notional (margin * leverage) es menor al límite físico de Binance ($5.05),
        la operación es rechazada estructuralmente a nivel de portfolio.
        """
        notional = margin * leverage
        if notional < min_notional:
            logger.warning(
                f"🛡️ [PHYSICS-FILTER] Rechazo estructural. Notional ${notional:.2f} "
                f"(Margen ${margin:.2f} x Lev {leverage}) < Mínimo ${min_notional:.2f}"
            )
            return False
        return True

    def get_smart_kelly_sizing(self, symbol: str, strategy_id: str, is_micro_account: bool = False, horizon: str = "SCALPING") -> float:
        """
        [PHASE 3.2] Kelly Criterion Dinámico para Micro-Cuentas
        QUÉ: Calcula la fracción óptima del capital a arriesgar/invertir.
        POR QUÉ: Maximiza el crecimiento geométrico. En micro-cuentas ($13),
           aplicamos Full Kelly (agresivo) con techo alto para escapar rápido del fondo.
           En cuentas estándar (>= $50), usamos Half-Kelly o Quarter-Kelly por seguridad.
        """
        perf = self.strategy_performance[strategy_id]
        total_trades = perf['trades']
        
        # Default allocation ratios when lacking data
        if total_trades < 5: 
            if self.kelly_winrate > 0.0:
                win_rate = self.kelly_winrate
                loss_rate = 1.0 - win_rate
                b = self.kelly_payoff_ratio if self.kelly_payoff_ratio > 0.0 else 2.0
                kelly_f = win_rate - (loss_rate / b)
                if is_micro_account:
                    multiplier = 1.0 if horizon == "MICROSCALPING" else 0.8
                    return max(0.02, min(0.25, kelly_f * multiplier)) # 🛑 LEY DE LA RUINA: Techo termodinámico 25%
                else:
                    return max(0.01, min(0.25, kelly_f * 0.5)) # Fallback a 0.01 mínimo
            return 0.25 if is_micro_account else 0.05
            
        wins = perf['wins']
        losses = perf['losses']
        
        if wins == 0: return 0.25 if is_micro_account else 0.01  # LEY DE LA RUINA: 25% inicial en $13
        if losses == 0: return 0.25 if is_micro_account else 0.10 # Perfect streak, capped at 25%
        
        win_rate = wins / total_trades
        loss_rate = 1.0 - win_rate
        
        total_win_amt = perf['total_win_pnl']
        total_loss_amt = perf['total_loss_pnl']
        
        avg_win = total_win_amt / wins if wins > 0 else 0.0
        avg_loss = total_loss_amt / losses if losses > 0 else 1.0
        
        b = avg_win / avg_loss if avg_loss > 0 else 2.0
        
        if b <= 0: return 0.10 if is_micro_account else 0.01
        
        # 🚀 FASE 13: QUANTUM STREAK SIZING (Anti-Martingale)
        streak = perf['current_streak']
        losing_streak = perf['losing_streak']
        
        volatility_mod = 1.0
        if streak > 0:
            volatility_mod = 1.0 + (streak * 0.1)
        elif losing_streak > 0:
            volatility_mod = max(0.1, 1.0 - (losing_streak * 0.25))
            
        if calculate_kelly_fraction:
            kelly_f, _ = calculate_kelly_fraction(float(win_rate), float(b), float(volatility_mod))
        else:
            kelly_f = max(0.0, (win_rate - (loss_rate / b)) * volatility_mod)
        
        if is_micro_account:
            # 🚀 FASE 12: ASYMMETRIC KELLY ALLOCATION
            # QUÉ: Límites dinámicos por horizonte para cuentas micro.
            # POR QUÉ: Microscalping es la fuerza motriz y puede consumir todo su margen asignado.
            #   Swing debe ser defensivo y operar con menor porción del suyo.
            if horizon in ("MICROSCALPING", "MICRO"):
                multiplier = 1.2
                cap = 0.25 # Techo termodinámico estricto
            elif horizon == "SCALPING":
                multiplier = 1.0
                cap = 0.25 
            else: # SWING
                multiplier = 0.8
                cap = 0.20 
                
            base_size = max(0.02, min(cap, kelly_f * multiplier))
            
            # ⚡ FASE 8: ANTI-MARTINGALE CUÁNTICO
            # QUÉ: Escala el sizing basado en rachas consecutivas.
            streak_mult = 1.0
            if getattr(self, '_win_streak', 0) >= 2:
                streak_mult = min(2.5, 1.0 + (self._win_streak - 1) * 0.15)
            elif getattr(self, '_loss_streak', 0) >= 2:
                streak_mult = max(0.5, 1.0 - (self._loss_streak - 1) * 0.20)
            
            final_size = max(0.02, min(0.25, base_size * streak_mult))
        else:
            # Half-Kelly for Standard accounts
            final_size = max(0.01, min(0.25, kelly_f * 0.5))
            
        import logging
        logging.getLogger("TraderGemini").debug(f"📐 [SMART KELLY] {strategy_id} WR:{win_rate:.2f} B:{b:.2f} KF:{kelly_f:.3f} -> Alloc:{final_size*100:.1f}%")
        
        return final_size

    def log_trade_report(self, event, pnl=None, fill_price=0):
        # Prints a real-time report of the trade execution, Win Rate, and Balance.
        # Now sends enhanced notifications with full trade context (Phase 4.5).
        try:
            # 1. Performance Stats — SESSION (Primary) + ALL-TIME (Secondary)
            # FORENSIC-V35: Use session stats as PRIMARY WR (not lifetime)
            session_wins = self._session_stats['wins']
            session_losses = self._session_stats['losses']
            session_total = self._session_stats['total']
            win_rate = (session_wins / session_total * 100) if session_total > 0 else 0.0
            
            # All-time (secondary display)
            total_wins = sum(d['wins'] for d in self.strategy_performance.values())
            total_losses = sum(d['losses'] for d in self.strategy_performance.values())
            total_trades = sum(d['trades'] for d in self.strategy_performance.values())
            alltime_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
            
            # Strategy-specific Win Rate
            strat_perf = self.strategy_performance[getattr(event, 'strategy_id', 'Unknown')]
            strat_wins = strat_perf['wins']
            strat_losses = strat_perf['losses']
            strat_total = strat_perf['trades']
            strat_win_rate = (strat_wins / strat_total * 100) if strat_total > 0 else 0.0
            
            # 2. Balance Stats
            equity = self.get_total_equity()
            balance_delta = equity - self.initial_capital
            balance_pct = (balance_delta / self.initial_capital) * 100
            
            is_exit = pnl is not None
            horizon = getattr(event, 'horizon', 'SCALPING')
            strategy_id_log = getattr(event, 'strategy_id', 'Unknown')
            if is_exit:
                direction_icon = "🟢 CLOSE SHORT" if event.direction == OrderSide.BUY else "🔴 CLOSE LONG"
            else:
                direction_icon = "🟢 ENTRY LONG" if event.direction == OrderSide.BUY else "🔴 ENTRY SHORT"
                
            pnl_str = f"+${pnl:.2f}" if pnl and pnl > 0 else (f"-${abs(pnl):.2f}" if pnl else "N/A")
            pnl_color = "🟢" if pnl and pnl > 0 else ("🔴" if pnl and pnl < 0 else "⚪")
            
            # FORENSIC-V21 FIX #7: Enriched terminal log with horizon, strategy, SL/TP
            print(f"\n📢 ========= [ TRADE EXECUTION — {horizon} ] =========", flush=True)
            print(f"   {direction_icon} {event.symbol} @ ${fill_price:.4f} (Qty: {event.quantity})", flush=True)
            print(f"   🏷️ Horizon: {horizon} | Strategy: {strategy_id_log}", flush=True)
            
            # FORENSIC FIX #4: Detailed Balance-per-trade tracking
            _meta = getattr(event, 'metadata', {}) or {}
            actual_order_type = _meta['actual_order_type']
            enriched_order_type = _meta['enriched_order_type']
            fee_tag = "Maker" if actual_order_type == 'limit' else "Taker"
            estimated_fee = getattr(event, 'commission', 0.0) or 0.0
            
            leverage = getattr(Config, 'BINANCE_LEVERAGE', 10.0) if getattr(Config, 'BINANCE_USE_FUTURES', False) else 1.0
            notional = event.quantity * fill_price
            margin = notional / leverage
            
            print(f"   📦 Notional Size: ${notional:.2f} ({leverage}x Lev)", flush=True)
            print(f"   💳 Margin Used:   ${margin:.2f}", flush=True)
            print(f"   💸 Fees Paid:     ${estimated_fee:.4f} ({fee_tag}) | Type: {enriched_order_type}", flush=True)
            
            ml_confidence = getattr(event, 'ml_confidence', None)
            predicted_duration = getattr(event, 'predicted_duration', None)
            predicted_magnitude = getattr(event, 'predicted_magnitude', None)
            if ml_confidence is not None:
                prob_str = f"{ml_confidence * 100:.1f}%"
                dur_str = f"| Horizon: {predicted_duration} bars" if predicted_duration else ""
                mag_str = f"| Target: +{predicted_magnitude * 100:.2f}%" if predicted_magnitude else ""
                print(f"   🔮 ML Prediction: Conf: {prob_str} {dur_str} {mag_str}", flush=True)
                
            print(f"   🏦 Available Cash:${self.current_cash:.2f}", flush=True)
            
            # FORENSIC-V23: Add forensic metadata to Terminal log
            confluence = _meta['multi_timeframe_score']
            neural_bias = _meta['neural_bias']
            rsi = _meta['rsi']
            adx = _meta['adx']
            setup_type = _meta['setup_type']
            
            print(f"   🔬 Setup: {setup_type}", flush=True)
            if confluence is not None or neural_bias is not None:
                forensic_str = ""
                if confluence is not None: forensic_str += f"Conf: {confluence:.2f} | "
                if neural_bias is not None: forensic_str += f"Bias: {neural_bias:.2f} | "
                if rsi is not None and adx is not None: forensic_str += f"RSI: {rsi:.1f} ADX: {adx:.1f}"
                print(f"   🧠 Forensic:      {forensic_str.strip(' |')}", flush=True)
            if pnl is not None:
                print(f"   💰 PnL Realized:  {pnl_color} {pnl_str}", flush=True)
            
            # FORENSIC-V21 FIX #7: Show SL/TP targets
            _vkey_base = f"{event.symbol}_{horizon}"
            _pos_temp = (
                self.virtual_ledger.get(f"{_vkey_base}_LONG") or
                self.virtual_ledger.get(f"{_vkey_base}_SHORT") or
                self.virtual_ledger.get(_vkey_base) or
                self.positions.get(event.symbol) or {}
            )
            sl_display = _pos_temp.get('sl_pct') if isinstance(_pos_temp, dict) else None
            tp_display = _pos_temp.get('tp_pct') if isinstance(_pos_temp, dict) else None
            if sl_display or tp_display:
                sl_str = f"{float(sl_display)*100:.2f}%" if sl_display else "N/A"
                tp_str = f"{float(tp_display)*100:.2f}%" if tp_display else "N/A"
                print(f"   🎯 SL: {sl_str} | TP: {tp_str}", flush=True)
            
            # ═══════════════════════════════════════════════════════════════
            # SOPHIA-GLOBAL FIX: ACCURATE WR DISPLAY
            # QUÉ: WR siempre muestra desglose completo session + all-time.
            # POR QUÉ: Si el primer trade de la sesión pierde, WR debe ser 0%.
            #   Antes podía heredar stats de sesiones anteriores por crash recovery.
            # PARA QUÉ: Telegram muestra datos 100% verificables.
            # CÓMO: Para ENTRIES → no mostramos WR (no hay resultado aún).
            #   Para CLOSES → WR incluye el trade actual (ya contabilizado).
            # ═══════════════════════════════════════════════════════════════
            if is_exit and session_total > 0:
                print(f"   🏆 WR Sesión: {win_rate:.1f}% ({session_wins}W/{session_losses}L de {session_total}) | All-Time: {alltime_win_rate:.1f}% ({total_wins}W/{total_losses}L)", flush=True)
            elif not is_exit:
                print(f"   📋 Trades en Sesión: {session_total} | All-Time: {total_trades}", flush=True)
            print(f"   🏦 Cumulative Account Balance: ${equity:.2f} ({'+' if balance_delta >=0 else ''}{balance_pct:.2f}%)", flush=True)
            print(f"========= [ /{horizon} — {strategy_id_log} ] =========\n", flush=True)
            
            # --- ENHANCED NOTIFICATIONS (Phase 4.5 + FORENSIC-V21) ---
            # Build enriched trade data dict for the enhanced notifier
            strategy_id = strategy_id_log
            # horizon already resolved above (FORENSIC-V21 FIX #7)
            commission = getattr(event, 'commission', 0.0) or 0.0
            
            # Re-evaluate all open positions relative to the anchor
            now = self._get_current_time()
            pos = _pos_temp
            sl_pct = pos['sl_pct'] or 0.0
            tp_pct = pos['tp_pct'] or 0.0
            entry_time = pos['entry_time']
            duration_str = 'N/A'
            if entry_time:
                # ── AEGIS-V16 FIX: USE SIMULATED CLOCK FOR DURATION ──
                dur_secs = (self._get_current_time() - entry_time).total_seconds()
                if dur_secs < 0: dur_secs = 0  # Safety fallback
                if dur_secs < 60:
                    duration_str = f"{dur_secs:.0f}s"
                elif dur_secs < 3600:
                    duration_str = f"{dur_secs/60:.1f}m"
                else:
                    duration_str = f"{dur_secs/3600:.1f}h"
            
            # Calculate MAE/MFE from watermarks
            entry_price = pos['avg_price']
            hwm = pos['high_water_mark']
            lwm = pos['low_water_mark']
            mfe_pct = 0.0
            mae_pct = 0.0
            if entry_price > 0:
                if pos['quantity'] >= 0:  # LONG
                    mfe_pct = ((hwm - entry_price) / entry_price) * 100 if hwm > entry_price else 0.0
                    mae_pct = ((entry_price - lwm) / entry_price) * 100 if lwm < entry_price else 0.0
                else:  # SHORT
                    mfe_pct = ((entry_price - lwm) / entry_price) * 100 if lwm < entry_price else 0.0
                    mae_pct = ((hwm - entry_price) / entry_price) * 100 if hwm > entry_price else 0.0
            
            is_close = pnl is not None
            
            # ═══════════════════════════════════════════════════════════════
            # FORENSIC-V21 FIX #2: Direction mapping for notifications
            # QUÉ: Para CLOSE, la dirección del trade es la POSICIÓN, no la orden.
            # POR QUÉ: Al cerrar un LONG vendemos (SELL), pero el trade ERA Long.
            #   Antes: SELL→SHORT (incorrecto para cierre)
            #   Ahora: SELL→LONG (correcto: cerramos un LONG)
            # PARA QUÉ: Telegram muestra dirección correcta del trade.
            # ═══════════════════════════════════════════════════════════════
            if is_close:
                # When closing: SELL means we had a LONG, BUY means we had a SHORT
                notif_direction = 'LONG' if event.direction == OrderSide.SELL else 'SHORT'
            else:
                # When opening: BUY = LONG, SELL = SHORT
                notif_direction = 'LONG' if event.direction == OrderSide.BUY else 'SHORT'
            
            # ═══════════════════════════════════════════════════════════════
            # FORENSIC-V21 FIX #5: Use _record_closed_trade data if available
            # QUÉ: Usa los datos computados por _record_closed_trade para
            #   poblar las notificaciones con PnL neto real, fees breakdown,
            #   y duración exacta.
            # POR QUÉ: _record_closed_trade calcula fees dinámicos (Maker vs
            #   Taker), PnL neto preciso, y duración desde entry_time.
            # ═══════════════════════════════════════════════════════════════
            closed_data = getattr(self, '_last_closed_trade_data', None)
            # ═══════════════════════════════════════════════════════════════
            # XAI AUTOPSY PROPAGATION: Extract forensic fields BEFORE clearing
            # QUÉ: Extrae xai_autopsy y sophia_narrative inyectados por
            #   _sophia_post_mortem_check ANTES de borrar _last_closed_trade_data.
            # ═══════════════════════════════════════════════════════════════
            notif_xai_autopsy = None
            notif_sophia_narrative = None
            # ═══════════════════════════════════════════════════════════════
            # FORENSIC FIX #2 (BUG CRÍTICO): EXTRACTION ANTES DE CLEAR
            # QUÉ: Extraer TODOS los datos de prediction_audit, strategy
            #   attribution y exit votes AQUÍ, ANTES de limpiar closed_data.
            # POR QUÉ: Antes, closed_data se limpiaba en L2536, y luego el
            #   bloque de L2579-2599 intentaba leer de closed_data → siempre
            #   era None → predicciones NUNCA aparecían en Telegram.
            # PARA QUÉ: Las notificaciones de cierre ahora mostrarán:
            #   - Predicción de magnitud vs realidad (MFE)
            #   - Tiempo estimado vs real
            #   - Punto óptimo de cierre (MFE price)
            #   - Ganancia perdida (missed profit)
            #   - Quién abrió y quién cerró el trade
            # CÓMO: Extracción atómica en un solo bloque if-else.
            # ═══════════════════════════════════════════════════════════════
            _pred_mag = None
            _pred_dur = None
            _pred_target = None
            _pred_conf = ml_confidence
            _optimal_exit = None
            _missed_profit = None
            _was_pred_correct = None
            _closer_strat = None
            _opener_strat = None
            _exit_ballot = None
            
            if is_close and closed_data and closed_data['symbol'] == event.symbol:
                notif_pnl = closed_data['gross_pnl']
                notif_commission = closed_data['fees_paid']
                notif_exit_reason = closed_data['exit_reason']
                notif_entry_price = closed_data['entry_price']
                notif_duration = f"{closed_data['duration_seconds']:.0f}s"
                notif_net_pnl = closed_data['net_pnl']
                notif_direction = closed_data['direction']
                notif_trade_id = closed_data['trade_id'] or getattr(event, 'trade_id', None) or 'UNKNOWN'
                notif_xai_autopsy = closed_data['xai_autopsy']
                notif_sophia_narrative = closed_data['sophia_narrative']
                
                # PREDICTION AUDIT — Extract BEFORE clear (FORENSIC FIX #2)
                _pred_mag = closed_data['predicted_magnitude']
                _pred_dur = closed_data['predicted_duration_bars']
                _pred_target = closed_data['predicted_target_price']
                _pred_conf = closed_data['prediction_confidence']
                _optimal_exit = closed_data['optimal_exit_price']
                _missed_profit = closed_data['missed_profit_pct']
                _was_pred_correct = closed_data['was_prediction_correct']
                
                # STRATEGY ATTRIBUTION — Extract BEFORE clear
                _closer_strat = closed_data['closer_strategy_id']
                _opener_strat = closed_data['opener_strategy_id']
                
                # EXIT BALLOT — Extract BEFORE clear  
                _exit_ballot = closed_data['exit_ballot']
                
                # NOW clear after ALL extraction is complete
                self._last_closed_trade_data = None
            else:
                notif_pnl = pnl if pnl is not None else 0.0
                notif_commission = commission
                notif_exit_reason = getattr(event, 'exit_reason', 'NORMAL_CLOSE')
                notif_entry_price = entry_price
                notif_duration = duration_str
                notif_net_pnl = notif_pnl - notif_commission  # FORENSIC-V23: Fixed 0.000 net_pnl fallback
                # ═══════════════════════════════════════════════════════════════
                # FORENSIC-V42 FIX: GUARANTEED TRADE_ID FOR ALL NOTIFICATIONS
                # QUÉ: Genera UUID de respaldo si FillEvent no trae trade_id.
                # POR QUÉ: Antes mostraba "UNKNOWN" o None en Telegram para OPENs
                #   porque la cadena Signal→RiskManager→Order no siempre propagaba
                #   trade_id (ahora sí, pero mantenemos fallback defensivo).
                # PARA QUÉ: 100% trazabilidad de cada trade en Telegram.
                # ═══════════════════════════════════════════════════════════════
                import uuid as _uuid
                _event_tid = getattr(event, 'trade_id', None)
                # Try virtual_ledger as intermediate source
                if not _event_tid:
                    _pos_side_guess = 'LONG' if event.direction == OrderSide.BUY else 'SHORT'
                    _vkey_guess = f"{event.symbol}_{horizon}_{_pos_side_guess}"
                    _vpos = self.virtual_ledger.get(_vkey_guess, {})
                    _event_tid = _vpos.get('trade_id') if isinstance(_vpos, dict) else None
                notif_trade_id = _event_tid or str(_uuid.uuid4())
                
                # For ENTRY events, extract predictions from the event/metadata itself
                if not is_close:
                    _pred_mag = predicted_magnitude
                    _pred_dur = predicted_duration
                    _pred_conf = ml_confidence
            
            # ═══════════════════════════════════════════════════════════════
            # CTOS PHASE 3: ACCURATE BALANCE TRACKING
            # QUÉ: balance_before = balance_after - net_pnl
            # POR QUÉ: Usar un mapa de equity pre-trade fallaba espectacularmente
            #   cuando múltiples trades superpuestos fluctuaban en M2M. Hacía que
            #   la suma de (after - before) no cuadrara con el net_pnl.
            # PARA QUÉ: Telegram muestra balance lógico y consistente.
            # ═══════════════════════════════════════════════════════════════
            if is_close:
                _balance_after = self.get_total_equity()
                _balance_before = _balance_after - notif_net_pnl
                # Remove from map just to prevent memory leak, though we no longer use it
                self._pre_trade_equity_map.pop(notif_trade_id, None)
            else:
                _balance_before = self.get_total_equity()
                _balance_after = _balance_before  # No PnL on entry
                self._pre_trade_equity_map[notif_trade_id] = _balance_before
            
            # Size in USD
            _open_size_usd = event.quantity * notif_entry_price if notif_entry_price else 0.0
            _close_size_usd = event.quantity * fill_price if is_close else 0.0
            
            # Daily growth progress
            _session_growth_pct = ((equity - self._session_start_equity) / self._session_start_equity * 100) if self._session_start_equity > 0 else 0.0
            _daily_target_pct = self._daily_growth_target * 100  # 4.73%
            _growth_progress = max(0.0, min(1.0, _session_growth_pct / _daily_target_pct)) if _daily_target_pct > 0 else 0.0
            
            # CTOS Phase 5: Growth Roadmap
            _compounding_engine = get_compounding_engine(self.initial_capital)
            _avg_net_pnl = sum([t['net_pnl'] for t in self.trade_history if t['net_pnl'] > 0]) / max(1, sum(1 for t in self.trade_history if t['net_pnl'] > 0))
            _roadmap = _compounding_engine.get_growth_roadmap(
                current_equity=self.get_total_equity(),
                current_day=1,  # TODO: Track actual campaign day
                avg_net_pnl_per_trade=_avg_net_pnl,
                trades_today=session_wins + session_losses if 'session_wins' in locals() else 0
            )
            
            trade_notification_data = {
                'trade_id': notif_trade_id,
                'symbol': event.symbol,
                'setup_type': setup_type,
                'strategy': strategy_id,
                'horizon': horizon,
                'direction': notif_direction,
                'order_type': enriched_order_type,
                'entry_price': notif_entry_price,
                'exit_price': fill_price if is_close else 0.0,
                'fill_price': fill_price,
                'quantity': event.quantity,
                'leverage': leverage,
                'margin_used': margin,
                'fee_tag': fee_tag,
                'sl_pct': sl_pct,
                'tp_pct': tp_pct,
                'pnl': notif_pnl,
                'commission': notif_commission,
                'mfe_pct': mfe_pct,
                'mae_pct': mae_pct,
                'duration': notif_duration,
                'exit_reason': notif_exit_reason,
                'ml_confidence': ml_confidence,
                'predicted_duration': predicted_duration,
                'predicted_magnitude': predicted_magnitude,
                # CTOS Phase 3: Corrected balance fields
                'balance_before': _balance_before,
                'balance_after': _balance_after,
                'session_start_equity': self._session_start_equity,
                'session_net_pnl': self._session_net_pnl,
                'session_growth_pct': _session_growth_pct,
                'daily_target_pct': _daily_target_pct,
                'growth_progress': _growth_progress,
                'thought_id': (getattr(event, 'metadata', {}) or {})['thought_id'],
                # CTOS Phase 3: Prediction audit fields
                'prediction_audit': {
                    'predicted_magnitude': _pred_mag,
                    'predicted_duration_bars': _pred_dur,
                    'predicted_target_price': _pred_target,
                    'confidence': _pred_conf,
                    'optimal_exit_price': _optimal_exit,
                    'missed_profit_pct': _missed_profit,
                    'was_correct': _was_pred_correct,
                },
                # CTOS Phase 3: Size tracking
                'open_size_usd': _open_size_usd,
                'close_size_usd': _close_size_usd,
                # CTOS Phase 3: Strategy attribution
                'closer_strategy': _closer_strat,
                'opener_strategy': _opener_strat,
                # FORENSIC FIX #2: Exit ballot (which strategies voted EXIT vs HOLD)
                'exit_ballot': _exit_ballot,
                # CTOS Phase 5: Exponential Compounding
                'growth_roadmap': _roadmap,
                # FORENSIC FIX #2: Diagnostic Stats for Balance section
                'diagnostic_stats': {
                    'avg_win_pnl': sum(d['total_win_pnl'] for d in self.strategy_performance.values()) / max(1, sum(d['wins'] for d in self.strategy_performance.values())) if sum(d['wins'] for d in self.strategy_performance.values()) > 0 else 0.0,
                    'avg_loss_pnl': sum(d['total_loss_pnl'] for d in self.strategy_performance.values()) / max(1, sum(d['losses'] for d in self.strategy_performance.values())) if sum(d['losses'] for d in self.strategy_performance.values()) > 0 else 0.0,
                    'profit_factor': (sum(d['total_win_pnl'] for d in self.strategy_performance.values()) / max(0.001, sum(d['total_loss_pnl'] for d in self.strategy_performance.values()))),
                    'total_session_trades': session_total,
                } if is_close else None,
                # ═══════════════════════════════════════════════════════════════
                # SOPHIA-GLOBAL FIX: WR sentinel for ENTRY notifications
                # QUÉ: Entries send win_rate=-1 so Telegram SKIPS WR display.
                # POR QUÉ: An ENTRY has no outcome yet. Showing "WR: 100%"
                #   after opening a position is misleading and confusing.
                # PARA QUÉ: Only CLOSE notifications show WR (with current trade included).
                # ═══════════════════════════════════════════════════════════════
                'win_rate': win_rate if is_close else -1.0,
                'alltime_win_rate': alltime_win_rate if is_close else -1.0,
                'session_wins': session_wins if is_close else 0,
                'session_losses': session_losses if is_close else 0,
                'strat_win_rate': strat_win_rate if is_close else -1.0,
                'strat_wins': strat_wins if is_close else 0,
                'strat_losses': strat_losses if is_close else 0,
                'volatility': 0.0,  # Populated by caller if available
                'spread': 0.0,
                'metadata': getattr(event, 'metadata', {}) or {}, # 🧠 FORENSIC-V22: Pass telemetry
                'timestamp': datetime.now(timezone.utc).strftime('%H:%M:%S UTC'),
            }
            
            # FORENSIC FIX: The user requested full spam telemetry for all trades,
            # so we no longer suppress BACKTEST_CLOSE notifications.
            if is_close:
                # ═══════════════════════════════════════════════════════════════
                # XAI AUTOPSY FINAL INJECTION: Merge forensic fields into payload
                # ═══════════════════════════════════════════════════════════════
                if notif_xai_autopsy:
                    trade_notification_data['xai_autopsy'] = notif_xai_autopsy
                if notif_sophia_narrative:
                    trade_notification_data['sophia_narrative'] = notif_sophia_narrative
                Notifier.send_trade_close(trade_notification_data)
            else:
                Notifier.send_trade_open(trade_notification_data)
            
            # --- ATOMIC DATABASE LOGGING (Phase 5 CTOS) ---
            trade_dict = {
                'symbol': event.symbol,
                'side': event.direction.name if hasattr(event.direction, 'name') else str(event.direction),
                'quantity': event.quantity,
                'price': fill_price,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'order_type': enriched_order_type,
                'strategy_id': strategy_id,
                'pnl': notif_pnl,
                'commission': notif_commission,
                'trade_id': notif_trade_id,
                'thought_id': (getattr(event, 'metadata', {}) or {})['thought_id']
            }
            
            # Use the position AFTER the fill, which is stored in self.positions/virtual_ledger
            current_pos = _pos_temp
            
            # Use exact v_key to prevent DB overwrites for LONG/SHORT
            _vkey_exact = _vkey_base
            if f"{_vkey_base}_LONG" in self.virtual_ledger:
                _vkey_exact = f"{_vkey_base}_LONG"
            elif f"{_vkey_base}_SHORT" in self.virtual_ledger:
                _vkey_exact = f"{_vkey_base}_SHORT"
                
            position_dict = {
                'symbol': _vkey_exact,
                'quantity': current_pos['quantity'],
                'entry_price': current_pos['avg_price'],
                'current_price': current_pos['current_price'],
                'pnl': current_pos['unrealized_pnl'],
                'sl_pct': current_pos['sl_pct'],
                'tp_pct': current_pos['tp_pct'],
                'horizon': horizon,
                'strategy_id': current_pos['strategy_version']
            }
            
            try:
                self.io_executor.submit(self.db.log_fill_event_atomic, trade_dict, position_dict)
                
                # ═══════════════════════════════════════════════════════════════
                # FORENSIC FIX #3: PERSIST EXIT DECISION NUCLEUS
                # QUÉ: Guarda el ballot en la tabla exit_decisions.
                # ═══════════════════════════════════════════════════════════════
                if is_close and _exit_ballot:
                    for v in _exit_ballot['exit_votes']:
                        self.io_executor.submit(
                            self.db.log_exit_decision,
                            trade_id=notif_trade_id,
                            symbol=event.symbol,
                            exit_reason=v['reason'],
                            proposing_strategy=v['vote'],
                            oracle_verdict="EXIT",
                            pnl_at_decision=current_pos['unrealized_pnl']
                        )
            except Exception as e:
                logger.error(f"⚠️ DB Atomic Fill Logging Error: {e}")
                
            # --- LEGACY NOTIFICATION (Phase 4 Backward-Compat) ---
            # FORENSIC-V21 FIX #3: Removed legacy duplicate notification.
            # send_trade_open/close above already provides ALL information.
            # Keeping this would cause triple notifications per trade.
        except Exception as e:
            logger.error(f"⚠️ Report Error: {e}")

    def log_to_csv(self, data):
        self.io_executor.submit(self._sync_log_to_csv, data)

    def _update_strategy_performance(self, strategy_id: str, pnl: float):
        """
        FORENSIC-V31: Helper to update strategy performance stats.
        
        PROFESSOR METHOD:
        - QUÉ: Actualiza contadores de wins/losses y PnL acumulado por estrategia.
        - POR QUÉ: El `pnl` que recibe ahora es NET PnL (después de fees), no bruto.
        - PARA QUÉ: Win Rate correcto en Telegram y Kelly Criterion preciso.
        - CÓMO: Un trade es WIN solo si net_pnl > 0 (ganó dinero REAL después de fees).
        - CUÁNDO: Cada vez que se cierra una posición en update_fill().
        - DÓNDE: portfolio.py → _update_strategy_performance()
        - QUIÉN: Llamado desde update_fill() con net_pnl_for_perf.
        
        CAMBIO CRÍTICO V31: Antes recibía gross_pnl (isolated_pnl antes de fees).
        Un trade con gross +$0.002 y fees $0.005 se contaba como WIN.
        Ahora recibe net_pnl = gross - fees → se cuenta correctamente como LOSS.
        """
        # Mutación 32: Reporte Hormonal (Dopamina / Cortisol)
        try:
            from core.synaptic_pruner import SynapticPruner
            SynapticPruner.get_instance().report_trade_result(strategy_id, pnl)
        except Exception as e:
            pass

        if strategy_id not in self.strategy_performance:
            self.strategy_performance[strategy_id] = {
                'trades': 0, 'wins': 0, 'losses': 0, 
                'pnl': 0.0, 'win_rate': 0.0,
                'total_win_pnl': 0.0, 'total_loss_pnl': 0.0,
                'current_streak': 0, 'max_streak': 0, 'losing_streak': 0
            }
            
        stats = self.strategy_performance[strategy_id]
        stats['trades'] += 1
        stats['pnl'] += pnl
        
        # FORENSIC-V31: pnl is now NET (after fees). A trade is only a WIN
        # if the trader actually made money after all costs.
        if pnl > 0:
            stats['wins'] += 1
            stats['total_win_pnl'] = stats['total_win_pnl'] + pnl
            stats['current_streak'] = stats['current_streak'] + 1
            stats['losing_streak'] = 0
            stats['max_streak'] = max(stats['max_streak'], stats['current_streak'])
        elif pnl < 0:
            stats['losses'] += 1
            stats['total_loss_pnl'] = stats['total_loss_pnl'] + abs(pnl)
            stats['current_streak'] = 0
            stats['losing_streak'] = stats['losing_streak'] + 1
        # Note: pnl == 0.0 exactly is neither win nor loss (breakeven)
            
        if stats['trades'] > 0:
            stats['win_rate'] = stats['wins'] / stats['trades']
        
        # ═══════════════════════════════════════════════════════════════
        # FORENSIC-V35: SESSION-LEVEL STATS UPDATE
        # QUÉ: Actualiza contadores de sesión en paralelo al all-time.
        # POR QUÉ: log_trade_report() y Telegram usan session stats como WR primario.
        # ═══════════════════════════════════════════════════════════════
        self._session_stats['total'] += 1
        self._session_stats['net_pnl'] += pnl
        if pnl > 0:
            self._session_stats['wins'] += 1
            # ⚡ FASE 8: ANTI-MARTINGALE — Track winning streaks
            self._win_streak += 1
            self._loss_streak = 0
            if self._win_streak > self._max_win_streak:
                self._max_win_streak = self._win_streak
        elif pnl < 0:
            self._session_stats['losses'] += 1
            # ⚡ FASE 8: ANTI-MARTINGALE — Track losing streaks
            self._loss_streak += 1
            self._win_streak = 0

    def get_statistics(self) -> Dict[str, Any]:
        """
        ═══════════════════════════════════════════════════════════════
        FORENSIC FIX #20: PHANTOM FEATURE RESURRECTION
        QUÉ: Retorna estadísticas agregadas del portafolio.
        POR QUÉ: RiskManager.get_win_rate() (L880) llama este método
          para calcular el Kelly Criterion dinámico. Sin este método,
          hasattr() siempre fallaba → Kelly usaba bootstrap_win_rate
          fijo (0.55), ignorando los datos REALES de trading.
        PARA QUÉ: Sizing adaptativo que EVOLUCIONA con el rendimiento real.
        CÓMO: Agrega wins/trades de TODAS las estrategias registradas.
        CUÁNDO: Llamado por RiskManager cada vez que necesita calcular
          el tamaño de una nueva posición.
        DÓNDE: core/portfolio.py → Portfolio.get_statistics()
        QUIÉN: Portfolio → RiskManager → Dynamic Kelly
        ═══════════════════════════════════════════════════════════════
        """
        total_trades = 0
        total_wins = 0
        total_pnl = 0.0

        for strat_id, stats in self.strategy_performance.items():
            total_trades += stats['trades']
            total_wins += stats['wins']
            total_pnl += stats['pnl']

        win_rate = (total_wins / total_trades) if total_trades > 0 else 0.0

        return {
            'total_trades': total_trades,
            'total_wins': total_wins,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
        }

    def get_strategy_metrics(self, strategy_id: str) -> Dict[str, float]:
        # PHASE 17: Meritocratic Scaling Metrics
        # QUÉ: Calcula el MeritFactor basado en el rendimiento histórico real.
        # POR QUÉ: Permite que el RiskManager premie a las estrategias ganadoras y castigue a las perdedoras.
        strat_data = self.strategy_performance.get(strategy_id, {})
        
        if not strat_data or strat_data['trades'] < 5:
            # Neutral Merit for new or unknown strategies
            return {
                'win_rate': 0.5,
                'total_pnl': 0.0,
                'total_trades': strat_data['trades'] if strat_data else 0,
                'merit_factor': 1.0 
            }
            
        wins = strat_data['wins']
        trades = strat_data['trades']
        win_rate = wins / trades
        
        # Profit Factor calculation
        total_win = strat_data['total_win_pnl']
        total_loss = abs(strat_data['total_loss_pnl'])
        
        profit_factor = (total_win / total_loss) if total_loss > 0 else 2.0 # Cap if no losses
        
        # MeritFactor: Weighted combination of WR and Profit Factor
        # Normalized around 1.0
        merit_factor = (win_rate / 0.5) * (profit_factor / 1.5)
        merit_factor = max(0.1, min(2.5, merit_factor)) # Clamp between 0.1x and 2.5x
        
        return {
            'win_rate': win_rate,
            'total_pnl': strat_data['pnl'],
            'total_trades': trades,
            'profit_factor': profit_factor,
            'merit_factor': merit_factor
        }

    def get_setup_performance(self, setup_type: str) -> dict:
        # Calcula métricas de rendimiento para un setup_type específico.
        # QUÉ: Win Rate y Expectancy.
        # POR QUÉ: Insumo crítico para el RiskManager (Meritocratic Sizing).
        relevant_trades = [t for t in self.trade_history if t['setup_type'] == setup_type]
        
        if not relevant_trades:
            return {'win_rate': 0.0, 'expectancy': 0.0, 'trades_count': 0}
            
        wins = sum(1 for t in relevant_trades if (t['pnl'] > 0 or t['pnl_pct'] > 0))
        total = len(relevant_trades)
        wr = wins / total
        
        # Expectancy = (Win% * AvgWin) - (Loss% * AvgLoss) - simplificado como PnL promedio
        pnls = [t['pnl_pct'] for t in relevant_trades]
        avg_pnl = sum(pnls) / total
        
        return {
            'win_rate': wr,
            'expectancy': avg_pnl * 100, # en puntos porcentuales
            'trades_count': total
        }
