import sqlite3
import json
import threading
from datetime import datetime, timezone
import os
from config import Config
from utils.logger import logger

class DatabaseHandler:
    def __init__(self, db_name="trader_gemini.db"):
        self.lock = threading.Lock()  # Lock initialized first
        if getattr(Config, 'IS_BACKTEST', False):
            self.db_path = "file:trader_gemini_mem?mode=memory&cache=shared"
        else:
            self.db_path = os.path.join(Config.DATA_DIR, db_name)
        self.conn = None
        self.create_tables()

    def get_connection(self):
        """
        Creates a database connection if one doesn't exist or is closed.
        """
        try:
            if self.conn is None:
                with self.lock:
                    if self.conn is None:
                        # FIXED: Python 3.12 Compatibility (Rule 5.1)
                        # Register explicit adapters for datetime
                        import sqlite3
                        sqlite3.register_adapter(datetime, lambda x: x.isoformat())
                        
                        # Check if converters already registered to avoid errors in some envs
                        try:
                            sqlite3.register_converter("timestamp", lambda x: datetime.fromisoformat(x.decode()))
                            sqlite3.register_converter("datetime", lambda x: datetime.fromisoformat(x.decode()))
                        except:
                            pass
                        
                        is_mem = "mode=memory" in self.db_path or self.db_path == ":memory:"
                        self.conn = sqlite3.connect(
                            self.db_path, 
                            check_same_thread=False,
                            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                            uri=is_mem
                        )
                        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
                        
                        # OPTIMIZATION: Enable WAL Mode for high concurrency (Rule 5.1)
                        if not is_mem:
                            self.conn.execute("PRAGMA journal_mode=WAL;")
                            self.conn.execute("PRAGMA synchronous=NORMAL;")
                            # Nano-latency DB enhancements (PHASE 52)
                            self.conn.execute("PRAGMA mmap_size=30000000000;") # 30GB memory map
                        else:
                            self.conn.execute("PRAGMA journal_mode=MEMORY;")
                            self.conn.execute("PRAGMA temp_store=MEMORY;")
                            
                        self.conn.execute("PRAGMA temp_store=MEMORY;")     # Temp tables in RAM
                        self.conn.execute("PRAGMA cache_size=-64000;")     # 64MB page cache
            return self.conn
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            if self.conn:
                try:
                    self.conn.close()
                except Exception:
                    pass
                self.conn = None
            return None

    def check_integrity(self):
        """
        Phase 43: Auto-Healing.
        Runs PRAGMA integrity_check. If failed, rotates DB.
        """
        conn = self.get_connection()
        if not conn:
            logger.error("🚨 DB CONNECTION FAILED: Healing database...")
            self.heal_database()
            return False
        
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check;")
            result = cursor.fetchone()[0]
            if result != "ok":
                logger.error(f"🚨 DB CORRUPTION DETECTED: {result}")
                self.heal_database()
                return False
            return True
        except Exception as e:
            logger.error(f"Integrity check failed: {e}")
            self.heal_database() # Phase 43: Heal on crash too
            return False

    def heal_database(self):
        """
        Rotates corrupted DB and starts fresh.
        """
        if self.conn:
            self.conn.close()
            self.conn = None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{self.db_path}.corrupt_{timestamp}"
        
        try:
            os.rename(self.db_path, backup_path)
            logger.warning(f"⚠️ ROTATED CORRUPT DB to {backup_path}")
            self.conn = self.get_connection() # Recast connection (creates new DB)
            self.create_tables()
            logger.info("✅ Database Auto-Healed successfully")
        except Exception as e:
            logger.critical(f"🔥 FATAL: Could not heal database: {e}")

    def create_tables(self):
        """
        Creates the necessary tables if they don't exist.
        """
        conn = self.get_connection()
        if not conn:
            return

        cursor = conn.cursor()
        
        try:
            # 1. TRADES TABLE
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    order_type TEXT,
                    strategy_id TEXT,
                    pnl REAL,
                    commission REAL
                )
            ''')

            # 2. SIGNALS TABLE
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    strength REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    strategy_id TEXT,
                    metadata TEXT
                )
            ''')

            # 3. POSITIONS TABLE (Snapshot of state)
            # We use this for crash recovery. 
            # When a position is closed, we update status to 'CLOSED' or delete it?
            # Better to keep history.
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    quantity REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL,
                    unrealized_pnl REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'OPEN',
                    sl_pct REAL,
                    tp_pct REAL,
                    horizon TEXT DEFAULT 'SCALPING',
                    strategy_id TEXT DEFAULT 'UNKNOWN'
                )
            ''')
            
            # 5. OMNISCIENT FORENSICS: Thoughts Table (Market Context at Decision)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS thoughts (
                    thought_id TEXT PRIMARY KEY,
                    trade_id TEXT,
                    symbol TEXT NOT NULL,
                    strategy_id TEXT,
                    horizon TEXT,
                    direction TEXT,
                    market_state TEXT,
                    metrics TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 6. OMNISCIENT FORENSICS: Exit Decisions Table (Why did we exit?)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS exit_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    exit_reason TEXT,
                    proposing_strategy TEXT,
                    oracle_verdict TEXT,
                    pnl_at_decision REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 7. OMNISCIENT FORENSICS: Trade Lifecycle (Tick-by-tick MFE/MAE)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_lifecycle (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    current_price REAL,
                    unrealized_pnl REAL,
                    mfe REAL,
                    mae REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 8. ASSET PROFILES: Liquidity and volatility tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS asset_profiles (
                    symbol TEXT PRIMARY KEY,
                    atr_14 REAL,
                    avg_volume_24h REAL,
                    volatility_pct REAL,
                    liquidity_score REAL,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 9. CTOS Phase 5: SYSTEM AWARENESS
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_awareness (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    active_strategies TEXT,
                    open_positions TEXT
                )
            ''')
            
            # ═══════════════════════════════════════════════════════════════
            # FORENSIC EXPANSION: TABLAS OMNISCIENTES
            # QUÉ: Tablas adicionales para trazabilidad forense total.
            # POR QUÉ: Necesitamos entender POR QUÉ se tomó cada decisión,
            #   qué activos tienen qué características, y el historial de sesiones.
            # PARA QUÉ: Diagnosticar trades perdedores y optimizar estrategias.
            # ═══════════════════════════════════════════════════════════════
            
            # 10. TRADE DECISION HISTORY: Por qué se abrió/rechazó cada señal
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_decision_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    horizon TEXT,
                    direction TEXT,
                    action TEXT NOT NULL,
                    reason TEXT,
                    strategy_id TEXT,
                    signal_strength REAL,
                    market_regime TEXT,
                    competing_signals TEXT,
                    equity_at_decision REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 11. ASSET INTELLIGENCE: Características por activo
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS asset_intelligence (
                    symbol TEXT PRIMARY KEY,
                    tick_size REAL,
                    lot_size REAL,
                    min_notional REAL,
                    avg_daily_range_pct REAL,
                    avg_daily_volume_usd REAL,
                    price_at_snapshot REAL,
                    market_cap_tier TEXT,
                    avg_spread_pct REAL,
                    best_scalp_timeframe TEXT,
                    best_swing_timeframe TEXT,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 12. SYSTEM HISTORY: Eventos del sistema
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    component TEXT,
                    details TEXT,
                    impact TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 13. SESSION LEDGER: Registro acumulado por sesión
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS session_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    start_equity REAL,
                    end_equity REAL,
                    total_trades INTEGER,
                    wins INTEGER,
                    losses INTEGER,
                    gross_pnl REAL,
                    total_fees REAL,
                    net_pnl REAL,
                    best_trade_pnl REAL,
                    worst_trade_pnl REAL,
                    avg_trade_duration_sec REAL,
                    symbols_traded TEXT,
                    start_time DATETIME,
                    end_time DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # ── INDICES FOR FORENSIC QUERIES ──
            index_definitions = [
                "CREATE INDEX IF NOT EXISTS idx_trade_decision_symbol ON trade_decision_history(symbol)",
                "CREATE INDEX IF NOT EXISTS idx_trade_decision_strategy ON trade_decision_history(strategy_id)",
                "CREATE INDEX IF NOT EXISTS idx_system_history_type ON system_history(event_type)",
                "CREATE INDEX IF NOT EXISTS idx_session_ledger_session ON session_ledger(session_id)",
                "CREATE INDEX IF NOT EXISTS idx_trade_chronicle_trade ON trade_chronicle(trade_id)",
                "CREATE INDEX IF NOT EXISTS idx_exit_log_trade ON exit_strategy_log(trade_id)",
            ]
            for idx_sql in index_definitions:
                try:
                    cursor.execute(idx_sql)
                except sqlite3.OperationalError:
                    pass  # Index already exists or table doesn't exist yet
            
            # --- Auto-Migration Logic for Existing Databases ---
            
            # Migrate positions
            cursor.execute("PRAGMA table_info(positions)")
            existing_pos_cols = [col[1] for col in cursor.fetchall()]
            pos_cols_to_add = {
                'sl_pct': 'REAL',
                'tp_pct': 'REAL',
                'horizon': 'TEXT DEFAULT "SCALPING"',
                'strategy_id': 'TEXT DEFAULT "UNKNOWN"',
                'trade_id': 'TEXT'
            }
            for col_name, col_type in pos_cols_to_add.items():
                if col_name not in existing_pos_cols:
                    try: cursor.execute(f"ALTER TABLE positions ADD COLUMN {col_name} {col_type}")
                    except sqlite3.OperationalError: pass
            
            # Migrate trades
            cursor.execute("PRAGMA table_info(trades)")
            existing_trade_cols = [col[1] for col in cursor.fetchall()]
            trade_cols_to_add = {
                'trade_id': 'TEXT',
                'thought_id': 'TEXT',
                'horizon': 'TEXT DEFAULT "SCALPING"'
            }
            for col_name, col_type in trade_cols_to_add.items():
                if col_name not in existing_trade_cols:
                    try: cursor.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}")
                    except sqlite3.OperationalError: pass
                    
            # Migrate signals
            cursor.execute("PRAGMA table_info(signals)")
            existing_sig_cols = [col[1] for col in cursor.fetchall()]
            sig_cols_to_add = {
                'trade_id': 'TEXT',
                'thought_id': 'TEXT',
                'horizon': 'TEXT DEFAULT "SCALPING"'
            }
            for col_name, col_type in sig_cols_to_add.items():
                if col_name not in existing_sig_cols:
                    try: cursor.execute(f"ALTER TABLE signals ADD COLUMN {col_name} {col_type}")
                    except sqlite3.OperationalError: pass
            
            # Migrate prediction_audit (FORENSIC FIX: position sizing columns)
            try:
                cursor.execute("PRAGMA table_info(prediction_audit)")
                existing_pa_cols = [col[1] for col in cursor.fetchall()]
                pa_cols_to_add = {
                    'open_size_usd': 'REAL DEFAULT 0.0',
                    'close_size_usd': 'REAL DEFAULT 0.0',
                    'size_delta_usd': 'REAL DEFAULT 0.0',
                    'predicted_close_size_usd': 'REAL',
                    'open_price_at_prediction': 'REAL'
                }
                for col_name, col_type in pa_cols_to_add.items():
                    if col_name not in existing_pa_cols:
                        try: cursor.execute(f"ALTER TABLE prediction_audit ADD COLUMN {col_name} {col_type}")
                        except sqlite3.OperationalError: pass
            except sqlite3.OperationalError:
                pass  # Table doesn't exist yet, will be created later

            # 9. ERRORS TABLE (Audit Trail)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    module TEXT,
                    message TEXT NOT NULL,
                    severity TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # ═══════════════════════════════════════════════════════════════
            # CTOS PHASE 2: OMNISCIENT MEMORY SYSTEM (Deep Forensic DB)
            # ═══════════════════════════════════════════════════════════════

            # 10. STRATEGY REPORT CARD
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_report_card (
                    strategy_id TEXT PRIMARY KEY,
                    total_trades INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0.0,
                    total_pnl REAL DEFAULT 0.0,
                    avg_rr_ratio REAL DEFAULT 0.0,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 11. PREDICTION LOG (ML vs Reality)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prediction_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    horizon TEXT,
                    predicted_direction TEXT,
                    confidence REAL,
                    actual_outcome TEXT,
                    pnl_realized REAL,
                    prediction_time DATETIME,
                    resolution_time DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 12. POSITION HEARTBEAT (Minute-by-minute tracking)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS position_heartbeat (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    unrealized_pnl REAL,
                    current_price REAL,
                    distance_to_tp REAL,
                    distance_to_sl REAL,
                    market_regime TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 13. BALANCE LEDGER (Equity curve tracking)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS balance_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_equity REAL,
                    available_margin REAL,
                    used_margin REAL,
                    open_positions_count INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 14. MARKET REGIME HISTORY
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_regime_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    regime_name TEXT,
                    btc_trend TEXT,
                    global_volatility REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # ═══════════════════════════════════════════════════════════════
            # CTOS PHASE 3: TRADE FORENSIC INTELLIGENCE TABLES
            # ═══════════════════════════════════════════════════════════════

            # 15. PREDICTION AUDIT: ¿Qué predijo cada estrategia vs realidad?
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prediction_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT NOT NULL,
                    thought_id TEXT,
                    strategy_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    horizon TEXT,
                    direction TEXT,
                    predicted_magnitude_pct REAL,
                    predicted_duration_bars INTEGER,
                    predicted_target_price REAL,
                    confidence REAL,
                    actual_magnitude_pct REAL,
                    actual_duration_bars INTEGER,
                    actual_exit_price REAL,
                    was_correct BOOLEAN,
                    optimal_exit_price REAL,
                    optimal_exit_bar INTEGER,
                    missed_profit_pct REAL,
                    entry_time DATETIME,
                    resolution_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    open_size_usd REAL DEFAULT 0.0,
                    close_size_usd REAL DEFAULT 0.0,
                    size_delta_usd REAL DEFAULT 0.0,
                    predicted_close_size_usd REAL,
                    open_price_at_prediction REAL
                )
            ''')

            # 16. EXIT STRATEGY LOG: Historia de decisiones de cierre
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS exit_strategy_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    bar_number INTEGER,
                    strategy_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reason TEXT,
                    unrealized_pnl REAL,
                    price_at_decision REAL,
                    was_overridden BOOLEAN DEFAULT 0,
                    override_reason TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 17. TRADE CHRONICLE: Historia minuto a minuto de posiciones
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_chronicle (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    horizon TEXT,
                    tick_number INTEGER,
                    current_price REAL,
                    entry_price REAL,
                    unrealized_pnl_pct REAL,
                    distance_to_tp_pct REAL,
                    distance_to_sl_pct REAL,
                    mfe_so_far REAL,
                    mae_so_far REAL,
                    market_regime TEXT,
                    volatility_1m REAL,
                    strategies_voting_exit TEXT,
                    strategies_voting_hold TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # ═══════════════════════════════════════════════════════════════
            # CTOS PHASE 4: OMNISCIENT INTELLIGENCE TABLES
            # ═══════════════════════════════════════════════════════════════

            # 18. STRATEGY AWARENESS: Registry of all strategies and their capabilities
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_awareness (
                    strategy_id TEXT PRIMARY KEY,
                    strategy_type TEXT NOT NULL,
                    capabilities TEXT,
                    supported_horizons TEXT,
                    supported_directions TEXT,
                    symbols TEXT,
                    last_signal_time DATETIME,
                    total_signals INTEGER DEFAULT 0,
                    is_active BOOLEAN DEFAULT 1,
                    registered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 19. SYSTEM SELF-AWARENESS: Snapshot of system capabilities and state
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_self_awareness (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_strategies INTEGER,
                    total_symbols INTEGER,
                    active_horizons TEXT,
                    active_modes TEXT,
                    system_state TEXT,
                    capabilities_json TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # --- Auto-Migration for PHASE 4 new columns ---

            # Migrate prediction_audit: add size tracking fields
            cursor.execute("PRAGMA table_info(prediction_audit)")
            existing_pa_cols = [col[1] for col in cursor.fetchall()]
            pa_new_cols = {
                'open_size_usd': 'REAL DEFAULT 0.0',
                'close_size_usd': 'REAL DEFAULT 0.0',
                'size_delta_usd': 'REAL DEFAULT 0.0',
                'predicted_close_size_usd': 'REAL',
                'open_price_at_prediction': 'REAL',
            }
            for col_name, col_type in pa_new_cols.items():
                if col_name not in existing_pa_cols:
                    try: cursor.execute(f"ALTER TABLE prediction_audit ADD COLUMN {col_name} {col_type}")
                    except sqlite3.OperationalError: pass

            # Migrate trade_chronicle: add oracle prediction and direction fields
            cursor.execute("PRAGMA table_info(trade_chronicle)")
            existing_tc_cols = [col[1] for col in cursor.fetchall()]
            tc_new_cols = {
                'oracle_prediction_magnitude': 'REAL',
                'oracle_prediction_target_price': 'REAL',
                'oracle_prediction_time_bars': 'INTEGER',
                'direction': 'TEXT',
                'entry_size_usd': 'REAL',
            }
            for col_name, col_type in tc_new_cols.items():
                if col_name not in existing_tc_cols:
                    try: cursor.execute(f"ALTER TABLE trade_chronicle ADD COLUMN {col_name} {col_type}")
                    except sqlite3.OperationalError: pass

            # Indexes for forensic queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_prediction_audit_trade ON prediction_audit(trade_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_exit_strategy_log_trade ON exit_strategy_log(trade_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_chronicle_trade ON trade_chronicle(trade_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_prediction_audit_strategy ON prediction_audit(strategy_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategy_awareness_type ON strategy_awareness(strategy_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_chronicle_symbol ON trade_chronicle(symbol)')

            conn.commit()
            logger.info(f"Database tables initialized at {self.db_path}")
            
        except sqlite3.Error as e:
            logger.error(f"Error creating tables: {e}")

    def log_trade(self, trade_dict):
        """
        Logs a executed trade.
        """
        if getattr(Config, 'IS_BACKTEST', False):
            return

        conn = self.get_connection()
        if not conn: return

        try:
            with self.lock:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO trades (symbol, side, quantity, price, timestamp, order_type, strategy_id, pnl, commission, trade_id, thought_id, horizon)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_dict.get('symbol'),
                    trade_dict.get('side'),
                    trade_dict.get('quantity'),
                    trade_dict.get('price'),
                    trade_dict.get('timestamp', datetime.now(timezone.utc)),
                    trade_dict.get('order_type', 'MARKET'),
                    trade_dict.get('strategy_id', 'UNKNOWN'),
                    trade_dict.get('pnl', 0.0),
                    trade_dict.get('commission', 0.0),
                    trade_dict.get('trade_id', None),
                    trade_dict.get('thought_id', None),
                    trade_dict.get('horizon', 'SCALPING')
                ))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error logging trade: {e}")

    def log_signal(self, signal_event):
        """
        Logs a generated signal.
        """
        if getattr(Config, 'IS_BACKTEST', False):
            return

        conn = self.get_connection()
        if not conn: return

        try:
            with self.lock:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO signals (symbol, signal_type, strength, timestamp, strategy_id, trade_id, thought_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal_event.symbol,
                    signal_event.signal_type.name if hasattr(signal_event.signal_type, 'name') else str(signal_event.signal_type),
                    signal_event.strength,
                    signal_event.timestamp if hasattr(signal_event, 'timestamp') else signal_event.datetime,
                    getattr(signal_event, 'strategy_id', 'UNKNOWN'),
                    getattr(signal_event, 'trade_id', None),
                    getattr(signal_event, 'metadata', {}).get('thought_id', None) if hasattr(signal_event, 'metadata') and signal_event.metadata else None
                ))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error logging signal: {e}")

    def log_thought(self, thought_id, trade_id, symbol, strategy_id, horizon, direction, market_state, metrics):
        """Logs a pre-decision thought process."""
        if getattr(Config, 'IS_BACKTEST', False):
            return

        conn = self.get_connection()
        if not conn: return
        try:
            with self.lock:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO thoughts (thought_id, trade_id, symbol, strategy_id, horizon, direction, market_state, metrics, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    thought_id, trade_id, symbol, strategy_id, horizon, direction,
                    json.dumps(market_state) if isinstance(market_state, dict) else str(market_state),
                    json.dumps(metrics) if isinstance(metrics, dict) else str(metrics),
                    datetime.now(timezone.utc)
                ))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error logging thought: {e}")

    def log_exit_decision(self, trade_id, symbol, exit_reason, proposing_strategy, oracle_verdict, pnl_at_decision):
        """Logs a centralized exit decision."""
        if getattr(Config, 'IS_BACKTEST', False):
            return

        conn = self.get_connection()
        if not conn: return
        try:
            with self.lock:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO exit_decisions (trade_id, symbol, exit_reason, proposing_strategy, oracle_verdict, pnl_at_decision, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_id, symbol, exit_reason, proposing_strategy, oracle_verdict, pnl_at_decision, datetime.now(timezone.utc)
                ))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error logging exit decision: {e}")

    def update_position(self, symbol, quantity, entry_price, current_price=None, pnl=None, sl_pct=None, tp_pct=None, horizon='SCALPING', strategy_id='UNKNOWN'):
        """
        Upserts a position state.
        If quantity is 0, marks as CLOSED or deletes.
        """
        if getattr(Config, 'IS_BACKTEST', False):
            return

        conn = self.get_connection()
        if not conn: return

        try:
            with self.lock:
                cursor = conn.cursor()
                
                if quantity == 0:
                    # Close position
                    cursor.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
                else:
                    # Upsert
                    cursor.execute('''
                        INSERT INTO positions (symbol, quantity, entry_price, current_price, unrealized_pnl, timestamp, status, sl_pct, tp_pct, horizon, strategy_id)
                        VALUES (?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?)
                        ON CONFLICT(symbol) DO UPDATE SET
                            quantity=excluded.quantity,
                            entry_price=excluded.entry_price,
                            current_price=excluded.current_price,
                            unrealized_pnl=excluded.unrealized_pnl,
                            timestamp=excluded.timestamp,
                            sl_pct=excluded.sl_pct,
                            tp_pct=excluded.tp_pct,
                            horizon=excluded.horizon,
                            strategy_id=excluded.strategy_id
                    ''', (
                        symbol, quantity, entry_price, current_price, pnl, datetime.now(timezone.utc), sl_pct, tp_pct, horizon, strategy_id
                    ))
                
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error updating position for {symbol}: {e}")

    def get_open_positions(self):
        """
        Retrieves all open positions for crash recovery.
        Returns a dictionary compatible with Portfolio.positions.
        """
        if getattr(Config, 'IS_BACKTEST', False):
            return {}

        conn = self.get_connection()
        if not conn: return {}

        try:
            with self.lock:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM positions WHERE status = 'OPEN'")
                rows = cursor.fetchall()
                
                positions = {}
                for row in rows:
                    keys = row.keys()
                    positions[row['symbol']] = {
                        'quantity': row['quantity'],
                        'entry_price': row['entry_price'],
                        'current_price': row['current_price'],
                        'unrealized_pnl': row['unrealized_pnl'],
                        'sl_pct': row['sl_pct'] if 'sl_pct' in keys else None,
                        'tp_pct': row['tp_pct'] if 'tp_pct' in keys else None,
                        'horizon': row['horizon'] if 'horizon' in keys else 'SCALPING',
                        'strategy_id': row['strategy_id'] if 'strategy_id' in keys else 'UNKNOWN'
                    }
                return positions
        except sqlite3.Error as e:
            logger.error(f"Error fetching open positions: {e}")
            return {}

    def log_error(self, module, message, severity="ERROR"):
        """
        Logs an error to the database.
        """
        if getattr(Config, 'IS_BACKTEST', False):
            return

        conn = self.get_connection()
        if not conn: return

        try:
            with self.lock:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO errors (module, message, severity, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (module, str(message), severity, datetime.now(timezone.utc)))
                conn.commit()
        except sqlite3.Error as e:
            # Fallback to file logger if DB fails
            logger.error(f"Failed to log error to DB: {e}")

    def log_strategy_performance(self, strategy_id: str, is_win: bool, pnl: float, rr_ratio: float):
        """Actualiza el scorecard de la estrategia."""
        if getattr(Config, 'IS_BACKTEST', False):
            return

        conn = self.get_connection()
        if not conn: return
        try:
            with self.lock:
                cursor = conn.cursor()
                win_val = 1 if is_win else 0
                loss_val = 0 if is_win else 1
                
                cursor.execute('''
                    INSERT INTO strategy_report_card (strategy_id, total_trades, wins, losses, win_rate, total_pnl, avg_rr_ratio)
                    VALUES (?, 1, ?, ?, ?, ?, ?)
                    ON CONFLICT(strategy_id) DO UPDATE SET
                        total_trades = total_trades + 1,
                        wins = wins + ?,
                        losses = losses + ?,
                        win_rate = CAST((wins + ?) AS REAL) / (total_trades + 1),
                        total_pnl = total_pnl + ?,
                        avg_rr_ratio = ((avg_rr_ratio * total_trades) + ?) / (total_trades + 1),
                        last_updated = ?
                ''', (
                    strategy_id, win_val, loss_val, 1.0 if is_win else 0.0, pnl, rr_ratio,
                    win_val, loss_val, win_val, pnl, rr_ratio, datetime.now(timezone.utc)
                ))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error logging strategy performance: {e}")

    def log_prediction(self, symbol: str, horizon: str, predicted_direction: str, confidence: float, actual_outcome: str, pnl: float, prediction_time):
        if getattr(Config, 'IS_BACKTEST', False):
            return

        conn = self.get_connection()
        if not conn: return
        try:
            with self.lock:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO prediction_log (symbol, horizon, predicted_direction, confidence, actual_outcome, pnl_realized, prediction_time, resolution_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (symbol, horizon, predicted_direction, confidence, actual_outcome, pnl, prediction_time, datetime.now(timezone.utc)))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error logging prediction: {e}")

    def log_position_heartbeat(self, trade_id: str, symbol: str, unrealized_pnl: float, current_price: float, dist_tp: float, dist_sl: float, regime: str):
        if getattr(Config, 'IS_BACKTEST', False):
            return

        conn = self.get_connection()
        if not conn: return
        try:
            with self.lock:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO position_heartbeat (trade_id, symbol, unrealized_pnl, current_price, distance_to_tp, distance_to_sl, market_regime, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (trade_id, symbol, unrealized_pnl, current_price, dist_tp, dist_sl, regime, datetime.now(timezone.utc)))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error logging position heartbeat: {e}")

    def log_balance_snapshot(self, equity: float, available_margin: float, used_margin: float, open_positions: int):
        if getattr(Config, 'IS_BACKTEST', False):
            return

        conn = self.get_connection()
        if not conn: return
        try:
            with self.lock:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO balance_ledger (total_equity, available_margin, used_margin, open_positions_count, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (equity, available_margin, used_margin, open_positions, datetime.now(timezone.utc)))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error logging balance snapshot: {e}")

    def log_market_regime(self, regime_name: str, btc_trend: str, global_volatility: float):
        if getattr(Config, 'IS_BACKTEST', False):
            return

        conn = self.get_connection()
        if not conn: return
        try:
            with self.lock:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO market_regime_history (regime_name, btc_trend, global_volatility, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (regime_name, btc_trend, global_volatility, datetime.now(timezone.utc)))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error logging market regime: {e}")

    # ═══════════════════════════════════════════════════════════════
    # CTOS PHASE 3: FORENSIC INTELLIGENCE LOGGING METHODS
    # ═══════════════════════════════════════════════════════════════

    def log_prediction_audit(self, trade_id: str, thought_id: str, strategy_id: str,
                             symbol: str, horizon: str, direction: str,
                             predicted_magnitude_pct: float, predicted_duration_bars: int,
                             predicted_target_price: float, confidence: float,
                             actual_magnitude_pct: float = None, actual_duration_bars: int = None,
                             actual_exit_price: float = None, was_correct: bool = None,
                             optimal_exit_price: float = None, optimal_exit_bar: int = None,
                             missed_profit_pct: float = None, entry_time=None):
        """
        🎯 CTOS-P3: Registra predicción de estrategia y (opcionalmente) su resultado real.
        
        QUÉ: Almacena qué predijo cada estrategia vs qué pasó realmente.
        POR QUÉ: Para saber qué estrategias predicen bien y cuáles mienten.
        PARA QUÉ: Feedback loop → rechazar estrategias con accuracy < 50%.
        CÓMO: Se llama al ABRIR (con predicción) y al CERRAR (con resultado real).
        """
        if getattr(Config, 'IS_BACKTEST', False):
            return

        conn = self.get_connection()
        if not conn: return
        try:
            with self.lock:
                cursor = conn.cursor()
                _entry_ts = entry_time
                if _entry_ts is not None:
                    if hasattr(_entry_ts, 'isoformat'):
                        _entry_ts = _entry_ts.isoformat()
                    else:
                        _entry_ts = str(_entry_ts)

                cursor.execute('''
                    INSERT INTO prediction_audit (
                        trade_id, thought_id, strategy_id, symbol, horizon, direction,
                        predicted_magnitude_pct, predicted_duration_bars, predicted_target_price,
                        confidence, actual_magnitude_pct, actual_duration_bars, actual_exit_price,
                        was_correct, optimal_exit_price, optimal_exit_bar, missed_profit_pct,
                        entry_time, resolution_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_id, thought_id, strategy_id, symbol, horizon, direction,
                    predicted_magnitude_pct, predicted_duration_bars, predicted_target_price,
                    confidence, actual_magnitude_pct, actual_duration_bars, actual_exit_price,
                    was_correct, optimal_exit_price, optimal_exit_bar, missed_profit_pct,
                    _entry_ts, datetime.now(timezone.utc)
                ))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error logging prediction audit: {e}")

    def update_prediction_audit_result(self, trade_id: str, strategy_id: str,
                                       actual_magnitude_pct: float, actual_duration_bars: int,
                                       actual_exit_price: float, was_correct: bool,
                                       optimal_exit_price: float, optimal_exit_bar: int,
                                       missed_profit_pct: float):
        """
        📊 CTOS-P3: Actualiza la predicción con el resultado real al cerrar el trade.
        """
        if getattr(Config, 'IS_BACKTEST', False):
            return

        conn = self.get_connection()
        if not conn: return
        try:
            with self.lock:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE prediction_audit SET
                        actual_magnitude_pct = ?,
                        actual_duration_bars = ?,
                        actual_exit_price = ?,
                        was_correct = ?,
                        optimal_exit_price = ?,
                        optimal_exit_bar = ?,
                        missed_profit_pct = ?,
                        resolution_time = ?
                    WHERE trade_id = ? AND strategy_id = ?
                ''', (
                    actual_magnitude_pct, actual_duration_bars, actual_exit_price,
                    was_correct, optimal_exit_price, optimal_exit_bar, missed_profit_pct,
                    datetime.now(timezone.utc), trade_id, strategy_id
                ))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error updating prediction audit: {e}")

    def log_exit_strategy_decision(self, trade_id: str, symbol: str, bar_number: int,
                                    strategy_id: str, action: str, reason: str,
                                    unrealized_pnl: float, price_at_decision: float,
                                    was_overridden: bool = False, override_reason: str = None):
        """
        🔄 CTOS-P3: Registra cada decisión de cierre/mantener de cada estrategia.
        
        QUÉ: Cada vez que una estrategia evalúa si cerrar una posición, se registra.
        POR QUÉ: Para saber por qué las estrategias de cierre no cerraron a tiempo.
        PARA QUÉ: Diagnosticar trades perdedores — qué estrategia debió cerrar y no lo hizo.
        """
        if getattr(Config, 'IS_BACKTEST', False):
            return

        conn = self.get_connection()
        if not conn: return
        try:
            with self.lock:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO exit_strategy_log (
                        trade_id, symbol, bar_number, strategy_id, action, reason,
                        unrealized_pnl, price_at_decision, was_overridden, override_reason, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_id, symbol, bar_number, strategy_id, action, reason,
                    unrealized_pnl, price_at_decision, was_overridden, override_reason,
                    datetime.now(timezone.utc)
                ))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error logging exit strategy decision: {e}")

    def log_trade_chronicle(self, trade_id: str, symbol: str, horizon: str,
                            tick_number: int, current_price: float, entry_price: float,
                            unrealized_pnl_pct: float, distance_to_tp_pct: float,
                            distance_to_sl_pct: float, mfe_so_far: float, mae_so_far: float,
                            market_regime: str = None, volatility_1m: float = None,
                            strategies_voting_exit: str = None, strategies_voting_hold: str = None,
                            oracle_prediction_magnitude: float = None,
                            oracle_prediction_target_price: float = None,
                            oracle_prediction_time_bars: int = None,
                            direction: str = None, entry_size_usd: float = None):
        """
        📜 CTOS-P4: Registra estado de posición abierta tick-a-tick.
        
        QUÉ: Snapshot del estado de una posición en cada intervalo.
        POR QUÉ: Para reconstruir la historia completa de cada trade.
        PARA QUÉ: Identificar el punto óptimo de cierre que nunca se tomó.
        CÓMO: Se llama desde update_market_price() cada N ticks.
        """
        if getattr(Config, 'IS_BACKTEST', False):
            return

        conn = self.get_connection()
        if not conn: return
        try:
            with self.lock:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO trade_chronicle (
                        trade_id, symbol, horizon, tick_number, current_price, entry_price,
                        unrealized_pnl_pct, distance_to_tp_pct, distance_to_sl_pct,
                        mfe_so_far, mae_so_far, market_regime, volatility_1m,
                        strategies_voting_exit, strategies_voting_hold,
                        oracle_prediction_magnitude, oracle_prediction_target_price,
                        oracle_prediction_time_bars, direction, entry_size_usd,
                        timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_id, symbol, horizon, tick_number, current_price, entry_price,
                    unrealized_pnl_pct, distance_to_tp_pct, distance_to_sl_pct,
                    mfe_so_far, mae_so_far, market_regime, volatility_1m,
                    strategies_voting_exit, strategies_voting_hold,
                    oracle_prediction_magnitude, oracle_prediction_target_price,
                    oracle_prediction_time_bars, direction, entry_size_usd,
                    datetime.now(timezone.utc)
                ))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error logging trade chronicle: {e}")

    def register_strategy(self, strategy_id: str, strategy_type: str,
                          capabilities: str = None, supported_horizons: str = None,
                          supported_directions: str = None, symbols: str = None):
        """
        🧠 CTOS-P4: Registra una estrategia en el registro de awareness.
        """
        if getattr(Config, 'IS_BACKTEST', False):
            return

        conn = self.get_connection()
        if not conn: return
        try:
            with self.lock:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO strategy_awareness (
                        strategy_id, strategy_type, capabilities, supported_horizons,
                        supported_directions, symbols, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(strategy_id) DO UPDATE SET
                        strategy_type=excluded.strategy_type,
                        capabilities=excluded.capabilities,
                        supported_horizons=excluded.supported_horizons,
                        supported_directions=excluded.supported_directions,
                        symbols=excluded.symbols,
                        last_updated=excluded.last_updated
                ''', (
                    strategy_id, strategy_type, capabilities, supported_horizons,
                    supported_directions, symbols, datetime.now(timezone.utc)
                ))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error registering strategy: {e}")

    def log_system_awareness(self, total_strategies: int, total_symbols: int,
                             active_horizons: str, active_modes: str,
                             system_state: str, capabilities_json: str = None):
        """
        🌐 CTOS-P4: Snapshot del estado del sistema completo.
        """
        if getattr(Config, 'IS_BACKTEST', False):
            return

        conn = self.get_connection()
        if not conn: return
        try:
            with self.lock:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO system_self_awareness (
                        total_strategies, total_symbols, active_horizons,
                        active_modes, system_state, capabilities_json, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    total_strategies, total_symbols, active_horizons,
                    active_modes, system_state, capabilities_json,
                    datetime.now(timezone.utc)
                ))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error logging system awareness: {e}")

    def log_fill_event_atomic(self, trade_dict, position_dict):
        """
        ATOMIC OPERATION (Rule 5.2):
        Logs trade and updates position in a SINGLE transaction.
        Ensures data consistency if bot crashes immediately after trade.
        """
        if getattr(Config, 'IS_BACKTEST', False):
            return

        conn = self.get_connection()
        if not conn: return

        try:
            with self.lock:
                cursor = conn.cursor()
                
                _ts = trade_dict.get('timestamp')
                if _ts is not None:
                    if hasattr(_ts, 'isoformat'):
                        _ts = _ts.isoformat()
                    else:
                        _ts = str(_ts)
                else:
                    _ts = datetime.now(timezone.utc).isoformat()
                
                # 1. Log Trade
                cursor.execute('''
                    INSERT INTO trades (symbol, side, quantity, price, timestamp, order_type, strategy_id, pnl, commission, trade_id, thought_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_dict.get('symbol'),
                    trade_dict.get('side'),
                    trade_dict.get('quantity'),
                    trade_dict.get('price'),
                    _ts,
                    trade_dict.get('order_type', 'MARKET'),
                    trade_dict.get('strategy_id', 'UNKNOWN'),
                    trade_dict.get('pnl', 0.0),
                    trade_dict.get('commission', 0.0),
                    trade_dict.get('trade_id'),
                    trade_dict.get('thought_id')
                ))
                
                # 2. Update Position
                symbol = position_dict['symbol']
                quantity = position_dict['quantity']
                
                if quantity == 0:
                    # Close position
                    cursor.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
                else:
                    # Upsert
                    cursor.execute('''
                        INSERT INTO positions (symbol, quantity, entry_price, current_price, unrealized_pnl, timestamp, status, sl_pct, tp_pct, horizon, strategy_id)
                        VALUES (?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?)
                        ON CONFLICT(symbol) DO UPDATE SET
                            quantity=excluded.quantity,
                            entry_price=excluded.entry_price,
                            current_price=excluded.current_price,
                            unrealized_pnl=excluded.unrealized_pnl,
                            timestamp=excluded.timestamp,
                            sl_pct=excluded.sl_pct,
                            tp_pct=excluded.tp_pct,
                            horizon=excluded.horizon,
                            strategy_id=excluded.strategy_id
                    ''', (
                        symbol, quantity, position_dict['entry_price'], 
                        position_dict['current_price'], position_dict.get('pnl', 0.0), 
                        datetime.now(timezone.utc).isoformat(),
                        position_dict.get('sl_pct'),
                        position_dict.get('tp_pct'),
                        position_dict.get('horizon', 'SCALPING'),
                        position_dict.get('strategy_id', 'UNKNOWN')
                    ))
                
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"⚠️ FATAL: Atomic DB Update failed: {e}")
            conn.rollback()

    def log_system_awareness_snapshot(self, active_strategies: dict, open_positions: dict):
        if getattr(Config, 'IS_BACKTEST', False):
            return

        conn = self.get_connection()
        if not conn: return
        try:
            with self.lock:
                cursor = conn.cursor()
                strategies_str = json.dumps(active_strategies)
                positions_str = json.dumps(open_positions)
                
                cursor.execute('''
                    INSERT INTO system_awareness (active_strategies, open_positions)
                    VALUES (?, ?)
                ''', (strategies_str, positions_str))
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging system awareness snapshot: {e}")

    def prune_historical_data(self, days_to_keep=7):
        if getattr(Config, 'IS_BACKTEST', False):
            return

        conn = self.get_connection()
        if not conn: return
        try:
            with self.lock:
                cursor = conn.cursor()
                logger.info(f"🧹 [DATABASE] Iniciando pruning de datos anteriores a {days_to_keep} días...")
                
                high_frequency_tables = [
                    'signals', 'trade_logs', 'thoughts', 'predictions', 
                    'trade_chronicle', 'exit_strategy_log', 'system_errors',
                    'market_regime_history'
                ]
                
                rows_deleted = 0
                for table in high_frequency_tables:
                    # Check if table exists
                    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
                    if cursor.fetchone():
                        cursor.execute(f'''
                            DELETE FROM {table} 
                            WHERE timestamp < datetime('now', '-{days_to_keep} days')
                        ''')
                        rows_deleted += cursor.rowcount
                        
                # Para prediction_audit, usamos resolution_time
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='prediction_audit'")
                if cursor.fetchone():
                    cursor.execute(f'''
                        DELETE FROM prediction_audit 
                        WHERE resolution_time < datetime('now', '-{days_to_keep} days')
                    ''')
                    rows_deleted += cursor.rowcount

                conn.commit()
                logger.info(f"✨ [DATABASE] Pruning completado. {rows_deleted} registros liberados. Ejecutando VACUUM...")
                
                # Ejecutar vacuum fuera de la transacción si SQLite lo permite
                try:
                    cursor.execute('VACUUM')
                except Exception as ve:
                    logger.debug(f"VACUUM skipped: {ve}")
        except Exception as e:
            logger.error(f"❌ [DATABASE] Error durante pruning: {e}")

    def close(self):
        """
        Closes the SQLite database connection and releases file locks.
        """
        with self.lock:
            if self.conn:
                try:
                    self.conn.close()
                    logger.info("🔌 SQLite connection successfully closed.")
                except Exception as e:
                    logger.error(f"Error closing SQLite connection: {e}")
                self.conn = None
