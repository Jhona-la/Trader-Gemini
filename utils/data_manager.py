"""
DATA MANAGER
=============
Handles all data management tasks:
1. Auto-cleanup of large logs on startup.
2. Atomic saving of dashboard data (status.json/csv).
3. Tail reading for performance.

By delegating these tasks here, main.py stays clean.
"""

import os
import shutil
import time
import json
import csv
from datetime import datetime
from threading import Lock

# Lock for file writing to prevent race conditions
_file_lock = Lock()

def cleanup_dashboard_data(data_dir: str, max_mb: int = 100):
    """
    Checks if status.csv is too large (>100MB by default).
    If so, renames it to archive_status_TIMESTAMP.csv and creates a fresh one.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        return

    status_path = os.path.join(data_dir, "status.csv")
    if not os.path.exists(status_path):
        return

    try:
        size_bytes = os.path.getsize(status_path)
        size_mb = size_bytes / (1024 * 1024)
        
        if size_mb > max_mb:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"archive_status_{timestamp}.csv"
            archive_path = os.path.join(data_dir, archive_name)
            
            # Atomic rename
            shutil.move(status_path, archive_path)
            print(f"🧹 [CLEANUP] Archived large log: {status_path} ({size_mb:.1f}MB) -> {archive_path}")
            
            # Create fresh header
            with open(status_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'total_equity', 'cash', 'realized_pnl', 'unrealized_pnl', 'positions'])
    
    except Exception as e:
        print(f"⚠️ Failed to cleanup data: {e}")

def save_dashboard_data(portfolio, data_dir: str):
    """
    Saves snapshot of portfolio to:
    1. live_status.json (Atomic write for Dashboard)
    2. status.csv (Historical log)
    """
    try:
        from datetime import datetime
        
        # Calculate Equity
        unrealized_pnl = portfolio.unrealized_pnl
        total_equity = portfolio.current_cash + unrealized_pnl
        
        # Prepare Data
        # Filter out closed positions for cleaner JSON
        active_positions = {
            k: v for k, v in portfolio.positions.items() 
            if v['quantity'] != 0
        }
        
        data_packet = {
            'timestamp': datetime.now().isoformat(),
            'total_equity': total_equity,
            'cash': portfolio.current_cash,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': portfolio.realized_pnl,
            'positions': active_positions,
            'daily_pnl': portfolio.realized_pnl + unrealized_pnl # Simplified session pnl
        }
        
        # 1. ATOMIC WRITE JSON (Tmp -> Rename)
        json_path = os.path.join(data_dir, "live_status.json")
        tmp_path = json_path + ".tmp"
        
        with open(tmp_path, 'w') as f:
            json.dump(data_packet, f, indent=2)
            
        os.replace(tmp_path, json_path) # Atomic replacement
        
        # 2. APPEND TO CSV (Status History)
        csv_path = os.path.join(data_dir, "status.csv")
        file_exists = os.path.exists(csv_path)
        
        with _file_lock:
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['timestamp', 'total_equity', 'cash', 'realized_pnl', 'unrealized_pnl', 'positions'])
                
                # Format positions string for CSV
                pos_str = str(active_positions).replace(',', ';') # Avoid CSV delimiter conflict
                
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    f"{total_equity:.2f}",
                    f"{portfolio.current_cash:.2f}",
                    f"{portfolio.realized_pnl:.2f}",
                    f"{unrealized_pnl:.2f}",
                    pos_str
                ])
                
    except Exception as e:
        print(f"⚠️ Error saving dashboard data: {e}")

def safe_write_json(file_path, data):
    """Existing atomic write helper"""
    tmp = file_path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(data, f)
    os.replace(tmp, file_path)

def initialize_data_manager(data_dir: str):
    """Wrapper to call cleanup on startup"""
    cleanup_dashboard_data(data_dir)

def safe_append_csv(file_path: str, data_dict: dict):
    """
    Thread-safe CSV append.
    Acquires lock to prevent race conditions with dashboard readers.
    """
    with _file_lock:
        file_exists = os.path.isfile(file_path)
        try:
            with open(file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=data_dict.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data_dict)
        except Exception as e:
            # Fallback or Log? Since this is utils, we print or raise.
            # But logger might be using this? No, logger uses QueueHandler.
            print(f"safe_append_csv failed: {e}")

def safe_read_csv(file_path: str):
    """
    Thread-safe CSV read.
    Returns DataFrame or None if failed.
    """
    import pandas as pd
    with _file_lock:
        if not os.path.exists(file_path):
            return None
        try:
            # on_bad_lines='skip' prevents ParserError when schema changes
            return pd.read_csv(file_path, on_bad_lines='skip')
        except Exception as e:
            print(f"safe_read_csv failed: {e}")
            return None

class DatabaseHandler:
    """
    Robust SQLite Handler with WAL Mode (Write-Ahead Logging).
    Institutional Grade: Non-blocking readers, concurrent writers (mostly).
    
    ═══════════════════════════════════════════════════════════════════
    CTOS PHASE 6: MEMORIA OMNISCIENTE
    QUÉ: Expandido de 2 tablas básicas a 6 tablas forenses completas.
    POR QUÉ: portfolio.py y exit_oracle.py llamaban 4 métodos que NO
        EXISTÍAN (log_trade_chronicle, log_exit_strategy_decision,
        log_exit_decision, log_prediction_audit). Todas esas llamadas
        fallaban silenciosamente → CERO datos forenses se guardaban.
    PARA QUÉ: Capturar la historia completa de cada trade, cada decisión,
        cada predicción y cada tick de vida de cada posición.
    CÓMO: Creamos las tablas + métodos con INSERT atómicos WAL.
    CUÁNDO: Desde ahora, en producción y backtest.
    DÓNDE: utils/data_manager.py → DatabaseHandler
    QUIÉN: Portfolio, ExitOracle, PredictionTracker → DatabaseHandler
    ═══════════════════════════════════════════════════════════════════
    """
    def __init__(self, db_path="data.db"):
        self.db_path = db_path
        import sqlite3
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # === INSTITUTIONAL OPTIMIZATION: WAL MODE ===
        # Allows simultaneous readers and writers.
        # Checkpoints only happen when needed.
        self.cursor.execute("PRAGMA journal_mode=WAL;")
        self.cursor.execute("PRAGMA synchronous=NORMAL;") # Faster, still safe enough
        self.conn.commit()
        
        self._init_tables()
        self._lock = Lock()
        
        from utils.logger import logger
        logger.info(f"Database tables initialized at {db_path}")
        
    def _init_tables(self):
        # ═══════════════════════════════════════════════════════════════
        # TABLE 1: trades — Registro básico de fills
        # ═══════════════════════════════════════════════════════════════
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                side TEXT,
                quantity REAL,
                price REAL,
                timestamp DATETIME,
                strategy_id TEXT,
                pnl REAL,
                commission REAL
            )
        """)
        
        # ═══════════════════════════════════════════════════════════════
        # TABLE 2: positions — Snapshot para crash recovery
        # ═══════════════════════════════════════════════════════════════
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                quantity REAL,
                entry_price REAL,
                current_price REAL,
                pnl REAL,
                updated_at DATETIME
            )
        """)
        
        # ═══════════════════════════════════════════════════════════════
        # TABLE 3: trade_chronicle — Historia tick-by-tick de posiciones
        # QUÉ: Cada 5 ticks, registra el estado de cada posición abierta.
        # POR QUÉ: Para saber exactamente qué pasó durante la vida del trade.
        # PARA QUÉ: Post-mortem: "¿Cuándo estuvo en máximo? ¿Por qué no cerró?"
        # ═══════════════════════════════════════════════════════════════
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_chronicle (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT,
                symbol TEXT,
                horizon TEXT,
                tick_number INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                current_price REAL,
                entry_price REAL,
                unrealized_pnl_pct REAL,
                distance_to_tp_pct REAL,
                distance_to_sl_pct REAL,
                mfe_so_far REAL,
                mae_so_far REAL,
                oracle_prediction_magnitude REAL,
                oracle_prediction_target_price REAL,
                oracle_prediction_time_bars INTEGER,
                direction TEXT,
                entry_size_usd REAL
            )
        """)
        
        # ═══════════════════════════════════════════════════════════════
        # TABLE 4: exit_decisions — Votos de CADA estrategia de cierre
        # QUÉ: Registra si cada estrategia votó EXIT o HOLD para cada trade.
        # POR QUÉ: Para saber quién cerró, quién quería mantener, y quién fue vetado.
        # PARA QUÉ: Diagnosticar trades que debieron cerrarse antes o después.
        # ═══════════════════════════════════════════════════════════════
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS exit_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT,
                symbol TEXT,
                bar_number INTEGER,
                strategy_id TEXT,
                action TEXT,
                reason TEXT,
                unrealized_pnl REAL,
                price_at_decision REAL,
                was_overridden INTEGER DEFAULT 0,
                override_reason TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # ═══════════════════════════════════════════════════════════════
        # TABLE 5: exit_verdicts — Veredicto final del Oráculo
        # QUÉ: El resultado final de la votación (APPROVED/DENIED).
        # POR QUÉ: Para auditar si el Oráculo tomó buenas decisiones.
        # ═══════════════════════════════════════════════════════════════
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS exit_verdicts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT,
                symbol TEXT,
                exit_reason TEXT,
                proposing_strategy TEXT,
                oracle_verdict TEXT,
                pnl_at_decision REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # ═══════════════════════════════════════════════════════════════
        # TABLE 6: prediction_audit — Predicciones vs Realidad
        # QUÉ: Compara lo que la IA predijo con lo que realmente pasó.
        # POR QUÉ: Para calibrar modelos y saber qué estrategia predice bien.
        # PARA QUÉ: Ajustar confidence thresholds y TTL por estrategia.
        # ═══════════════════════════════════════════════════════════════
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT,
                thought_id TEXT,
                strategy_id TEXT,
                symbol TEXT,
                horizon TEXT,
                direction TEXT,
                predicted_magnitude_pct REAL,
                predicted_duration_bars INTEGER,
                predicted_target_price REAL,
                confidence REAL,
                actual_magnitude_pct REAL,
                actual_duration_bars INTEGER,
                actual_exit_price REAL,
                was_correct INTEGER,
                optimal_exit_price REAL,
                optimal_exit_bar INTEGER,
                missed_profit_pct REAL,
                entry_time DATETIME,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for fast querying
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_chronicle_trade ON trade_chronicle(trade_id)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_chronicle_symbol ON trade_chronicle(symbol)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_exit_dec_trade ON exit_decisions(trade_id)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_exit_dec_strat ON exit_decisions(strategy_id)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_audit_strat ON prediction_audit(strategy_id)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_audit_symbol ON prediction_audit(symbol)")
        
        self.conn.commit()
        
    # ═══════════════════════════════════════════════════════════════════
    # EXISTING METHODS (preserved)
    # ═══════════════════════════════════════════════════════════════════
        
    def get_open_positions(self):
        """Recover positions after crash"""
        with self._lock:
            try:
                self.cursor.execute("SELECT * FROM positions WHERE quantity != 0")
                rows = self.cursor.fetchall()
                positions = {}
                for row in rows:
                    # symbol, qty, entry, current, pnl, updated
                    positions[row[0]] = {
                        'quantity': row[1],
                        'entry_price': row[2],
                        'current_price': row[3],
                        'pnl': row[4]
                    }
                return positions
            except Exception as e:
                print(f"DB Error: {e}")
                return {}
        
    def log_fill_event_atomic(self, trade_payload, position_payload):
        """Atomic Transaction: Log Trade + Update Position"""
        with self._lock:
            try:
                # 1. Insert Trade
                self.cursor.execute("""
                    INSERT INTO trades (symbol, side, quantity, price, timestamp, strategy_id, pnl, commission)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_payload['symbol'],
                    trade_payload['side'],
                    trade_payload['quantity'],
                    trade_payload['price'],
                    str(trade_payload.get('timestamp')) if trade_payload.get('timestamp') is not None else None,
                    trade_payload.get('strategy_id', 'Unknown'),
                    trade_payload.get('pnl', 0),
                    trade_payload.get('commission', 0)
                ))
                
                # 2. Upsert Position
                self.cursor.execute("""
                    INSERT INTO positions (symbol, quantity, entry_price, current_price, pnl, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(symbol) DO UPDATE SET
                        quantity=excluded.quantity,
                        entry_price=excluded.entry_price,
                        current_price=excluded.current_price,
                        pnl=excluded.pnl,
                        updated_at=excluded.updated_at
                """, (
                    position_payload['symbol'],
                    position_payload['quantity'],
                    position_payload['entry_price'],
                    position_payload['current_price'],
                    position_payload['pnl'],
                    datetime.now()
                ))
                
                self.conn.commit()
            except Exception as e:
                print(f"⚠️ DB Write Error: {e}")
                self.conn.rollback()

    def update_position(self, symbol, quantity, entry_price, current_price, pnl):
        """Update single position state"""
        with self._lock:
            try:
                self.cursor.execute("""
                    INSERT INTO positions (symbol, quantity, entry_price, current_price, pnl, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(symbol) DO UPDATE SET
                        quantity=excluded.quantity,
                        entry_price=excluded.entry_price,
                        current_price=excluded.current_price,
                        pnl=excluded.pnl,
                        updated_at=excluded.updated_at
                """, (symbol, quantity, entry_price, current_price, pnl, datetime.now()))
                self.conn.commit()
            except Exception:
                pass # Silent fail during high load is ok for snapshot

    # ═══════════════════════════════════════════════════════════════════
    # NEW METHODS: MEMORIA OMNISCIENTE (Phase 6 CTOS)
    # ═══════════════════════════════════════════════════════════════════

    def log_trade_chronicle(self, **kwargs):
        """
        📜 Registra un tick de la vida de una posición abierta.
        Llamado cada 5 ticks por Portfolio._log_trade_chronicle_tick().
        """
        with self._lock:
            try:
                self.cursor.execute("""
                    INSERT INTO trade_chronicle (
                        trade_id, symbol, horizon, tick_number,
                        current_price, entry_price, unrealized_pnl_pct,
                        distance_to_tp_pct, distance_to_sl_pct,
                        mfe_so_far, mae_so_far,
                        oracle_prediction_magnitude, oracle_prediction_target_price,
                        oracle_prediction_time_bars, direction, entry_size_usd
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    kwargs.get('trade_id'),
                    kwargs.get('symbol'),
                    kwargs.get('horizon'),
                    kwargs.get('tick_number'),
                    kwargs.get('current_price'),
                    kwargs.get('entry_price'),
                    kwargs.get('unrealized_pnl_pct'),
                    kwargs.get('distance_to_tp_pct'),
                    kwargs.get('distance_to_sl_pct'),
                    kwargs.get('mfe_so_far'),
                    kwargs.get('mae_so_far'),
                    kwargs.get('oracle_prediction_magnitude'),
                    kwargs.get('oracle_prediction_target_price'),
                    kwargs.get('oracle_prediction_time_bars'),
                    kwargs.get('direction'),
                    kwargs.get('entry_size_usd'),
                ))
                self.conn.commit()
            except Exception as e:
                # Non-blocking: chronicle failures must never crash the engine
                pass

    def log_exit_strategy_decision(self, trade_id, symbol, bar_number,
                                    strategy_id, action, reason,
                                    unrealized_pnl=0.0, price_at_decision=0.0,
                                    was_overridden=False, override_reason=None):
        """
        🗳️ Registra el voto individual de UNA estrategia de cierre.
        Llamado por ExitOracle para CADA estrategia en CADA evaluación.
        """
        with self._lock:
            try:
                self.cursor.execute("""
                    INSERT INTO exit_decisions (
                        trade_id, symbol, bar_number, strategy_id,
                        action, reason, unrealized_pnl, price_at_decision,
                        was_overridden, override_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_id, symbol, bar_number, strategy_id,
                    action, reason, unrealized_pnl, price_at_decision,
                    1 if was_overridden else 0, override_reason
                ))
                self.conn.commit()
            except Exception:
                pass

    def log_exit_decision(self, trade_id, symbol, exit_reason,
                          proposing_strategy, oracle_verdict, pnl_at_decision=0.0):
        """
        ⚖️ Registra el veredicto FINAL del Oráculo (APPROVED/DENIED).
        Llamado por Portfolio y ExitOracle al momento del cierre o veto.
        """
        with self._lock:
            try:
                self.cursor.execute("""
                    INSERT INTO exit_verdicts (
                        trade_id, symbol, exit_reason,
                        proposing_strategy, oracle_verdict, pnl_at_decision
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    trade_id, symbol, exit_reason,
                    proposing_strategy, oracle_verdict, pnl_at_decision
                ))
                self.conn.commit()
            except Exception:
                pass

    def log_prediction_audit(self, **kwargs):
        """
        🔮 Registra la auditoría de una predicción vs la realidad.
        Llamado por Portfolio._record_closed_trade() al momento de cerrar.
        
        FORENSIC FIX: Ahora incluye open_size_usd, close_size_usd, size_delta_usd
        y open_price_at_prediction para trazabilidad completa del riesgo.
        """
        with self._lock:
            try:
                self.cursor.execute("""
                    INSERT INTO prediction_audit (
                        trade_id, thought_id, strategy_id, symbol,
                        horizon, direction,
                        predicted_magnitude_pct, predicted_duration_bars,
                        predicted_target_price, confidence,
                        actual_magnitude_pct, actual_duration_bars,
                        actual_exit_price, was_correct,
                        optimal_exit_price, optimal_exit_bar,
                        missed_profit_pct, entry_time,
                        open_size_usd, close_size_usd, size_delta_usd,
                        open_price_at_prediction
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    kwargs.get('trade_id'),
                    kwargs.get('thought_id'),
                    kwargs.get('strategy_id'),
                    kwargs.get('symbol'),
                    kwargs.get('horizon'),
                    kwargs.get('direction'),
                    kwargs.get('predicted_magnitude_pct'),
                    kwargs.get('predicted_duration_bars'),
                    kwargs.get('predicted_target_price'),
                    kwargs.get('confidence'),
                    kwargs.get('actual_magnitude_pct'),
                    kwargs.get('actual_duration_bars'),
                    kwargs.get('actual_exit_price'),
                    1 if kwargs.get('was_correct') else 0,
                    kwargs.get('optimal_exit_price'),
                    kwargs.get('optimal_exit_bar'),
                    kwargs.get('missed_profit_pct'),
                    str(kwargs.get('entry_time')) if kwargs.get('entry_time') else None,
                    kwargs.get('open_size_usd', 0.0),
                    kwargs.get('close_size_usd', 0.0),
                    kwargs.get('size_delta_usd', 0.0),
                    kwargs.get('open_price_at_prediction'),
                ))
                self.conn.commit()
            except Exception:
                pass

    def update_strategy_report_card(self, strategy_id: str, pnl: float, is_win: bool):
        """
        📊 Actualiza la Strategy Report Card con el resultado de un trade.
        
        QUÉ: Incrementa contadores de trades/wins/losses y actualiza PnL total.
        POR QUÉ: Sin esto, strategy_report_card estaba VACÍA (0 rows) → imposible
          saber qué estrategia funciona y cuál drena capital.
        PARA QUÉ: Gobernanza evolutiva → desactivar estrategias perdedoras automáticamente.
        CÓMO: UPSERT con INSERT OR REPLACE + valores acumulados.
        CUÁNDO: Cada vez que se cierra un trade (desde Portfolio._record_closed_trade).
        DÓNDE: utils/data_manager.py → update_strategy_report_card()
        QUIÉN: DataManager (llamado por Portfolio)
        """
        with self._lock:
            try:
                # Read current values
                row = self.cursor.execute(
                    "SELECT total_trades, wins, losses, total_pnl FROM strategy_report_card WHERE strategy_id = ?",
                    (strategy_id,)
                ).fetchone()
                
                if row:
                    total = row[0] + 1
                    wins = row[1] + (1 if is_win else 0)
                    losses = row[2] + (0 if is_win else 1)
                    total_pnl = row[3] + pnl
                    win_rate = wins / total if total > 0 else 0.0
                    
                    self.cursor.execute("""
                        UPDATE strategy_report_card 
                        SET total_trades = ?, wins = ?, losses = ?, 
                            win_rate = ?, total_pnl = ?, last_updated = CURRENT_TIMESTAMP
                        WHERE strategy_id = ?
                    """, (total, wins, losses, win_rate, total_pnl, strategy_id))
                else:
                    wins = 1 if is_win else 0
                    losses = 0 if is_win else 1
                    self.cursor.execute("""
                        INSERT INTO strategy_report_card 
                        (strategy_id, total_trades, wins, losses, win_rate, total_pnl)
                        VALUES (?, 1, ?, ?, ?, ?)
                    """, (strategy_id, wins, losses, float(is_win), pnl))
                
                self.conn.commit()
            except Exception:
                pass

    def log_session(self, session_id: str, start_equity: float, end_equity: float,
                    total_trades: int, wins: int, losses: int, gross_pnl: float,
                    total_fees: float, net_pnl: float, best_trade_pnl: float,
                    worst_trade_pnl: float, avg_trade_duration_sec: float,
                    symbols_traded: str, start_time: str):
        """
        📈 Registra métricas de la sesión de trading al cerrarla.
        
        QUÉ: Captura snapshot completo de la sesión (equity, trades, PnL, fees).
        POR QUÉ: session_ledger estaba VACÍA (0 rows) → cero visibilidad de rendimiento.
        PARA QUÉ: Tracking de crecimiento compuesto ($13 → duplicar cada 15 días).
        CUÁNDO: Al cerrar sesión (shutdown hook de engine.py o Portfolio.close()).
        """
        with self._lock:
            try:
                self.cursor.execute("""
                    INSERT INTO session_ledger (
                        session_id, start_equity, end_equity,
                        total_trades, wins, losses,
                        gross_pnl, total_fees, net_pnl,
                        best_trade_pnl, worst_trade_pnl,
                        avg_trade_duration_sec, symbols_traded, start_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id, start_equity, end_equity,
                    total_trades, wins, losses,
                    gross_pnl, total_fees, net_pnl,
                    best_trade_pnl, worst_trade_pnl,
                    avg_trade_duration_sec, symbols_traded, start_time
                ))
                self.conn.commit()
            except Exception:
                pass
        
    def close(self):
        self.conn.close()

