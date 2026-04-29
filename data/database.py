import sqlite3
import json
from datetime import datetime, timezone
import os
from config import Config
from utils.logger import logger

class DatabaseHandler:
    def __init__(self, db_name="trader_gemini.db"):
        self.db_path = os.path.join(Config.DATA_DIR, db_name)
        self.conn = None
        self.create_tables()

    def get_connection(self):
        """
        Creates a database connection if one doesn't exist or is closed.
        """
        try:
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
                
                self.conn = sqlite3.connect(
                    self.db_path, 
                    check_same_thread=False,
                    detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
                )
                self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
                
                # OPTIMIZATION: Enable WAL Mode for high concurrency (Rule 5.1)
                self.conn.execute("PRAGMA journal_mode=WAL;")
                self.conn.execute("PRAGMA synchronous=NORMAL;")
            return self.conn
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            return None

    def check_integrity(self):
        """
        Phase 43: Auto-Healing.
        Runs PRAGMA integrity_check. If failed, rotates DB.
        """
        conn = self.get_connection()
        if not conn: return False
        
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
            
            # --- Auto-Migration Logic for Existing Databases ---
            cursor.execute("PRAGMA table_info(positions)")
            existing_columns = [col[1] for col in cursor.fetchall()]
            
            columns_to_add = {
                'sl_pct': 'REAL',
                'tp_pct': 'REAL',
                'horizon': 'TEXT DEFAULT "SCALPING"',
                'strategy_id': 'TEXT DEFAULT "UNKNOWN"'
            }
            
            for col_name, col_type in columns_to_add.items():
                if col_name not in existing_columns:
                    try:
                        cursor.execute(f"ALTER TABLE positions ADD COLUMN {col_name} {col_type}")
                    except sqlite3.OperationalError as e:
                        # Ignorar si ya se está añadiendo por otro hilo concurrente
                        pass

            # 4. ERRORS TABLE (Audit Trail)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    module TEXT,
                    message TEXT NOT NULL,
                    severity TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            logger.info(f"Database tables initialized at {self.db_path}")
            
        except sqlite3.Error as e:
            logger.error(f"Error creating tables: {e}")

    def log_trade(self, trade_dict):
        """
        Logs a executed trade.
        """
        conn = self.get_connection()
        if not conn: return

        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO trades (symbol, side, quantity, price, timestamp, order_type, strategy_id, pnl, commission)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_dict.get('symbol'),
                trade_dict.get('side'),
                trade_dict.get('quantity'),
                trade_dict.get('price'),
                trade_dict.get('timestamp', datetime.now(timezone.utc)),
                trade_dict.get('order_type', 'MARKET'),
                trade_dict.get('strategy_id', 'UNKNOWN'),
                trade_dict.get('pnl', 0.0),
                trade_dict.get('commission', 0.0)
            ))
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error logging trade: {e}")

    def log_signal(self, signal_event):
        """
        Logs a generated signal.
        """
        conn = self.get_connection()
        if not conn: return

        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO signals (symbol, signal_type, strength, timestamp, strategy_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                signal_event.symbol,
                signal_event.signal_type,
                signal_event.strength,
                signal_event.timestamp,
                getattr(signal_event, 'strategy_id', 'UNKNOWN')
            ))
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error logging signal: {e}")

    def update_position(self, symbol, quantity, entry_price, current_price=None, pnl=None, sl_pct=None, tp_pct=None, horizon='SCALPING', strategy_id='UNKNOWN'):
        """
        Upserts a position state.
        If quantity is 0, marks as CLOSED or deletes.
        """
        conn = self.get_connection()
        if not conn: return

        try:
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
        conn = self.get_connection()
        if not conn: return {}

        try:
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
        conn = self.get_connection()
        if not conn: return

        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO errors (module, message, severity, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (module, str(message), severity, datetime.now(timezone.utc)))
            conn.commit()
        except sqlite3.Error as e:
            # Fallback to file logger if DB fails
            logger.error(f"Failed to log error to DB: {e}")

    def log_fill_event_atomic(self, trade_dict, position_dict):
        """
        ATOMIC OPERATION (Rule 5.2):
        Logs trade and updates position in a SINGLE transaction.
        Ensures data consistency if bot crashes immediately after trade.
        """
        conn = self.get_connection()
        if not conn: return

        try:
            cursor = conn.cursor()
            
            # 1. Log Trade
            cursor.execute('''
                INSERT INTO trades (symbol, side, quantity, price, timestamp, order_type, strategy_id, pnl, commission)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_dict.get('symbol'),
                trade_dict.get('side'),
                trade_dict.get('quantity'),
                trade_dict.get('price'),
                trade_dict.get('timestamp', datetime.now(timezone.utc)),
                trade_dict.get('order_type', 'MARKET'),
                trade_dict.get('strategy_id', 'UNKNOWN'),
                trade_dict.get('pnl', 0.0),
                trade_dict.get('commission', 0.0)
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
                    datetime.now(timezone.utc),
                    position_dict.get('sl_pct'),
                    position_dict.get('tp_pct'),
                    position_dict.get('horizon', 'SCALPING'),
                    position_dict.get('strategy_id', 'UNKNOWN')
                ))
            
            conn.commit()
            # logger.info(f"✅ Atomic DB Update: {symbol} Trade Logged & Position Updated")
            
        except sqlite3.Error as e:
            logger.error(f"⚠️ FATAL: Atomic DB Update failed: {e}")
            conn.rollback()

    def close(self):
        if self.conn:
            self.conn.close()
