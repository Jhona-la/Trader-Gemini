import sqlite3
import json
from datetime import datetime
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
                self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
                self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            return self.conn
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            return None

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
                    status TEXT DEFAULT 'OPEN'
                )
            ''')

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
                trade_dict.get('timestamp', datetime.now()),
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

    def update_position(self, symbol, quantity, entry_price, current_price=None, pnl=None):
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
                    INSERT INTO positions (symbol, quantity, entry_price, current_price, unrealized_pnl, timestamp, status)
                    VALUES (?, ?, ?, ?, ?, ?, 'OPEN')
                    ON CONFLICT(symbol) DO UPDATE SET
                        quantity=excluded.quantity,
                        entry_price=excluded.entry_price,
                        current_price=excluded.current_price,
                        unrealized_pnl=excluded.unrealized_pnl,
                        timestamp=excluded.timestamp
                ''', (
                    symbol, quantity, entry_price, current_price, pnl, datetime.now()
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
                positions[row['symbol']] = {
                    'quantity': row['quantity'],
                    'entry_price': row['entry_price'],
                    'current_price': row['current_price'],
                    'unrealized_pnl': row['unrealized_pnl']
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
            ''', (module, str(message), severity, datetime.now()))
            conn.commit()
        except sqlite3.Error as e:
            # Fallback to file logger if DB fails
            logger.error(f"Failed to log error to DB: {e}")

    def close(self):
        if self.conn:
            self.conn.close()
