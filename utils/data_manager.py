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

class DatabaseHandler:
    """
    Robust SQLite Handler with WAL Mode (Write-Ahead Logging).
    Institutional Grade: Non-blocking readers, concurrent writers (mostly).
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
        
    def _init_tables(self):
        # Trades Table
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
        
        # Positions Snapshot Table (for Crash Recovery)
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
        self.conn.commit()
        
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
                    trade_payload.get('timestamp'),
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
        
    def close(self):
        self.conn.close()
