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
    Mock DatabaseHandler for compatibility.
    Real implementation pending Phase X.
    """
    def __init__(self, db_path="data.db"):
        pass
        
    def get_open_positions(self):
        return {}
        
    def close(self):
        pass
