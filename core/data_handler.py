"""
📊 DATA HANDLER
===============

PROFESSOR METHOD:
- QUÉ: Gestor centralizado de I/O de datos con validación estricta de tipos.
- POR QUÉ: Para garantizar integridad de datos y evitar corrupción JSON/CSV.
- PARA QUÉ: Sincronización robusta entre Trader (escritura) y Dashboard (lectura).
- CÓMO: 
    1. Type Guard: Fuerza todo float a 8 decimales.
    2. Atomic Write: Escritura en .tmp -> rename.
    3. Rate Limit: Control de frecuencia de escritura.
"""

import os
# import json (Removed Phase 3)
import csv
import time
import shutil
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any, List, Union
from datetime import datetime
import polars as pl

from utils.logger import logger
from utils.data_sync import atomic_write_json, touch_timestamp

class DataHandler:
    """
    Singleton para manejo seguro de datos del bot.
    Enforce schema strictness & atomic writes.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.last_write_time = {}
        self.min_write_interval = 1.0  # 1 segundo entre escrituras del mismo archivo
        self.max_health_log_lines = 1000 # Mantener solo las últimas 1000 comprobaciones
        self._initialized = True

    # =========================================================================
    # TYPE GUARD & PRECISION
    # =========================================================================
    
    def _enforce_precision(self, value, decimals=8):
        """Convierte valor a float con precisión específica usando Decimal."""
        if isinstance(value, (int, float, Decimal, str)):
            try:
                return round(float(value), decimals)
            except:
                return value
        return value

    def _sanitize_dict(self, data: Dict) -> Dict:
        """Recorre diccionario recursivamente y aplica precisión a números."""
        sanitized = {}
        for k, v in data.items():
            if isinstance(v, dict):
                sanitized[k] = self._sanitize_dict(v)
            elif isinstance(v, list):
                sanitized[k] = [self._sanitize_dict(i) if isinstance(i, dict) else self._enforce_precision(i) for i in v]
            elif isinstance(v, (int, float, Decimal)):
                sanitized[k] = self._enforce_precision(v)
            else:
                sanitized[k] = v
        return sanitized

    # =========================================================================
    # LIVE STATUS (JSON)
    # =========================================================================

    def save_live_status(self, filepath: str, status_data: Dict[str, Any]):
        """
        Guarda el estado del bot con validación de schema y escritura atómica.
        Estructura esperada:
        {
            "timestamp": "ISO...",
            "performance_metrics": {...},
            "positions": {...},
            ...
        }
        """
        # Rate Limit Check
        now = time.time()
        if now - self.last_write_time.get(filepath, 0) < self.min_write_interval:
            return  # Skip write to save I/O
            
        # 1. Type Guard (Sanitize)
        clean_data = self._sanitize_dict(status_data)
        
        # 2. Add Heartbeat Metadata
        from datetime import timezone
        clean_data['last_heartbeat'] = datetime.now(timezone.utc).isoformat()
        
        # 3. Atomic Write
        if atomic_write_json(clean_data, filepath):
            self.last_write_time[filepath] = now
            # Signal update to dashboard
            touch_timestamp(os.path.join(os.path.dirname(filepath), "last_update.txt"))
        else:
            logger.error(f"❌ Failed to save live status to {filepath}")

    def load_cached_status(self, filepath: str = None) -> Dict[str, Any]:
        """
        Loads the current bot status from JSON cache.
        Defaults to Config.DATA_DIR/live_status.json if filepath not provided.
        """
        if not filepath:
            from config import Config
            filepath = os.path.join(Config.DATA_DIR, "live_status.json")
            
        if not os.path.exists(filepath):
            return {}
            
        try:
            from utils.fast_json import FastJson
            return FastJson.load_from_file(filepath) or {}
        except Exception as e:
            logger.error(f"❌ Error loading cached status from {filepath}: {e}")
            return {}

    # =========================================================================
    # TRADES (CSV)
    # =========================================================================

    def log_trade(self, filepath: str, trade_data: Dict[str, Any]):
        """
        Registra un trade en CSV con esquema estricto.
        Schema: [timestamp, symbol, direction, entry_price, exit_price, quantity, pnl, fee, net_pnl, is_reverse]
        """
        schema = [
            'timestamp', 'symbol', 'direction', 
            'entry_price', 'exit_price', 'quantity', 
            'pnl', 'fee', 'net_pnl', 'is_reverse',
            'strategy_id' # Phase 6: Strategy Competition
        ]
        
        # Ensure directory
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare row
        row = {}
        for field in schema:
            val = trade_data.get(field)
            if field in ['entry_price', 'exit_price', 'quantity', 'pnl', 'fee', 'net_pnl']:
                row[field] = self._enforce_precision(val)
            elif field == 'is_reverse':
                row[field] = bool(val)
            elif field == 'strategy_id':
                row[field] = str(val if val else "MANUAL")
            else:
                row[field] = val
                
        # Write (Append mode is safest for CSV logs)
        file_exists = os.path.exists(filepath)
        
        try:
            with open(filepath, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=schema)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            logger.error(f"❌ Error logging trade to {filepath}: {e}")

    # =========================================================================
    # HEALTH LOGS (JSON LINES)
    # =========================================================================

    def log_health_check(self, data: Dict[str, Any]):
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        filepath = os.path.join(log_dir, "health_log.json")
        
        try:
            # Check if rotation needed
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                if len(lines) >= self.max_health_log_lines:
                    # Keep only the last N-1 lines and append new one
                    lines = lines[-(self.max_health_log_lines-1):]
                    with open(filepath, 'w') as f:
                        f.writelines(lines)
            
            with open(filepath, 'a') as f:
                from utils.fast_json import FastJson
                f.write(FastJson.dumps(data) + "\n")
        except Exception as e:
            logger.error(f"❌ Error writing health log: {e}")

    # =========================================================================
    # HISTORICAL STATUS (CSV)
    # =========================================================================

    def append_status_log(self, filepath: str, data: Dict[str, Any]):
        """
        Appends a historical status snapshot to status.csv.
        Forces header creation if file doesn't exist.
        Uses os.replace for atomic safety as requested for 24/7 stress test.
        """
        schema = ['timestamp', 'total_equity', 'available_balance', 'realized_pnl', 'unrealized_pnl']
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        tmp_path = filepath + ".tmp"
        file_exists = os.path.exists(filepath)
        
        try:
            # 1. If file exists, copy to tmp first (since we are appending)
            if file_exists:
                shutil.copy2(filepath, tmp_path)
            
            # 2. Append to tmp
            with open(tmp_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=schema)
                if not file_exists:
                    writer.writeheader()
                row = {k: self._enforce_precision(v) for k, v in data.items() if k in schema}
                writer.writerow(row)
            
            # 3. Atomic Replace
            os.replace(tmp_path, filepath)
                
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            logger.error(f"❌ Error appending status log to {filepath}: {e}")

    def load_trades_df(self, filepath: str) -> 'pl.DataFrame':
        """
        Lee trades.csv eficientemente y retorna DataFrame (Polars) con tipos correctos.
        """
        import polars as pl
        if not os.path.exists(filepath):
            return pl.DataFrame()
            
        try:
            df = pl.read_csv(filepath, ignore_errors=True)
            # Enforce types
            numeric_cols = ['entry_price', 'exit_price', 'quantity', 'pnl', 'fee', 'net_pnl']
            for col in numeric_cols:
                if col in df.columns:
                    df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False).fill_null(0.0))
            
            if 'is_reverse' in df.columns:
                df = df.with_columns(pl.col('is_reverse').cast(pl.Boolean, strict=False))
                
            return df
        except Exception as e:
            logger.error(f"❌ Error loading trades DF from {filepath}: {e}")
            return pl.DataFrame()

# Global instance getter
def get_data_handler():
    return DataHandler()
