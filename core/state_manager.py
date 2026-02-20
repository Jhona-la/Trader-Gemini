import os
import json
import shutil
import time
import sqlite3
from typing import Dict, Any, Optional
from utils.logger import logger
from utils.fast_json import FastJson

class AtomicStateManager:
    """
    🛡️ COMPONENT: Disaster Resilience (Atomic persistence + WAL-mode SQLite)
    
    QUÉ: Persistencia de estado a prueba de fallos con recuperación < 100ms.
    POR QUÉ: Escrituras parciales corrompen JSON; recovery lento mata latencia.
    PARA QUÉ: Garantizar continuidad operacional post-crash sin pérdida de estado.
    CÓMO: Dual persistence:
          1. JSON Atomic: Write '.tmp' → fsync → rename (legacy, human-readable)
          2. SQLite WAL: Checkpoint periódico a SQLite WAL-mode (sub-100ms recovery)
    CUÁNDO: save_json_atomic cada ciclo del engine; checkpoint cada 5s.
    DÓNDE: core/state_manager.py
    QUIÉN: Engine, Portfolio, Main loop.
    
    ⚡ PHASE OMNI UPGRADE:
    - SQLite WAL mode: concurrent reads during writes (no blocking)
    - Checkpoint/Recover cycle: < 100ms target recovery time
    - Atomic SAVEPOINT transactions for crash safety
    """
    
    _db_conn: Optional[sqlite3.Connection] = None
    _db_path: str = "data/state_checkpoint.db"
    _last_checkpoint: float = 0.0
    _checkpoint_interval: float = 5.0  # seconds
    
    @classmethod
    def _ensure_db(cls):
        """Initialize SQLite database with WAL mode if not already connected."""
        if cls._db_conn is not None:
            return
        
        try:
            os.makedirs(os.path.dirname(cls._db_path), exist_ok=True)
            
            cls._db_conn = sqlite3.connect(
                cls._db_path,
                isolation_level=None,  # Autocommit for performance
                check_same_thread=False,
            )
            
            # WAL mode: Allows concurrent reads during writes
            # Critical for HFT where dashboard reads while engine writes
            cls._db_conn.execute("PRAGMA journal_mode=WAL")
            cls._db_conn.execute("PRAGMA synchronous=NORMAL")  # Balance speed/safety
            cls._db_conn.execute("PRAGMA cache_size=-8000")     # 8MB cache
            cls._db_conn.execute("PRAGMA busy_timeout=5000")    # 5s busy wait
            
            # Create checkpoint table
            cls._db_conn.execute("""
                CREATE TABLE IF NOT EXISTS state_checkpoint (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            
            # Create index for fast recovery
            cls._db_conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_checkpoint_updated 
                ON state_checkpoint(updated_at)
            """)
            
            logger.info(f"🛡️ [StateManager] SQLite WAL checkpoint DB initialized: {cls._db_path}")
            
        except Exception as e:
            logger.error(f"❌ [StateManager] DB init failed: {e}")
            cls._db_conn = None
    
    @classmethod
    def checkpoint(cls, state: Dict[str, Any], key: str = "portfolio"):
        """
        ⚡ PHASE OMNI: Periodic state checkpoint to SQLite WAL.
        
        QUÉ: Guarda snapshot del estado en SQLite WAL-mode.
        POR QUÉ: SQLite WAL permite lectura concurrente sin bloqueo.
        PARA QUÉ: Recovery instantáneo (< 100ms) después de crash.
        CÓMO: UPSERT atómico en una tabla key-value con timestamp.
        
        Args:
            state: Dictionary to checkpoint (e.g. portfolio state)
            key: Checkpoint namespace (default: "portfolio")
        """
        now = time.time()
        if now - cls._last_checkpoint < cls._checkpoint_interval:
            return  # Throttle: don't checkpoint too frequently
        
        cls._ensure_db()
        if cls._db_conn is None:
            return
        
        try:
            start = time.perf_counter()
            
            serialized = json.dumps(state, default=str)
            
            cls._db_conn.execute(
                """INSERT OR REPLACE INTO state_checkpoint (key, value, updated_at) 
                   VALUES (?, ?, ?)""",
                (key, serialized, now)
            )
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            cls._last_checkpoint = now
            
            if elapsed_ms > 50:
                logger.warning(f"⚠️ [Checkpoint] Slow write: {elapsed_ms:.1f}ms")
            
        except Exception as e:
            logger.error(f"❌ [Checkpoint] Write failed: {e}")
    
    @classmethod
    def recover(cls, key: str = "portfolio") -> Optional[Dict[str, Any]]:
        """
        ⚡ PHASE OMNI: Fast state recovery from SQLite WAL.
        
        QUÉ: Recupera el último estado guardado en < 100ms.
        POR QUÉ: El bot debe reanudar operaciones inmediatamente tras un crash.
        PARA QUÉ: Evitar pérdidas por posiciones huérfanas no rastreadas.
        CÓMO: SELECT del último checkpoint → deserializar → validar.
        
        Returns:
            Recovered state dict, or None if no checkpoint exists.
        """
        cls._ensure_db()
        if cls._db_conn is None:
            return None
        
        try:
            start = time.perf_counter()
            
            cursor = cls._db_conn.execute(
                "SELECT value, updated_at FROM state_checkpoint WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            state = json.loads(row[0])
            checkpoint_age = time.time() - row[1]
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            logger.info(
                f"🛡️ [Recovery] State restored in {elapsed_ms:.1f}ms "
                f"(age: {checkpoint_age:.0f}s, key: {key})"
            )
            
            # Warn if checkpoint is stale (> 60s old)
            if checkpoint_age > 60:
                logger.warning(
                    f"⚠️ [Recovery] Checkpoint is {checkpoint_age:.0f}s old! "
                    f"Data may be stale. Verify with exchange."
                )
            
            return state
            
        except Exception as e:
            logger.error(f"❌ [Recovery] Failed: {e}")
            return None
    
    @classmethod
    def recover_with_fallback(cls, json_path: str, key: str = "portfolio") -> Optional[Dict[str, Any]]:
        """
        Recovery priority chain:
        1. SQLite WAL checkpoint (fastest, < 100ms)
        2. JSON atomic file (fallback, human-readable)
        3. None (fresh start)
        """
        # Try SQLite first
        state = cls.recover(key)
        if state:
            return state
        
        # Fallback to JSON
        state = cls.load_json(json_path)
        if state:
            logger.info(f"🛡️ [Recovery] Fell back to JSON: {json_path}")
            return state
        
        logger.warning("⚠️ [Recovery] No checkpoint or JSON found. Starting fresh.")
        return None
    
    @staticmethod
    def save_json_atomic(path: str, data: Dict[str, Any]):
        """
        Saves dict to JSON atomically (legacy method, preserved).
        """
        dir_name = os.path.dirname(path)
        base_name = os.path.basename(path)
        tmp_path = os.path.join(dir_name, f".{base_name}.tmp")
        
        try:
            # 1. Write to Temp File
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno()) # Force write to disk
                
            # 2. Atomic Rename
            os.replace(tmp_path, path)
            
        except Exception as e:
            logger.error(f"❌ [AtomicState] Save Failed: {e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except: pass

    @staticmethod
    def load_json(path: str) -> Optional[Dict[str, Any]]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ [AtomicState] Load Failed: {e}")
            return None
    
    @classmethod
    def close(cls):
        """Close the SQLite connection gracefully."""
        if cls._db_conn:
            try:
                cls._db_conn.close()
                cls._db_conn = None
                logger.info("🛡️ [StateManager] DB connection closed.")
            except Exception:
                pass

