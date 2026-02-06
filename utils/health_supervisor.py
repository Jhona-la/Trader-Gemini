import threading
import time
try:
    import ujson as json
except ImportError:
    import json
import os
from datetime import datetime, timezone
from utils.logger import logger
from core.data_handler import get_data_handler
from core.api_manager import get_api_manager

class HealthSupervisor(threading.Thread):
    """
    🕵️ CI-HMA Agent: Continuous Integrity & Health Monitoring Agent.
    
    Performs a 'Triple-Check' every 60s:
    1. State A: Real Exchange Balance (API)
    2. State B: Persisted Bot State (File)
    3. State C: Dashboard View (UI File)
    
    Alerts on discrepancies > 0.05%.
    """
    
    def __init__(self, check_interval=60):
        super().__init__()
        self.interval = check_interval
        self.daemon = True # Daemon thread dies with main process
        self.running = False
        self.dh = get_data_handler()
        self.api = get_api_manager()
        self.last_check = None
        
        # Paths
        self.status_file = "dashboard/data/futures/live_status.json" # Default
        self.integrity_file = "dashboard/data/futures/integrity.json"
        
    def run(self):
        logger.info("🩺 [HEALTH] Supervisor Agent STARTED.")
        self.running = True
        
        while self.running:
            try:
                self._perform_triple_check()
            except Exception as e:
                logger.error(f"🩺 [HEALTH] Check Failed: {e}")
            
            # Sleep in chunks to allow quick stop
            for _ in range(self.interval):
                if not self.running: break
                time.sleep(1)
                
    def stop(self):
        self.running = False
        
    def _perform_triple_check(self):
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # 1. State A: API (Real)
        # Force a fresh fetch if possible, or use cached if recent
        # We want TRUTH, so we try specific call
        api_balance = 0.0
        try:
            # We assume futures mode for now as primary
            acct = self.api.get_account_balance(is_prod=True) # Check PROD by default for safety
            if not acct:
                # Try testnet/demo if prod fails or config says so?
                # For now, check what API manager has
                acct = self.api.get_account_balance(is_prod=False)
            
            if acct:
                api_balance = float(acct.get('total_equity', 0))
        except:
            pass
            
        # 2. State B: File (Persisted)
        file_balance = 0.0
        file_ts = 0
        try:
            if os.path.exists(self.status_file):
                with open(self.status_file, 'r') as f:
                    data = json.load(f)
                    file_balance = float(data.get('total_equity', 0))
                    file_ts = os.path.getmtime(self.status_file)
        except:
            pass
            
        # 3. State C: UI (Dashboard Report)
        ui_balance = 0.0
        ui_ts = 0
        try:
            if os.path.exists(self.integrity_file):
                with open(self.integrity_file, 'r') as f:
                    data = json.load(f)
                    ui_balance = float(data.get('displayed_equity', 0))
                    ui_ts = data.get('timestamp_epoch', 0)
        except:
            pass
            
        # --- ANALYSIS ---
        sync_status = "OK"
        notes = []
        
        # Check A vs B (API vs Bot)
        if api_balance > 0 and file_balance > 0:
            diff_ab = abs(api_balance - file_balance)
            pct_ab = (diff_ab / api_balance) * 100
            if pct_ab > 0.05: # 0.05% tolerance
                sync_status = "CRITICAL_DESYNC_API"
                notes.append(f"API(${api_balance}) vs FILE(${file_balance}) Diff: {pct_ab:.4f}%")
                logger.error(f"🚨 [HEALTH] CRITICAL DATA DESYNC! {notes[-1]}")
                
        # Check B vs C (Bot vs UI)
        if file_balance > 0 and ui_balance > 0:
            diff_bc = abs(file_balance - ui_balance)
            pct_bc = (diff_bc / file_balance) * 100
            if pct_bc > 0.05:
                 if sync_status == "OK": sync_status = "UI_LAG_WARNING"
                 notes.append(f"FILE(${file_balance}) vs UI(${ui_balance}) Diff: {pct_bc:.4f}%")
        
        # Check Latency (System Time vs UI Last Report)
        ui_latency = 0
        if ui_ts > 0:
            ui_latency = (time.time() - ui_ts)
            if ui_latency > 10: # Phase 6 Final: Record spikes > 10s
                notes.append(f"⏱️ LATENCY SPIKE DETECTED ({int(ui_latency)}s)")
                
                # MODO PROFESOR: Si la latencia es extrema (> 1000s), probablemente el Dashboard está APAGADO.
                # No lanzamos alarma de "High Latency" si es evidente que es por desconexión.
                if ui_latency > 300:
                    notes.append("🖥️ DASHBOARD OFFLINE")
                else:
                    logger.warning(f"⚠️ [HEALTH] The Pulse: High latency detected ({ui_latency:.1f}s)")
            
            if ui_latency > 120: # 2 mins no UI update
                notes.append(f"UI Stale ({int(ui_latency)}s)")
                
        # --- LOGGING ---
        health_record = {
            "timestamp": timestamp,
            "status": sync_status,
            "api_balance": api_balance,
            "file_balance": file_balance,
            "ui_balance": ui_balance,
            "ui_latency_sec": round(ui_latency, 2),
            "notes": "; ".join(notes)
        }
        
        self.dh.log_health_check(health_record)
        
        if sync_status == "OK":
            logger.info(f"🩺 [HEALTH] Sync OK. Equity: ${api_balance:.2f}")

# Singleton
_supervisor = None

def start_health_supervisor():
    global _supervisor
    if _supervisor is None or not _supervisor.is_alive():
        _supervisor = HealthSupervisor()
        _supervisor.start()
    return _supervisor
