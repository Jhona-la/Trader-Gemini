import os
import json
import gzip
import traceback
import platform
import psutil
from datetime import datetime, timezone
from config import Config
from utils.logger import logger

class ForensicRecorder:
    """
    🕵️ FORENSIC BLACK BOX
    Records deep system state during Crashes, Kill-Switch events, or Anomalies.
    Saves compressed snapshots for post-mortem analysis (Loki/ELK ready format).
    """
    def __init__(self, engine):
        self.engine = engine
        self.dump_dir = os.path.join(Config.BASE_DIR, 'logs', 'forensics')
        os.makedirs(self.dump_dir, exist_ok=True)
        
    def capture_snapshot(self, trigger_reason: str, exception: Exception = None):
        """
        Captures entire system state synchronously (Blocking IS intended here).
        We want to freeze the crime scene.
        """
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # 1. System Vital Signs
            system_stats = {
                'cpu_percent': psutil.cpu_percent(interval=None),
                'memory_percent': psutil.virtual_memory().percent,
                'threads': psutil.Process().num_threads(),
                'platform': platform.platform()
            }
            
            # 2. Portfolio State
            portfolio_state = {}
            if self.engine.portfolio:
                portfolio_state = {
                    'cash': self.engine.portfolio.current_cash,
                    'equity': self.engine.portfolio.get_total_equity(),
                    'positions': self.engine.portfolio.positions,
                    'realized_pnl': self.engine.portfolio.realized_pnl
                }
                
            # 3. Strategy State (Deep Dive)
            strategy_dump = {}
            for strat in self.engine.strategies:
                if hasattr(strat, 'export_state'):
                    strategy_dump[strat.strategy_id] = strat.export_state()
                else:
                    strategy_dump[str(strat)] = "No export_state method"

            # 4. Recent Event History (if available)
            # We assumes Engine might have a debug history or we just skip
            
            snapshot = {
                'timestamp': timestamp,
                'trigger': trigger_reason,
                'exception': ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__)) if exception else None,
                'system': system_stats,
                'portfolio': portfolio_state,
                'strategies': strategy_dump
            }
            
            filename = f"crash_dump_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json.gz"
            filepath = os.path.join(self.dump_dir, filename)
            
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                json.dump(snapshot, f, indent=2, default=str)
                
            logger.critical(f"📼 [FORENSICS] BLACK BOX SAVED: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Failed to save Forensic Snapshot: {e}")
            return None
