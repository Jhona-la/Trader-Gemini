import psutil
import platform
import os
import time
from datetime import datetime, timezone
from utils.logger import logger
from utils.notifier import Notifier
from config import Config

class SystemMonitor:
    """
     Phase 15: Institutional System Monitoring
     Tracks CPU, RAM, Disk usage and alerts on exhaustion.
    """
    
    def __init__(self):
        self.cpu_history = []
        self.ram_history = []
        self.last_check = 0
        self.check_interval = 10 # Seconds
        self.alert_cooldown = 300 # 5 minutes
        self.last_alert_time = 0
        
        # Thresholds (Defaults if not in Config)
        self.cpu_threshold = getattr(Config.Observability, 'CPU_THRESHOLD', 85.0)
        self.ram_threshold = getattr(Config.Observability, 'RAM_THRESHOLD', 85.0)
        self.disk_min_gb = getattr(Config.Observability, 'DISK_MIN_GB', 2.0)
        
        logger.info(f"🏥 SystemMonitor initialized (CPU>{self.cpu_threshold}%, RAM>{self.ram_threshold}%)")
        
    def check_health(self) -> dict:
        """
        Polls system resources and returns a dict of metrics.
        Triggers alerts if thresholds are breached.
        """
        now = time.time()
        if now - self.last_check < self.check_interval:
            return {} # Too soon
            
        metrics = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'cpu_pct': psutil.cpu_percent(interval=None),
            'ram_pct': psutil.virtual_memory().percent,
            'ram_used_gb': psutil.virtual_memory().used / (1024**3),
            'disk_free_gb': psutil.disk_usage('.').free / (1024**3),
            'process_ram_mb': psutil.Process(os.getpid()).memory_info().rss / (1024**2)
        }
        
        self.cpu_history.append(metrics['cpu_pct'])
        self.ram_history.append(metrics['ram_pct'])
        
        # Keep history short (last 10 checks)
        if len(self.cpu_history) > 10: self.cpu_history.pop(0)
        if len(self.ram_history) > 10: self.ram_history.pop(0)
        
        self._save_to_file(metrics)
        self._analyze_risks(metrics)
        self.last_check = now
        return metrics

    def _save_to_file(self, metrics):
        """Persist metrics for Dashboard"""
        import json
        try:
            path = "dashboard/data/system_health.json"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                json.dump(metrics, f)
        except Exception as e:
            logger.warning(f"Failed to save system health: {e}")
        
    def _analyze_risks(self, metrics):
        """Check for resource exhaustion"""
        alerts = []
        
        # CPU Sustained High
        if len(self.cpu_history) >= 3 and sum(self.cpu_history[-3:]) / 3 > self.cpu_threshold:
            alerts.append(f"🔥 CPU HIGH LOAD: {metrics['cpu_pct']}%")
            
        # RAM Danger
        if metrics['ram_pct'] > self.ram_threshold:
            alerts.append(f"💾 RAM EXHAUSTION: {metrics['ram_pct']}% ({metrics['ram_used_gb']:.1f}GB Used)")
            
        # Disk Space
        if metrics['disk_free_gb'] < self.disk_min_gb:
            alerts.append(f"💿 LOW DISK SPACE: {metrics['disk_free_gb']:.2f}GB Free")
            
        # Send Alert
        if alerts and (time.time() - self.last_alert_time > self.alert_cooldown):
            msg = "\n".join(alerts)
            full_msg = f"🏥 **SYSTEM HEALTH ALERT**\n\n{msg}"
            Notifier.send_telegram(full_msg, priority="WARNING")
            logger.warning(f"System Alert: {msg}")
            self.last_alert_time = time.time()

# Singleton Instance
system_monitor = SystemMonitor()
