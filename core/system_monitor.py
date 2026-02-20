import psutil
import time
from config import Config
from utils.logger import logger

class SystemMonitor:
    """
    🏥 COMPONENT: Disaster Resilience (Graceful Degradation)
    QUÉ: Monitor de salud del sistema (CPU, RAM, Latencia).
    POR QUÉ: Si el sistema está saturado, debemos degradar funcionalidades no esenciales (Entrenamiento, Logs verbales).
    """
    
    def __init__(self):
        self.last_check = 0
        self.check_interval = 5.0 # Seconds
        self.is_safe_mode = False
        
    def check_health(self) -> bool:
        """
        Returns True if system is healthy, False if degraded (Safe Mode).
        """
        now = time.time()
        if now - self.last_check < self.check_interval:
            return not self.is_safe_mode
            
        self.last_check = now
        
        # 1. CPU Check
        cpu_usage = psutil.cpu_percent(interval=None)
        
        # 2. Memory Check
        mem = psutil.virtual_memory()
        mem_usage = mem.percent
        
        # Thresholds
        CRITICAL_CPU = 90.0
        CRITICAL_MEM = 85.0
        
        if cpu_usage > CRITICAL_CPU or mem_usage > CRITICAL_MEM:
            if not self.is_safe_mode:
                logger.warning(f"⚠️ [SystemMonitor] HIGH LOAD DETECTED (CPU={cpu_usage}%, MEM={mem_usage}%). Entering SAFE MODE.")
                self.enter_safe_mode()
            return False
            
        # Recovery
        if self.is_safe_mode:
            if cpu_usage < (CRITICAL_CPU - 10) and mem_usage < (CRITICAL_MEM - 10):
                logger.info(f"✅ [SystemMonitor] Load Normalized (CPU={cpu_usage}%, MEM={mem_usage}%). Exiting SAFE MODE.")
                self.exit_safe_mode()
                
        return True
        
    def enter_safe_mode(self):
        self.is_safe_mode = True
        # Disable heavy logging or training tasks
        
    def exit_safe_mode(self):
        self.is_safe_mode = False
