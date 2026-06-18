import time
import logging
from collections import deque
from config import Config

logger = logging.getLogger("TraderGemini.SystemHealth")

class SystemHealthMonitor:
    """
    [PHASE 3.3] System Health Monitor (Risk Shield)
    QUÉ: Evalúa la salud de la infraestructura y el entorno de mercado.
    POR QUÉ: Evitar pérdidas debidas a lag, bloqueos de API, o cambios repentinos de régimen que el bot no entiende.
    PARA QUÉ: Gatillar el Kill-Switch de manera proactiva.
    CÓMO: Mantiene un puntaje (0-100) basado en penalizaciones dinámicas.
    """
    
    def __init__(self, risk_manager=None, portfolio=None):
        self.risk_manager = risk_manager
        self.portfolio = portfolio
        
        # Pilares de salud
        self.api_score = 25.0
        self.latency_score = 25.0
        self.regime_score = 25.0
        self.drawdown_score = 25.0
        
        # Rastreadores
        self.recent_api_errors = deque(maxlen=20)
        self.recent_latencies = deque(maxlen=50)
        self.last_shs_eval = time.time()
        
        # Configuración de penalizaciones
        self.LATENCY_THRESHOLD_MS = 1500  # Latencia > 1500ms penaliza
        self.MICRO_LATENCY_MS = 800       # Latencia extrema para Microscalping
        
        # Recuperación (Cooldown)
        self.in_cooldown = False
        self.cooldown_end = 0.0

    def record_api_error(self, error_type="GENERAL"):
        """Registra un error de API o timeout."""
        self.recent_api_errors.append({"time": time.time(), "type": error_type})
        
    def record_latency(self, latency_ms: float):
        """Registra el Delta T (Signal -> Fill o Ping)."""
        self.recent_latencies.append({"time": time.time(), "ms": latency_ms})
        
    def get_shs(self) -> float:
        """
        Calcula y retorna el System Health Score (SHS) actual.
        100 = Perfecto. <60 = Reducir riesgo. <30 = KILL SWITCH.
        """
        now = time.time()
        
        # 1. API Health (Max 25)
        # Limpiamos errores viejos (> 5 mins)
        self.recent_api_errors = deque([e for e in self.recent_api_errors if now - e["time"] < 300], maxlen=20)
        error_count = len(self.recent_api_errors)
        self.api_score = max(0.0, 25.0 - (error_count * 5.0)) # 5 errores en 5 mins = 0 pts
        
        # 2. Execution Latency (Max 25)
        self.recent_latencies = deque([l for l in self.recent_latencies if now - l["time"] < 300], maxlen=50)
        if self.recent_latencies:
            avg_latency = sum(l["ms"] for l in self.recent_latencies) / len(self.recent_latencies)
            if avg_latency > self.LATENCY_THRESHOLD_MS:
                self.latency_score = max(0.0, 25.0 - ((avg_latency - self.LATENCY_THRESHOLD_MS) / 100.0))
            elif avg_latency > self.MICRO_LATENCY_MS and getattr(Config.Strategies, "ACTIVE_HORIZON", "SCALPING") == "MICROSCALPING":
                self.latency_score = max(0.0, 25.0 - ((avg_latency - self.MICRO_LATENCY_MS) / 50.0))
            else:
                self.latency_score = 25.0
        else:
            self.latency_score = 25.0 # Sin data, asumimos ok
            
        # 3. Winning Streak / Regime (Max 25)
        if self.risk_manager:
            # Penalizar fuertemente si hay un cúmulo de pérdidas consecutivas a nivel global
            total_consecutive = sum(self.risk_manager.consecutive_losses.values())
            self.regime_score = max(0.0, 25.0 - (total_consecutive * 8.0)) # 3 pérdidas = 1.0 pts (casi 0)
        else:
            self.regime_score = 25.0
            
        # 4. Drawdown Velocity (Max 25)
        if self.portfolio and hasattr(self.portfolio, 'peak_capital'):
            equity = self.portfolio.get_total_equity()
            peak = self.portfolio.peak_capital
            if peak > 0:
                dd_pct = (peak - equity) / peak
                # Max Drawdown tolerable asimilado a la cuenta de $13 = 10%
                max_dd_tolerado = getattr(Config.Risk, "MAX_DRAWDOWN", 0.10)
                if dd_pct > max_dd_tolerado * 0.5: # Si supera la mitad del DD máximo permitido
                    penalizacion = (dd_pct / max_dd_tolerado) * 25.0
                    self.drawdown_score = max(0.0, 25.0 - penalizacion)
                else:
                    self.drawdown_score = 25.0
            else:
                self.drawdown_score = 25.0
        else:
            self.drawdown_score = 25.0
            
        # Total SHS
        total_shs = self.api_score + self.latency_score + self.regime_score + self.drawdown_score
        
        # Cooldown management
        if self.in_cooldown:
            if now > self.cooldown_end and total_shs > 80.0:
                logger.info("🟢 [SYSTEM HEALTH] Cooldown finished. System recovered.")
                self.in_cooldown = False
            else:
                # Si estamos en cooldown, forzamos el SHS por debajo de 60 para mantener el modo restrictivo
                return min(total_shs, 59.0)
                
        if total_shs < 60.0 and not self.in_cooldown:
            logger.warning(f"⚠️ [SYSTEM HEALTH] SHS dropped to {total_shs:.1f}. Entering Cooldown Mode (5m).")
            self.in_cooldown = True
            self.cooldown_end = now + 300 # 5 minutos de cooldown
            
        return total_shs
