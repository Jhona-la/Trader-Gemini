from typing import Dict, List, Any
from utils.logger import logger

class LossAnalyzer:
    """
    SISTEMA DE AUTO-DIAGNÓSTICO: Analizador de Dinámicas de Pérdida.
    Detecta de forma post-mortem o in-vivo por qué las posiciones pierden dinero (ej. Fees).
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LossAnalyzer, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.loss_patterns = {
                "fee_death": self.detect_fee_death,
                "slippage_erosion": self.detect_slippage_erosion,
                "overtrading": self.detect_overtrading
            }
            self.streak_analyzer = {"consecutive_losses": 0}
            self.initialized = True
            
    def analyze_trade(self, trade_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Ejecuta el diagnóstico sobre un trade completado.
        trade_data: dict con 'gross_pnl', 'net_pnl', 'fees', 'slippage_pct', 'duration_sec'
        """
        issues = []
        
        # Track streak
        if trade_data.get("net_pnl", 0) < 0:
            self.streak_analyzer["consecutive_losses"] += 1
        else:
            self.streak_analyzer["consecutive_losses"] = 0
            
        if self.streak_analyzer["consecutive_losses"] >= 5:
             issues.append({
                 "pattern": "consistent_losses",
                 "description": "5+ pérdidas consecutivas.",
                 "severity": "HIGH",
                 "corrective_action": "Activar cooldown de mercado.",
                 "metadata": {"loss_streak": self.streak_analyzer["consecutive_losses"]}
             })
        
        for pattern_name, pattern_func in self.loss_patterns.items():
            result = pattern_func(trade_data)
            if result:
                issues.append({
                    "pattern": pattern_name,
                    "description": self.get_pattern_description(pattern_name),
                    "severity": self.get_pattern_severity(pattern_name),
                    "corrective_action": self.get_corrective_action(pattern_name),
                    "metadata": result
                })
                
        return issues
        
    def get_pattern_description(self, pattern: str) -> str:
        descriptions = {
            "fee_death": "Las comisiones representan más del 40% del PnL Bruto, neutralizando la ganancia.",
            "slippage_erosion": "Slippage en entrada/salida superior al esperado erosionando el margen.",
            "overtrading": "Alta frecuencia de entradas sin confirmación de rango."
        }
        return descriptions.get(pattern, "Patrón de pérdida anómalo.")
        
    def get_pattern_severity(self, pattern: str) -> str:
        severities = {
            "fee_death": "CRITICAL",
            "slippage_erosion": "HIGH",
            "overtrading": "MEDIUM"
        }
        return severities.get(pattern, "LOW")
        
    def get_corrective_action(self, pattern: str) -> str:
        actions = {
            "fee_death": "Elevar TAKE_PROFIT_MIN_THRESHOLD para absorber cost drags.",
            "slippage_erosion": "Uso estricto de LIMIT post-only o tightening de slippage guard.",
            "overtrading": "Añadir cooldown de 5 mins extra."
        }
        return actions.get(pattern, "Investigación manual.")

    def detect_fee_death(self, trade_data: Dict[str, Any]) -> dict:
        """Detectar si los fees destruyen la rentabilidad (Gross PNL > 0 pero Net PNL = 0 o negativo)"""
        # Ignorar FLIP_EXIT y TURBO_BE, ya que son cierres defensivos donde el micro-profit 
        # distorsiona el cálculo del ratio de fees, causando falsos positivos.
        exit_reason = trade_data.get("exit_reason", "")
        if exit_reason in ["FLIP_EXIT", "TURBO_BE", "TIME_STOP_ZOMBIE", "PREDICTIVE_DECAY"]:
            return {}
            
        gross = trade_data.get("gross_pnl", 0)
        fees = trade_data.get("fees", 0)
        net = trade_data.get("net_pnl", 0)
        
        if gross > 0 and fees > 0:
            fee_ratio = fees / gross
            if fee_ratio > 0.4: # Las comisiones se llevan más del 40%
                return {"fee_ratio": fee_ratio * 100}
        
        # O net_pnl muy negativo en un short timeframe
        return {}

    def detect_slippage_erosion(self, trade_data: Dict[str, Any]) -> dict:
        slippage_pct = trade_data.get("slippage_pct", 0)
        if abs(slippage_pct) > 0.15: # Slippage mayor a 0.15% en Scalping es mortal
            return {"slippage_diff": slippage_pct}
        return {}

    def detect_overtrading(self, trade_data: Dict[str, Any]) -> dict:
        duration = trade_data.get("duration_sec", 999)
        net = trade_data.get("net_pnl", 0)
        
        # Entró y salió en pérdida en menos de 30 segundos
        if duration < 30 and net < 0:
            return {"action_velocity": "hyper_active"}
        return {}

_loss_analyzer = None
def get_loss_analyzer() -> LossAnalyzer:
    global _loss_analyzer
    if not _loss_analyzer:
        _loss_analyzer = LossAnalyzer()
    return _loss_analyzer
