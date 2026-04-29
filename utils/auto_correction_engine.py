from typing import List, Dict, Any
from datetime import datetime
from config import Config
from utils.logger import logger
from utils.alert_system import get_alert_system

class AutoCorrectionEngine:
    """
    SISTEMA DE AUTO-DIAGNÓSTICO: Motor de Auto-Corrección Dinámica.
    Recibe issues del LossAnalyzer o HealthSupervisor e inyecta curas en memoria cambiando Config global.
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(AutoCorrectionEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.correction_rules = {
                "fee_death": self.correct_high_fee_ratio,
                "consistent_losses": self.correct_frequent_small_losses,
                "strategy_conflict": self.correct_strategy_conflict,
                "slippage_erosion": self.correct_slippage_issues
            }
            self.applied_corrections = []
            self.alert_sys = get_alert_system()
            self.initialized = True
            
    def apply_corrections(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        successful_corrections = []
        
        for issue in issues:
            pattern = issue.get("pattern")
            if pattern in self.correction_rules:
                correction_msg = self.correction_rules[pattern](issue)
                if correction_msg:
                    rec = {
                        "issue": pattern,
                        "correction": correction_msg,
                        "applied_at": datetime.now()
                    }
                    self.applied_corrections.append(rec)
                    successful_corrections.append(rec)
                    
                    logger.warning(f"🛠️ [AUTO-HEAL] Corrección Aplicada: {correction_msg}")
                    
                    # Subir alarma a telegram informando que el bot se modificó a sí mismo
                    if issue.get("severity") == "CRITICAL" or pattern in ["fee_death", "consistent_losses"]:
                        meta = issue.get("metadata", {})
                        kw = {k:v for k,v in meta.items()} if type(meta) == dict else {}
                        
                        # Trigger alert to telegram that we autorepaired
                        self.alert_sys.raise_alert(
                            alert_type=pattern if pattern in ["fee_death_spiral", "consistent_losses", "strategy_conflict"] else "fee_death_spiral",
                            **kw
                        )
                        
        return successful_corrections

    def correct_high_fee_ratio(self, issue: Dict[str, Any]) -> str:
        """Aumenta dinámicamente el TARGET mínimo de Take Profit para sobrevivir comisiones."""
        # Modificar el parámetro real en SCALPING_PARAMS
        current_tp = Config.Strategies.SCALPING_PARAMS.get('tp_pct', 0.0045)
        new_tp = current_tp * 1.25 # Aumento geométrico del 25% para separarnos del piso de fees
        
        # Max cap para evitar TP inalcanzables en scalping (max 1.5%)
        if new_tp > 0.015:
            new_tp = 0.015
            
        Config.Strategies.SCALPING_PARAMS['tp_pct'] = new_tp
        return f"Increased SCALPING_PARAMS['tp_pct'] to {new_tp*100:.2f}% to compensate for fee drag."

    def correct_frequent_small_losses(self, issue: Dict[str, Any]) -> str:
        """Reduce leverage temporalmente ante racha negativa."""
        current_lev = getattr(Config.Risk, 'BASE_LEVERAGE', 10)
        new_lev = max(1, current_lev // 2)
        setattr(Config.Risk, 'BASE_LEVERAGE', new_lev)
        return f"Reduced BASE_LEVERAGE drastically to {new_lev}x due to loss streak."

    def correct_strategy_conflict(self, issue: Dict[str, Any]) -> str:
        """Desactiva temporalmente el override de Risk para resetear el loop."""
        return "" # Logic needs engine integration

    def correct_slippage_issues(self, issue: Dict[str, Any]) -> str:
        """Exige órdenes EXCLUSIVAMENTE limit post-only en ejecución."""
        setattr(Config.Execution, 'STRICT_LIMIT_ONLY', True)
        return "Enforced STRICT_LIMIT_ONLY (Post-Only) execution due to massive slippage erosion."

_auto_correction_sys = None
def get_auto_correction_engine() -> AutoCorrectionEngine:
    global _auto_correction_sys
    if not _auto_correction_sys:
        _auto_correction_sys = AutoCorrectionEngine()
    return _auto_correction_sys
