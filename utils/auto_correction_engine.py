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
        current_tp = Config.Horizons.Scalping.get('tp_pct', 0.0035)
        # FORENSIC-V60 FIX: Reduced multiplier from 1.25x to 1.10x
        # QUÉ: 1.25x geometric increase was pushing TP to 1.50% in ~5 iterations.
        # POR QUÉ: BTC 5m candles rarely reach 1.50% without retracing.
        # PARA QUÉ: Keep TP within achievable range for M5 scalping.
        new_tp = current_tp * 1.10  # Conservative 10% increase
        
        # [FORENSIC-V90] Cap at 0.40% to allow the 0.35% TP to survive
        if new_tp > 0.004:
            new_tp = 0.004
            
        Config.Horizons.Scalping['tp_pct'] = new_tp
        return f"Increased SCALPING_PARAMS['tp_pct'] to {new_tp*100:.2f}% to compensate for fee drag."

    def correct_frequent_small_losses(self, issue: Dict[str, Any]) -> str:
        """Reduce leverage temporalmente ante racha negativa."""
        current_lev = getattr(Config.Risk, 'BASE_LEVERAGE', 10)
        # FORENSIC-V60 FIX: Floor at 3x instead of 1x
        # QUÉ: At 1x leverage with $13 capital, position size is ~$1.30 notional.
        # POR QUÉ: At $1.30 notional, the 0.02% maker fee ($0.00026) is nearly
        #   the same magnitude as the expected PnL from a 0.20% move ($0.0026).
        #   This makes it mathematically impossible to overcome the fee floor.
        # PARA QUÉ: Keep leverage >= 3x so position sizes generate meaningful PnL.
        new_lev = max(3, current_lev // 2)
        setattr(Config.Risk, 'BASE_LEVERAGE', new_lev)
        return f"Reduced BASE_LEVERAGE to {new_lev}x due to loss streak."

    def correct_strategy_conflict(self, issue: Dict[str, Any]) -> str:
        """Desactiva temporalmente el override de Risk para resetear el loop."""
        return "" # Logic needs engine integration

    def correct_slippage_issues(self, issue: Dict[str, Any]) -> str:
        """
        FORENSIC FIX: Anteriormente forzaba STRICT_LIMIT_ONLY, causando una espiral
        de la muerte al no cruzar el spread cuando el mercado caía. Ahora simplemente
        alerta sobre el slippage sin alterar el mecanismo atómico de cruce.
        """
        # NO enforcing limit-only! That kills the HFT exit logic.
        logger.warning("🚨 [AUTO-CORRECTION] High slippage detected, but maintaining MARKET exits to preserve trade closure guarantees.")
        return "Acknowledged slippage erosion, but preserving MARKET exits to prevent Zombie Trades."

_auto_correction_sys = None
def get_auto_correction_engine() -> AutoCorrectionEngine:
    global _auto_correction_sys
    if not _auto_correction_sys:
        _auto_correction_sys = AutoCorrectionEngine()
    return _auto_correction_sys
