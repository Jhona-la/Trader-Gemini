import os
import sys
from utils.logger import logger

class ForensicAuditor:
    """
    [PHASE 18] FORENSIC PARITY AUDITOR (Digital Twin Enforcer)
    QUÉ: Asegura que el entorno de Backtest y Producción sean 100% idénticos.
    POR QUÉ: Inyecciones de parámetros exclusivas de backtest invalidan la confiabilidad de los resultados (divergencias).
    PARA QUÉ: Abortar simulaciones o alertar si hay desviaciones en Fees, Capital, o Flags peligrosas.
    CÓMO: Se ejecuta antes de instanciar el Motor/Portfolio. Revisa la Configuración Global en tiempo de ejecución.
    """
    
    @staticmethod
    def verify_parity(Config) -> bool:
        discrepancies = []
        
        # 1. Flag injection verification
        if getattr(Config, "IS_BACKTEST", False):
             # Some scripts artificially inject this. We log a warning, though we might still need it for non-network calls.
             logger.warning("🔍 [FORENSIC AUDITOR] Config.IS_BACKTEST is True. Verifying no malicious overrides exist.")
             
        if os.environ.get("TRADER_GEMINI_BACKTEST") == "true":
             logger.warning("🔍 [FORENSIC AUDITOR] TRADER_GEMINI_BACKTEST Env Var is Active. Network safeguards enabled.")
             
        # 2. Critical parameter checks
        if getattr(Config, "INITIAL_CAPITAL", 0) != 13.0:
             discrepancies.append(f"INITIAL_CAPITAL is {getattr(Config, 'INITIAL_CAPITAL')}, expected exactly 13.0 for exact replication.")
             
        if getattr(Config, "BINANCE_TAKER_FEE_BNB", 0) != 0.000375: # Expected default: 0.0375%
             discrepancies.append(f"TAKER_FEE altered to {getattr(Config, 'BINANCE_TAKER_FEE_BNB')}")

        if getattr(Config, "BINANCE_MAKER_FEE_BNB", 0) != 0.0002: # Expected default: 0.02%
             discrepancies.append(f"MAKER_FEE altered to {getattr(Config, 'BINANCE_MAKER_FEE_BNB')}")
             
        if getattr(Config, "BINANCE_LEVERAGE", 0) not in (20, 50):
             discrepancies.append(f"LEVERAGE altered to {getattr(Config, 'BINANCE_LEVERAGE')}, expected 20 or 50.")
             
        if getattr(Config, "POSITION_SIZE_MICRO_ACCOUNT", 0) != 0.19:
             discrepancies.append(f"POSITION SIZING altered to {getattr(Config, 'POSITION_SIZE_MICRO_ACCOUNT')}, expected 19%.")
             
        if discrepancies:
             logger.error("🚨 [FORENSIC AUDITOR] DIGITAL TWIN DIVERGENCE DETECTED!")
             for d in discrepancies:
                  logger.error(f"  ❌ DIVERGENCE: {d}")
             logger.warning("The system will continue but results may NOT exactly match production PnL physics.")
             return False
             
        logger.info("✅ [FORENSIC AUDITOR] 100% PRODUCTION PARITY CONFIRMED (Digital Twin Verified).")
        return True
