"""
🧮 PROTOCOLO CRITERIO-AXIOMA: Integridad Aritmética (THE CALC-CHECKER)
====================================================================

QUÉ: Módulo de validación estricta de precisión usando tipo Decimal.
POR QUÉ: En operaciones de alta frecuencia, C++/Numba o Python truncan los float64,
     lo que genera fugas (precision loss) que arruinan la Ecuación Contable.
PARA QUÉ: Detectar inmediatamente desviaciones matemáticas antes de que propaguen.
CÓMO: Auditor que re-calcula operaciones críticas (PnL, Size) usando Decimal
      y compara el delta contra un épsilon extremadamente bajo (1e-7).
DÓNDE: utils/axioma_math.py
QUIÉN: Portfolio y engine lo invocan para double-checks vitales.
"""

from decimal import Decimal, getcontext
from utils.logger import logger
import traceback

class PrecisionAuditor:
    """
    Motor de alta precisión para auditar los float64 del engine de trading.
    """
    
    # 28 decimales de precisión en Python math por defecto
    getcontext().prec = 28
    
    # Tolerancia estricta para considerar "Corrupción Aritmética"
    STRICT_EPSILON = Decimal('0.0000001') # 1e-7
    
    @staticmethod
    def verify_pnl(entry_price: float, exit_price: float, quantity: float, engine_pnl: float) -> bool:
        """
        Audita el cálculo de PnL (Profit and Loss) re-calculándolo con Decimals.
        Retorna True si la validación ES EXACTA, o levanta warning/alerta.
        """
        try:
            # Convertimos strings limpios para evitar float artifacts
            d_entry = Decimal(str(entry_price))
            d_exit = Decimal(str(exit_price))
            d_qty = Decimal(str(quantity))
            
            # Pnl = (exit_price - entry_price) * quantity (en base account currency)
            d_pnl = (d_exit - d_entry) * d_qty
            
            d_engine_pnl = Decimal(str(engine_pnl))
            delta = abs(d_pnl - d_engine_pnl)
            
            if delta > PrecisionAuditor.STRICT_EPSILON:
                logger.error(
                    f"🚨 [AXIOMA] PRECISION LOSS en PnL! Delta="
                    f"{delta.normalize():f} > {PrecisionAuditor.STRICT_EPSILON}. "
                    f"FloatEngine={engine_pnl}, Strict={d_pnl}"
                )
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"⚠️ [AXIOMA] Falló la validación estricta de PnL: {e}")
            return False

    @staticmethod
    def verify_fraction(numerator: float, denominator: float, engine_result: float) -> bool:
        """
        Audita operaciones fraccionales críticas (ej: Multiplicadores Kelly Criterion o Size)
        """
        try:
            # Protege divide by zero
            if denominator == 0:
                if engine_result != 0:
                    logger.error(f"🚨 [AXIOMA] DIV/0 pero engine devolvió {engine_result}")
                    return False
                return True
                
            d_num = Decimal(str(numerator))
            d_den = Decimal(str(denominator))
            
            d_res = d_num / d_den
            d_engine_res = Decimal(str(engine_result))
            
            delta = abs(d_res - d_engine_res)
            
            if delta > PrecisionAuditor.STRICT_EPSILON:
                logger.error(
                    f"🚨 [AXIOMA] PRECISION LOSS en División! Delta="
                    f"{delta.normalize():f} > {PrecisionAuditor.STRICT_EPSILON}. "
                    f"FloatEngine={engine_result}, Strict={d_res}"
                )
                return False
                
            return True
        except Exception as e:
            logger.error(f"⚠️ [AXIOMA] Falló la validación de fracción: {e}")
            return False

