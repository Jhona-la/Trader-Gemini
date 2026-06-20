import os
import traceback
from utils.logger import logger

class SystemIntegrityError(Exception):
    """
    Excepción crítica levantada cuando se detecta un contrato roto o un dato 
    requerido faltante en el hot-path. 
    
    El sistema debe abortar el tick inmediatamente en lugar de asumir un 
    valor fallback (como 0.0 o {}), asegurando un colapso rastreable (Fail-Fast)
    para no alimentar basura al oráculo de Machine Learning.
    
    [PHASE V] Dogma de Muerte Digna: Autodestrucción agresiva (OOM kill code 137).
    """
    def __init__(self, message="System Integrity Compromised"):
        super().__init__(message)
        logger.critical(f"💀 [FAIL-FAST] SYSTEM INTEGRITY ERROR: {message}")
        logger.critical(traceback.format_exc())
        logger.critical("💀 [FAIL-FAST] Ejecutando autodestrucción inmediata (os._exit 137)...")
        os._exit(137)

class NaNIntegrityError(SystemIntegrityError):
    """
    Excepción lanzada cuando se detecta un valor NaN en un tensor antes 
    de entrar a la inferencia de la Red Neuronal o XGBoost. Previene la 
    'Demencia Predictiva' (donde la IA inventa respuestas basadas en inputs inválidos).
    """
    def __init__(self, message="NaN Detected in Tensor"):
        super().__init__(message)

