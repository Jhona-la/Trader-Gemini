class SystemIntegrityError(Exception):
    """
    Excepción crítica levantada cuando se detecta un contrato roto o un dato 
    requerido faltante en el hot-path. 
    
    El sistema debe abortar el tick inmediatamente en lugar de asumir un 
    valor fallback (como 0.0 o {}), asegurando un colapso rastreable (Fail-Fast)
    para no alimentar basura al oráculo de Machine Learning.
    """
    pass

class NaNIntegrityError(SystemIntegrityError):
    """
    Excepción lanzada cuando se detecta un valor NaN en un tensor antes 
    de entrar a la inferencia de la Red Neuronal o XGBoost. Previene la 
    'Demencia Predictiva' (donde la IA inventa respuestas basadas en inputs inválidos).
    """
    pass
