from datetime import datetime, timezone

def ensure_utc_aware(dt: datetime) -> datetime:
    """
    Valida que un datetime sea timezone-aware UTC.
    Lanza ValueError si es naive o no-UTC.
    Reference: Master Bible v2.0.1 Phase 1.1
    """
    if not isinstance(dt, datetime):
        raise TypeError(f"Expected datetime, got {type(dt)}")
        
    if dt.tzinfo is None:
        raise ValueError(f"Datetime naive detectado: {dt}. DEBE ser UTC-aware.")
    
    if dt.tzinfo != timezone.utc:
        # Convert explicit conversion relative to UTC if it has other timezone
        # But per Bible strictness, we might wan to force explicit UTC input.
        # "Lanza ValueError si es naive o no-UTC" -> implies we reject non-UTC.
        raise ValueError(f"Datetime no está en UTC: {dt.tzinfo}. Convertir a UTC.")
        
    return dt
