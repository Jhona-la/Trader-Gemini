"""
🔄 DATA SYNC UTILS
==================

PROFESSOR METHOD:
- QUÉ: Utilidades para sincronización segura de datos entre procesos.
- POR QUÉ: Para evitar corrupción de datos por lecturas/escrituras concurrentes.
- PARA QUÉ: Integridad de datos en sistema multi-hilo (Bot + Dashboard).
- CÓMO: Escritura atómica (temp -> rename).
- CUÁNDO: En cada guardado de estado.
- DÓNDE: Usado por API Manager y Portfolio.
"""

import os
import json
import time
from typing import Dict, Any

def atomic_write_json(data: Dict[str, Any], filepath: str):
    """
    Write JSON data to a file atomically.
    1. Write to .tmp file
    2. Renaissance .tmp to target file
    """
    temp_path = f"{filepath}.tmp"
    try:
        # Write to temp file
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Atomic rename (overwrite)
        os.replace(temp_path, filepath)
        return True
    except Exception as e:
        # Cleanup
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return False

def touch_timestamp(filepath: str):
    """Update file modification time to signal changes."""
    try:
        with open(filepath, 'a'):
            os.utime(filepath, None)
    except:
        pass
