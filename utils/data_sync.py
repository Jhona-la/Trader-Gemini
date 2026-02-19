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
# import json (Removed Phase 3)
import time
from typing import Dict, Any

def atomic_write_json(data: Dict[str, Any], filepath: str):
    """
    Write JSON data to a file atomically (Highest Performance).
    1. Serialize with orjson (bytes)
    2. Write to .tmp file (binary)
    3. Rename .tmp to target file
    """
    from utils.fast_json import FastJson
    temp_path = f"{filepath}.tmp"
    try:
        # Serialize first (Catch serialization errors before touching disk)
        # Using FastJson wrapper ensures consistent settings
        json_bytes = FastJson.dumps(data).encode('utf-8') 
        # Wait, FastJson.dumps returns str. orjson returns bytes.
        # Let's use orjson directly here for bytes efficiency?
        # fast_json.py implementation: return orjson.dumps(...).decode('utf-8')
        # We want bytes for file write.
        
        import orjson
        # OPT_NAIVE_UTC | OPT_SERIALIZE_NUMPY | OPT_INDENT_2
        json_bytes = orjson.dumps(data, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2)

        # Write to temp file (Binary)
        with open(temp_path, 'wb') as f:
            f.write(json_bytes)
        
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
