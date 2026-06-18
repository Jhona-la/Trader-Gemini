# core/metal/ingester_bridge.py
# FASE II: MEMBRANA FFI CERO-ALLOC (Cython)

import os
import platform

class QuantumRetina:
    def __init__(self):
        try:
            # Importar módulo Cython compilado en-sitio
            import quantum_ingester
            self._ingest_raw_ws_frame = quantum_ingester.ingest_raw_ws_frame
            self.available = True
        except ImportError as e:
            print(f"[RETINA] Advertencia: Librería Cython no encontrada. Fallback a Python. Error: {e}")
            self.available = False

    def ingest(self, arena_pointer, raw_bytes: bytes, batch_idx: int = 0) -> int:
        """
        Inyecta el payload bruto (bytes) directamente a Cython/C.
        Zero-Copy FFI boundary.
        """
        if not self.available:
            return 1 # Fallback status
        
        # arena_pointer debe ser el int devuelto por id() o ctype address, 
        # pero en Cython lo recibimos como un entero (size_t)
        return self._ingest_raw_ws_frame(arena_pointer, raw_bytes, batch_idx)

# Singleton
retina_bridge = QuantumRetina()
