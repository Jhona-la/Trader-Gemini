import time

class QuantumTimer:
    """
    ⏱️ [QUANTUM TELEMETRY] Reloj atómico para auditoría de latencia HFT.
    Utiliza time.perf_counter_ns() para evitar el drift del reloj del sistema operativo.
    """
    def __init__(self):
        self.start_ns = time.perf_counter_ns()
        
    def stop(self) -> dict:
        end_ns = time.perf_counter_ns()
        elapsed_ns = end_ns - self.start_ns
        
        return {
            "ns": elapsed_ns,
            "us": elapsed_ns / 1_000,
            "ms": elapsed_ns / 1_000_000,
            "s": elapsed_ns / 1_000_000_000
        }
        
    def format_all(self) -> str:
        """Devuelve un string formateado con todas las resoluciones temporales."""
        t = self.stop()
        return f"{t['s']:.6f}s | {t['ms']:.3f}ms | {t['us']:.1f}μs | {t['ns']}ns"
