import time
import functools
from datetime import datetime
from config import Config

def trace_execution(func):
    """
    Decorador para trazar la ejecución de funciones críticas.
    Imprime [ENTER] y [EXIT] con tiempos de ejecución.
    Si tarda > 2.0s, emite una alerta [SLOW].
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 1. Verificar si el modo debug está activo
        if not getattr(Config, 'DEBUG_TRACE_ENABLED', False):
            return func(*args, **kwargs)

        func_name = func.__name__
        # Intentar obtener clase si es método de instancia
        if args and hasattr(args[0], '__class__'):
            func_name = f"{args[0].__class__.__name__}.{func_name}"

        # 2. Log de Entrada
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        print(f"🔹 [{timestamp}] [ENTER] {func_name}...", flush=True)
        
        start_time = time.time()
        
        try:
            # 3. Ejecución
            result = func(*args, **kwargs)
            return result
        finally:
            # 4. Log de Salida (en finally para capturar errores también)
            end_time = time.time()
            duration = end_time - start_time
            
            dur_str = f"{duration:.4f}s"
            
            if duration > 2.0:
                print(f"   ⚠️ [SLOW] {func_name} tomó {dur_str} (POSIBLE CUELLO DE BOTELLA)", flush=True)
            
            print(f"   🔸 [EXIT] {func_name} ({dur_str})", flush=True)

    return wrapper
