import time
from typing import Callable, List

class GlobalClock:
    """
    Sincronizador maestro de tiempo (Time Synchronization).
    Garantiza que todos los componentes operen sobre el mismo "snapshot" de tiempo exacto.
    Elimina la fragmentación temporal donde un módulo usa t=100ms y otro t=105ms.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalClock, cls).__new__(cls)
            cls._instance.current_tick_ns = time.time_ns()
            cls._instance.tick_subscribers = []
        return cls._instance

    def subscribe(self, callback: Callable[[int], None]):
        """Suscribe un módulo para recibir el pulso de reloj maestro."""
        if callback not in self.tick_subscribers:
            self.tick_subscribers.append(callback)

    def tick(self) -> int:
        """
        Dispara un nuevo ciclo de sistema.
        Congela el tiempo y notifica a todos los subsistemas dependientes.
        """
        self.current_tick_ns = time.time_ns()
        
        # Notificar a componentes críticos (como Feature Engine y SSOT state freeze)
        for callback in self.tick_subscribers:
            callback(self.current_tick_ns)
            
        return self.current_tick_ns

    def get_time_ns(self) -> int:
        """Retorna el tiempo congelado actual. Módulos DEBEN usar esto en vez de time.time()"""
        return self.current_tick_ns

# Global instance
global_clock = GlobalClock()
