import os
import sys
import psutil
import socket
import ctypes
from utils.logger import logger

class OSTuner:
    """
    🌑 COMPONENT: Layer 0 Optimizer (Protocol Nadir-Soberano)
    QUÉ: Ajusta el Sistema Operativo y el Kernel para HFT.
    POR QUÉ: Reducir latencia de interrupciones y mejorar estabilidad de red.
    """

    @staticmethod
    def tune_process_priority():
        """
        Sets the process priority to HIGH/REALTIME to reduce context switching latency.
        Requiere Admin/Elevated Privileges en Windows para REALTIME, pero HIGH es seguro.
        """
        try:
            pid = os.getpid()
            p = psutil.Process(pid)
            
            # Windows Priority Classes
            # HIGH_PRIORITY_CLASS = 0x00000080
            # ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
            
            if sys.platform == 'win32':
                p.nice(psutil.HIGH_PRIORITY_CLASS)
                logger.info(f"🚀 [Layer 0] Process Priority set to HIGH (PID: {pid})")
            else:
                p.nice(-10) # Linux/Mac (Negative is higher priority)
                logger.info(f"🚀 [Layer 0] Process Nice set to -10 (PID: {pid})")
                
        except Exception as e:
            logger.warning(f"⚠️ [Layer 0] Failed to set Process Priority: {e}")

    @staticmethod
    def tune_network_stack():
        """
        Disables Nagle's Algorithm (TCP_NODELAY) globally for this process.
        Esto fuerza a que los paquetes pequeños (órdenes) salgan inmediatamente.
        """
        try:
            # Monkey Patch socket to force TCP_NODELAY on new connections
            raw_socket = socket.socket
            
            def new_socket(*args, **kwargs):
                s = raw_socket(*args, **kwargs)
                try:
                    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                except Exception:
                    pass # Not a TCP socket or OS doesn't support it
                return s
            
            socket.socket = new_socket
            logger.info("⚡ [Layer 0] TCP_NODELAY enforced globally (Nagle Disabled)")
            
        except Exception as e:
            logger.error(f"❌ [Layer 0] Network Stack Tuning Failed: {e}")

    @staticmethod
    def set_cpu_affinity():
        """
        Pin process to Performance Cores on Ryzen 5700U (8 Cores, 16 Threads).
        Evita que el proceso salte entre núcleos, invalidando la caché L1/L2.
        Stragegy: Use physical cores 0, 2, 4, 6 (skip SMT threads if possible or use all phys).
        """
        try:
            p = psutil.Process()
            # Ryzen 5700U has 8 physical cores. Let's pin to the last 4 physical cores 
            # to avoid interference from OS tasks usually on Core 0.
            # Logical indices: 0-15. Physicals are usually event numbers or first half.
            # Simple approach: Use Cores 4-7 (Logical 8-15) for isolation.
            
            # Use last 8 logical processors (High performance cores usually)
            allowed_cpus = list(range(8, 16)) if psutil.cpu_count() >= 16 else list(range(psutil.cpu_count()))
            
            p.cpu_affinity(allowed_cpus)
            logger.info(f"🎯 [Layer 0] CPU Affinity set to cores: {allowed_cpus}")
            
        except Exception as e:
            logger.warning(f"⚠️ [Layer 0] CPU Affinity Failed: {e}")

    @staticmethod
    def optimize():
        """Run all optimizations."""
        logger.info("🌑 [NADIR-SOBERANO] Initiating Layer 0 Optimization...")
        OSTuner.tune_process_priority()
        OSTuner.tune_network_stack()
        OSTuner.set_cpu_affinity()
