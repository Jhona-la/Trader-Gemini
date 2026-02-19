
import os
import psutil
from utils.logger import setup_logger

logger = setup_logger("CPUAffinity")

class CPUManager:
    """
    🔬 PHASE 29: CPU AFFINITY
    Pins the trading process to specific CPU cores to minimize Context Switching.
    
    Windows/Linux compatible via psutil.
    """
    
    @staticmethod
    def pin_process(core_ids: list[int] = None):
        """
        Pins current process to specified core IDs.
        AEGIS-ULTRA: Defaults to Physical Cores (0, 2, 4, 6) for Ryzen 5700U if Config enabled.
        """
        try:
            from config import Config
            p = psutil.Process()
            available_cores = psutil.cpu_count(logical=True)
            
            # AEGIS-ULTRA: Hardware Profile
            if not core_ids and hasattr(Config, 'Aegis') and Config.Aegis.CORE_PINNING:
                # Ryzen 5700U: 8 Cores / 16 Threads
                # Windows usually maps Physical Cores to 0, 2, 4, 6, 8, 10, 12, 14
                if available_cores >= 16:
                     # Pin to first 4 Physical Cores to avoid Hyper-Threading contention
                     # and leave the rest for OS/Background tasks.
                     core_ids = [0, 2, 4, 6] 
                     logger.info(f"⚙️ [AEGIS] Ryzen 5700U Profile: Pinning to Physical Cores {core_ids}")
                elif available_cores >= 8:
                     core_ids = [0, 2, 4, 6]
                else:
                     # Fallback
                     core_ids = [i for i in range(available_cores) if i % 2 == 0]

            if not core_ids:
                # Auto-strategy (Legacy):
                # If we have > 4 cores, pin to [2, 3] just to be safe from OS noise on 0,1
                if available_cores >= 4:
                    core_ids = [2, 3] # Use cores 2 and 3
                elif available_cores >= 2:
                    core_ids = [1] # Use core 1
                else:
                    logger.warning(f"CPU Affinity: Only {available_cores} core(s). Skipping pinning.")
                    return

            # Validate
            core_ids = [c for c in core_ids if c < available_cores]
            if not core_ids:
                return

            # Apply
            p.cpu_affinity(core_ids)
            logger.info(f"🔬 CPU Affinity: Process pinned to Cores {core_ids}")
            
        except Exception as e:
            logger.error(f"Failed to set CPU affinity: {e}")

    @staticmethod
    def set_priority(level: str = "HIGH"):
        """
        Sets process priority.
        Levels: NORMAL, HIGH, REALTIME
        WARNING: REALTIME can freeze Windows if the process hangs. Use with caution.
        """
        try:
            p = psutil.Process()
            if os.name == 'nt':
                if level == "REALTIME":
                    p.nice(psutil.REALTIME_PRIORITY_CLASS)
                elif level == "HIGH":
                    p.nice(psutil.HIGH_PRIORITY_CLASS)
                else:
                    p.nice(psutil.NORMAL_PRIORITY_CLASS)
            else:
                # Linux nice values: -20 (max priority) to 19 (min)
                if level == "REALTIME":
                    p.nice(-20)
                elif level == "HIGH":
                    p.nice(-10)
                else:
                    p.nice(0)
            
            logger.info(f"🚀 Process Priority set to {level}")
        except Exception as e:
            logger.warning(f"Could not set process priority to {level}: {e}")
