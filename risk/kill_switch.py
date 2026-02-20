
import os
import sys
import time
import asyncio
from utils.logger import setup_logger
from config import Config

logger = setup_logger("KillSwitch")

class KillSwitch:
    """
    🛡️ PHASE 19: ATOMIC INTEGRITY KILL-SWITCH
    Safety mechanism to stop trading immediately.
    
    [SS-010 FIX] Replaced destructive sys.exit(1) with cooperative shutdown:
    - Sets self.active = True to block all new orders
    - Writes atomic lock file for restart prevention
    - Signals shutdown_callback for Engine to handle graceful cleanup
    - Engine is responsible for: close positions → flush DB → close WebSockets → exit
    
    Features:
    1. Soft Stop: Signal engine to close positions, then exit (Normal risk limit).
    2. Hard Stop: Set active flag + lock file (Critical bug). Engine decides exit.
    3. Atomic Lock: File-based persistence to prevent auto-restart loops.
    """
    
    LOCK_FILE = "STOP_TRADING.LOCK"
    
    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.active = False
        self.activation_reason = "NONE"
        self.peak_equity = 0.0
        self.daily_losses = 0
        self.api_errors = 0
        self.MAX_DAILY_LOSSES = getattr(Config, 'MAX_DAILY_LOSSES', 5)
        self.MAX_API_ERRORS = 10
        
        # [SS-010] Cooperative shutdown mechanism
        self._shutdown_callback = None  # Callable set by Engine for graceful stop
        self._forensic_callback = None # Phase 20: Forensic Dump
        
        # Check integrity on startup
        if self.check_atomic_lock():
            self.active = True
            self.activation_reason = "ATOMIC_LOCK_FOUND"
            logger.critical("🚨 ATOMIC LOCK FOUND! Bot has been permanently disabled.")
            logger.critical(f"   Remove '{self.LOCK_FILE}' manually to restart.")
            # [SS-010 FIX] Don't sys.exit here — let main.py check and exit cleanly
            # The startup code in main.py should call check_status() before starting the loop.

    def set_shutdown_callback(self, callback):
        """
        [SS-010] Register a callback the Engine provides for graceful shutdown.
        Callback signature: callback(reason: str) -> None
        The Engine will: close positions → flush DB → stop event loop.
        """
        self._shutdown_callback = callback

    def set_forensic_callback(self, callback):
        """Phase 20: Register callback for Black Box recording."""
        self._forensic_callback = callback

    def check_status(self):
        """Returns True if trading is allowed, False if Kill Switch is active."""
        if self.active:
            return False
        if self.check_atomic_lock():
            self.active = True
            self.activation_reason = "MANUAL_LOCK_FOUND"
            return False
        return True

    def record_loss(self):
        """Record a losing trade and check daily limits."""
        self.daily_losses += 1
        if self.daily_losses >= self.MAX_DAILY_LOSSES:
            self.activate(f"MAX_DAILY_LOSSES_REACHED ({self.MAX_DAILY_LOSSES})")

    def record_api_error(self):
        """Record an API error and check for system instability."""
        self.api_errors += 1
        if self.api_errors >= self.MAX_API_ERRORS:
            self.activate(f"MAX_API_ERRORS_REACHED ({self.MAX_API_ERRORS})")

    def reset_api_errors(self):
        """Reset the API error counter."""
        self.api_errors = 0

    def update_equity(self, current_equity):
        """Update peak equity and check for extreme drawdown."""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - current_equity) / self.peak_equity
            self.check_triggers(drawdown)

    def check_triggers(self, current_drawdown):
        """
        Routine check called by Engine.
        """
        # 1. Max Drawdown Check (Institutional Hard Limit Phase 71)
        if current_drawdown > 0.015: # 1.5% Strict Limit
            logger.critical(f"🚨 KILL SWITCH TRIGGERED: Drawdown {current_drawdown*100:.2f}% > 1.5%")
            self.activate("MAX_DRAWDOWN_EXCEEDED")
            
        # 2. Atomic Lock External Check (If user placed file manually)
        if self.check_atomic_lock():
             self.activate("MANUAL_LOCK_FOUND")

    def activate(self, reason="UNKNOWN"):
        """
        [SS-010 FIX] Cooperative Kill Switch activation.
        
        QUÉ: Detiene trading y señaliza al Engine para shutdown graceful.
        POR QUÉ: sys.exit(1) dejaba posiciones huérfanas en Binance,
                 corrompía SQLite WAL, y no flusheaba logs.
        CÓMO: 1) Flag → bloquea órdenes, 2) Lock file → previene restart,
              3) Callback → Engine cierra posiciones → flush → exit limpio.
        """
        if self.active: return
        self.active = True
        self.activation_reason = reason
        
        logger.critical(f"🛑 KILL SWITCH ACTIVATED: {reason}")
        
        # 1. Persist the Stop (Atomic Lock) — survives process restart
        self._create_atomic_lock(reason)
        
        # 🕵️ Phase 20: Forensic Snapshot (Before Shutdown)
        if self._forensic_callback:
            try:
                logger.warning("🕵️ Capturing Forensic Snapshot...")
                self._forensic_callback(reason)
            except Exception as e:
                logger.error(f"Forensic snapshot failed: {e}")

        # 2. Signal Engine for graceful shutdown
        # [SS-010 FIX] Engine callback handles: close positions → flush DB → stop loop
        if self._shutdown_callback:
            try:
                logger.critical("💀 Requesting graceful shutdown via Engine callback...")
                self._shutdown_callback(reason)
            except Exception as e:
                logger.error(f"Shutdown callback failed: {e}")
                # Fallback: still don't sys.exit — self.active=True blocks all orders
        else:
            logger.warning("⚠️ No shutdown callback registered. Orders blocked but process continues.")
            logger.warning("   Engine should check kill_switch.active and handle shutdown.")

    def check_atomic_lock(self):
        return os.path.exists(self.LOCK_FILE)

    def _create_atomic_lock(self, reason):
        try:
            with open(self.LOCK_FILE, "w") as f:
                f.write(f"KILLED AT {time.time()}\nREASON: {reason}\n")
            logger.warning(f"🔒 Atomic Lock file created: {self.LOCK_FILE}")
        except Exception as e:
            logger.error(f"Failed to create lock file: {e}")
