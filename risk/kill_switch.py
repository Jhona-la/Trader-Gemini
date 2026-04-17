
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
        Routine check called by Engine. Dynamic limit based on Horizon (Phase 3.2).
        """
        import math
        horizon_str = getattr(Config.Strategies, 'ACTIVE_HORIZON', '1D')
        horizon_days = int(horizon_str.replace('D', '')) if 'D' in horizon_str else 1
        h_sqrt = math.sqrt(horizon_days)
        
        # Scale drawdown by square root of time (1D = 2.0%, 7D = 5.2%, 15D = 7.7%, 30D = 10.9%)
        max_dd = 0.02 * h_sqrt

        # 1. Max Drawdown Check (Shadow-Monitor Sovereign Limit)
        if current_drawdown > max_dd:
            logger.critical(f"🚨 KILL SWITCH: Drawdown {current_drawdown*100:.2f}% > Limit {max_dd*100:.2f}%")
            self.activate("MAX_DRAWDOWN_EXCEEDED")
            
        # 2. Hard Capital Floor Protection
        if self.portfolio:
            current_capital = self.portfolio.get_total_equity()
            max_loss_floor_pct = 0.04 * h_sqrt  # 1D = 4%, 30D = 21.9%
            min_capital = Config.INITIAL_CAPITAL * max(0.01, (1.0 - max_loss_floor_pct))
            
            if current_capital <= min_capital and current_capital > 0:
                logger.critical(f"🚨 SOVEREIGN KILL SWITCH L2: Capital {current_capital:.2f} <= ${min_capital:.2f} (Floor {max_loss_floor_pct*100:.1f}%).")
                self.activate("CRITICAL_CAPITAL_FLOOR_REACHED")
            
        # 3. Atomic Lock External Check (If user placed file manually)
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

        # 📢 Phase 4.5: Enhanced Notification — CRITICAL RISK ALERT
        try:
            from utils.notifier import Notifier
            
            # Gather context for the alert
            balance = 0.0
            drawdown_pct = 0.0
            open_positions = 0
            if self.portfolio:
                balance = self.portfolio.get_total_equity()
                if self.peak_equity > 0:
                    drawdown_pct = ((self.peak_equity - balance) / self.peak_equity) * 100
                open_positions = len([
                    s for s, p in self.portfolio.positions.items() 
                    if p.get('quantity', 0) != 0
                ])
            
            Notifier.send_risk_alert({
                'type': f'KILL_SWITCH: {reason}',
                'level': 'critical',
                'message': (
                    f"☠️ *El Kill Switch ha sido ACTIVADO.*\n"
                    f"Todas las órdenes nuevas están BLOQUEADAS.\n"
                    f"Razón: `{reason}`\n\n"
                    f"El archivo `{self.LOCK_FILE}` ha sido creado.\n"
                    f"Elimínalo manualmente para reiniciar el bot."
                ),
                'drawdown': drawdown_pct,
                'balance': balance,
                'open_positions': open_positions,
                'recommended_action': (
                    "1. Verificar posiciones abiertas en Binance\n"
                    "2. Cerrar manualmente si es necesario\n"
                    "3. Analizar la causa del drawdown\n"
                    "4. Eliminar STOP_TRADING.LOCK cuando esté listo"
                ),
            })
            
            # Also send a system alert for redundancy
            Notifier.send_system_alert(
                "KILL_SWITCH",
                f"Kill Switch activado: `{reason}`\nBalance: `${balance:,.2f}`",
                priority="CRITICAL"
            )
        except Exception as e:
            logger.error(f"Kill Switch notification failed: {e}")

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
