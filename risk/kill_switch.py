from datetime import datetime, timezone, timedelta
from utils.notifier import Notifier
from utils.logger import logger

class KillSwitch:
    """
    Enhanced Kill Switch - Optimized for $12→$50 Growth Challenge
    """
    
    def __init__(self, 
                 growth_drawdown_pct=0.25,    # 25% en growth phase
                 standard_drawdown_pct=0.15,  # 15% en standard  
                 max_daily_losses=5,          # Más permisivo
                 max_api_errors=5,
                 recovery_threshold=0.02,     # Auto-recovery at 2% recovery
                 min_equity=12.50):           # Hard floor (Rule 3.3)
    
        self.is_active = False
        self.activation_reason = None
        self.activation_time = None
        
        # Dynamic thresholds based on phase
        self.growth_drawdown_pct = growth_drawdown_pct
        self.standard_drawdown_pct = standard_drawdown_pct
        self.max_daily_losses = max_daily_losses
        self.max_api_errors = max_api_errors
        self.recovery_threshold = recovery_threshold
        self.min_equity = min_equity
        
        # Tracking
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.daily_losses = 0
        self.last_reset_date = datetime.now(timezone.utc).date()
        self.consecutive_api_errors = 0
        
        # Recovery tracking
        self.activation_equity = 0.0

    def update_equity(self, current_equity: float):
        """Enhanced equity tracking with auto-daily reset"""
        self.current_equity = current_equity
        
        # Auto-reset daily losses at midnight
        self._auto_reset_daily_losses()
        
        if self.is_active:
            # Check for recovery
            recovery = current_equity - self.activation_equity
            recovery_pct = recovery / self.activation_equity if self.activation_equity > 0 else 0
            
            if recovery_pct >= self.recovery_threshold:
                self._deactivate(f"Auto-recovery: +{recovery_pct*100:.1f}% from activation")
            return

        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            
        # 1. HARD EQUITY FLOOR (Phase 6 Absolute Protection)
        if current_equity <= self.min_equity:
            self._activate(f"HARD EQUITY FLOOR: ${current_equity:.2f} <= ${self.min_equity:.2f}")
            return

        # 2. DRAWDOWN PROTECTION
        drawdown_pct = self._get_drawdown_threshold(current_equity)
        current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        
        if current_drawdown >= drawdown_pct:
            self._activate(f"DRAWDOWN: {current_drawdown*100:.1f}% (Threshold: {drawdown_pct*100}%)")

    def _get_drawdown_threshold(self, current_equity: float) -> float:
        """Dynamic drawdown based on account size"""
        if current_equity < 30:  # Growth phase
            return self.growth_drawdown_pct
        elif current_equity < 100:  # Consolidation
            return self.standard_drawdown_pct * 1.5  # 1.5x more tolerant
        else:  # Standard phase
            return self.standard_drawdown_pct

    def _auto_reset_daily_losses(self):
        """Auto-reset daily losses at midnight"""
        current_date = datetime.now(timezone.utc).date()
        if current_date != self.last_reset_date:
            self.daily_losses = 0
            self.last_reset_date = current_date
            print(f"🔄 Daily losses reset: {current_date}")

    def record_loss(self):
        """Record loss with phase-aware thresholds"""
        if self.is_active:
            return
            
        self.daily_losses += 1
        
        # More tolerant in growth phase
        current_threshold = self.max_daily_losses
        if self.current_equity < 30:  # Growth phase
            current_threshold = int(self.max_daily_losses * 1.5)  # 1.5x more losses allowed
        
        if self.daily_losses >= current_threshold:
            self._activate(f"DAILY LOSSES: {self.daily_losses} (Threshold: {current_threshold})")

    def record_api_error(self):
        """Record API error with exponential backoff consideration"""
        self.consecutive_api_errors += 1
        
        if self.consecutive_api_errors >= self.max_api_errors:
            self._activate(f"API ERRORS: {self.consecutive_api_errors} consecutive")

    def reset_api_errors(self):
        """Reset API errors on successful operation"""
        self.consecutive_api_errors = 0

    def reset_daily_losses(self):
        """Manual reset option"""
        self.daily_losses = 0
        self.last_reset_date = datetime.now(timezone.utc).date()

    def _activate(self, reason: str):
        """Activate kill switch"""
        self.is_active = True
        self.activation_reason = reason
        self.activation_time = datetime.now(timezone.utc)
        self.activation_equity = self.current_equity
        
        msg = f"💀 **KILL SWITCH ACTIVATED** 💀\n\n"
        msg += f"📉 **Reason**: `{reason}`\n"
        msg += f"💰 **Equity**: `${self.activation_equity:.2f}`\n"
        msg += f"🚫 Trading suspended until recovery."
        
        logger.critical(f"💀 KILL SWITCH ACTIVATED: {reason}")
        Notifier.send_telegram(msg, priority="CRITICAL")

    def _deactivate(self, reason: str):
        """Deactivate kill switch (auto-recovery)"""
        self.is_active = False
        self.activation_reason = None
        self.activation_time = None
        self.consecutive_api_errors = 0
        
        msg = f"✅ **KILL SWITCH DEACTIVATED** ✅\n\n"
        msg += f"📈 **Reason**: `{reason}`\n"
        msg += f"💰 **Current Equity**: `${self.current_equity:.2f}`\n"
        msg += f"🟢 Trading resumed."
        
        logger.info(f"✅ KILL SWITCH DEACTIVATED: {reason}")
        Notifier.send_telegram(msg, priority="INFO")

    def check_status(self) -> bool:
        """Check if trading is allowed"""
        return not self.is_active

    def get_status(self) -> dict:
        """Get detailed kill switch status"""
        return {
            'active': self.is_active,
            'reason': self.activation_reason,
            'activation_time': self.activation_time,
            'peak_equity': self.peak_equity,
            'current_equity': self.current_equity,
            'daily_losses': self.daily_losses,
            'consecutive_api_errors': self.consecutive_api_errors,
            'drawdown_threshold': self._get_drawdown_threshold(self.current_equity) * 100
        }
