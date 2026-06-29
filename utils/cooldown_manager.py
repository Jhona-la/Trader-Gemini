"""
Cooldown Manager Centralizado
Evita duplicate signals y overtrading

Este módulo centraliza toda la lógica de cooldown que antes estaba
fragmentada en engine.py, risk_manager.py y statistical.py
"""
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
from collections import defaultdict
import threading


class CooldownManager:
    """
    Sistema centralizado de cooldowns para evitar overtrading.
    
    Niveles:
    1. GLOBAL: Tiempo mínimo entre cualquier operación
    2. SYMBOL: Tiempo mínimo entre trades del mismo símbolo
    3. PATTERN: Tiempo mínimo para mismo patrón en mismo símbolo
    4. STRATEGY: Tiempo mínimo por estrategia
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern para una única instancia global"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Cooldowns en segundos
        # BUG #55 FIX: Respect Config.COOLDOWN_PERIOD_SECONDS if available
        from config import Config
        # ================================================================
        # FORENSIC-RESCUE-PROTOCOL: COOLDOWNS RESTORED & HORIZON-AWARE
        # QUÉ: Tiempos mínimos entre trades por nivel y por horizonte.
        # POR QUÉ: Bypass anterior de 0.0s causaba overtrading extremo.
        #   Ahora separamos Scalping (60s) de Swing (1h) para evitar que
        #   un trade de scalping bloquee una señal legítima de swing.
        # PARA QUÉ: Coexistencia perfecta sin interferencias cruzadas.
        # ================================================================
        self.GLOBAL_COOLDOWN = 0.0
        self.SCALPING_SYMBOL_COOLDOWN = 60.0    # 1 min
        self.SWING_SYMBOL_COOLDOWN = 3600.0     # 1 hora
        self.SCALPING_PATTERN_COOLDOWN = 60.0
        self.SWING_PATTERN_COOLDOWN = 3600.0
        self.STRATEGY_COOLDOWN = 0.0  # FORENSIC FIX: was 60.0, blocking multi-symbol parallel execution
        
        # State tracking
        self.last_global_trade: Optional[datetime] = None
        self.last_symbol_trade: Dict[str, datetime] = {}
        self.last_pattern_trade: Dict[str, datetime] = {}
        self.last_strategy_trade: Dict[str, datetime] = {}
        
        # Virtual clock for backtests (Phase 99: Parity Fix)
        self._virtual_now: Optional[datetime] = None
        
        # Thread safety
        self._state_lock = threading.RLock()
        
        # Statistics
        self.blocked_count = defaultdict(int)
        
        self._initialized = True
    
    def set_virtual_time(self, time: datetime):
        """Set the virtual clock for backtesting (deterministico)"""
        with self._state_lock:
            self._virtual_now = time

    def _get_now(self) -> datetime:
        """Get current time (virtual or real)"""
        if self._virtual_now:
            return self._virtual_now
        return datetime.now(timezone.utc)

    
    def can_trade_global(self) -> bool:
        """Check global cooldown (fastest check)"""
        with self._state_lock:
            if self.last_global_trade is None:
                return True
            
            elapsed = (self._get_now() - self.last_global_trade).total_seconds()
            if elapsed < self.GLOBAL_COOLDOWN:
                self.blocked_count['global'] += 1
                return False
            return True

    
    def can_trade_symbol(self, symbol: str, horizon: str = "SCALPING", volatility_factor: float = 1.0, win_streak: int = 0) -> bool:
        """Check symbol-specific cooldown, segregated by horizon.
           FASE 9.5 DYNAMIC COOLDOWN: SCALPING cooldown is reduced in high volatility or winning streaks.
        """
        with self._state_lock:
            key = f"{symbol}_{horizon}"
            if key not in self.last_symbol_trade:
                return True
            
            elapsed = (self._get_now() - self.last_symbol_trade[key]).total_seconds()
            
            if horizon in ["SCALPING", "MICROSCALPING"]:
                # Dynamic cooldown calculation
                base_cd = self.SCALPING_SYMBOL_COOLDOWN
                # Max 60% reduction from volatility
                vol_mult = max(0.4, 1.0 - (volatility_factor * 0.2)) 
                # Max 40% reduction from win streak
                streak_mult = max(0.6, 1.0 - (win_streak * 0.1))
                cooldown_val = base_cd * vol_mult * streak_mult
                # Hard minimum of 15 seconds to prevent spam
                cooldown_val = max(15.0, cooldown_val)
            else:
                cooldown_val = self.SWING_SYMBOL_COOLDOWN

            if elapsed < cooldown_val:
                self.blocked_count[f'symbol_{key}'] += 1
                return False
            return True

    def can_trade_pattern(self, symbol: str, pattern: str, horizon: str = "SCALPING") -> bool:
        """Check pattern-specific cooldown, segregated by horizon."""
        with self._state_lock:
            key = f"{symbol}_{pattern}_{horizon}"
            if key not in self.last_pattern_trade:
                return True
            
            elapsed = (self._get_now() - self.last_pattern_trade[key]).total_seconds()
            cooldown_val = self.SCALPING_PATTERN_COOLDOWN if horizon in ["SCALPING", "MICROSCALPING"] else self.SWING_PATTERN_COOLDOWN
            if elapsed < cooldown_val:
                self.blocked_count[f'pattern_{key}'] += 1
                return False
            return True

    def can_trade_strategy(self, strategy_id: str) -> bool:
        """Check strategy-specific cooldown"""
        with self._state_lock:
            if strategy_id not in self.last_strategy_trade:
                return True
            
            elapsed = (self._get_now() - self.last_strategy_trade[strategy_id]).total_seconds()
            if elapsed < self.STRATEGY_COOLDOWN:
                self.blocked_count[f'strategy_{strategy_id}'] += 1
                return False
            return True

    def can_evaluate(self, strategy_id: str, symbol: str) -> bool:
        """Check if a strategy can be evaluated for a specific symbol."""
        with self._state_lock:
            if not self.can_trade_strategy(strategy_id):
                return False
            if not self.can_trade_symbol(symbol, horizon="SCALPING"):
                return False
            return True

    def can_trade(self, symbol: str, pattern: Optional[str] = None, 
                  strategy_id: Optional[str] = None, horizon: str = "SCALPING",
                  volatility_factor: float = 1.0, win_streak: int = 0) -> tuple:
        """
        Comprehensive check - all cooldowns.
        
        Returns:
            (can_trade: bool, reason: str)
        """
        # Level 1: Global
        if not self.can_trade_global():
            return False, f"Global cooldown ({self.GLOBAL_COOLDOWN}s)"
        
        # Level 2: Symbol (Horizon-Aware & Dynamic)
        if not self.can_trade_symbol(symbol, horizon=horizon, volatility_factor=volatility_factor, win_streak=win_streak):
            remaining = self.get_remaining_cooldown(symbol, 'symbol', horizon=horizon)
            return False, f"Symbol cooldown ({remaining:.0f}s remaining)"
        
        # Level 3: Pattern (Horizon-Aware)
        if pattern and not self.can_trade_pattern(symbol, pattern, horizon=horizon):
            remaining = self.get_remaining_cooldown(symbol, 'pattern', pattern=pattern, horizon=horizon)
            return False, f"Pattern cooldown ({remaining:.0f}s remaining)"
        
        # Level 4: Strategy (optional)
        if strategy_id and not self.can_trade_strategy(strategy_id):
            remaining = self.get_remaining_cooldown(strategy_id, 'strategy')
            return False, f"Strategy cooldown ({remaining:.0f}s remaining)"
            
        # Level 5: Loss Streak Custom Cooldown
        with self._state_lock:
            if hasattr(self, 'custom_cooldowns'):
                loss_key = f"loss_streak_{symbol}_{horizon}"
                if loss_key in self.custom_cooldowns:
                    now = self._get_now()
                    last_time = self.custom_cooldowns[loss_key]
                    elapsed = (now - last_time).total_seconds()
                    # RESCUE PROTOCOL: Check for config value instead of hardcoded 1800s
                    try:
                        from config import Config
                        cooldown_val = getattr(Config, 'COOLDOWN_PERIOD_SECONDS', 1800)
                    except ImportError:
                        cooldown_val = 1800
                    if elapsed < cooldown_val:
                        remaining = cooldown_val - elapsed
                        return False, f"Loss streak cooldown ({remaining:.0f}s remaining)"
        
        return True, "OK"
    
    def record_trade(self, symbol: str, pattern: Optional[str] = None, 
                     strategy_id: Optional[str] = None, horizon: str = "SCALPING"):
        """Record that a trade was executed (segregated by horizon)"""
        with self._state_lock:
            now = self._get_now()

            # Update all levels
            self.last_global_trade = now
            self.last_symbol_trade[f"{symbol}_{horizon}"] = now
            
            if pattern:
                key = f"{symbol}_{pattern}_{horizon}"
                self.last_pattern_trade[key] = now
            
            if strategy_id:
                self.last_strategy_trade[strategy_id] = now
    
    def get_remaining_cooldown(self, identifier: str, level: str, 
                               pattern: Optional[str] = None, horizon: str = "SCALPING") -> float:
        """Get remaining cooldown time in seconds"""
        with self._state_lock:
            now = self._get_now()
            
            if level == 'global':
                if self.last_global_trade is None:
                    return 0.0
                elapsed = (now - self.last_global_trade).total_seconds()
                return max(0, self.GLOBAL_COOLDOWN - elapsed)
            
            elif level == 'symbol':
                key = f"{identifier}_{horizon}"
                if key not in self.last_symbol_trade:
                    return 0.0
                elapsed = (now - self.last_symbol_trade[key]).total_seconds()
                cooldown_val = self.SCALPING_SYMBOL_COOLDOWN if horizon in ["SCALPING", "MICROSCALPING"] else self.SWING_SYMBOL_COOLDOWN
                return max(0, cooldown_val - elapsed)
            
            elif level == 'pattern':
                key = f"{identifier}_{pattern}_{horizon}"
                if key not in self.last_pattern_trade:
                    return 0.0
                elapsed = (now - self.last_pattern_trade[key]).total_seconds()
                cooldown_val = self.SCALPING_PATTERN_COOLDOWN if horizon in ["SCALPING", "MICROSCALPING"] else self.SWING_PATTERN_COOLDOWN
                return max(0, cooldown_val - elapsed)
            
            elif level == 'strategy':
                if identifier not in self.last_strategy_trade:
                    return 0.0
                elapsed = (now - self.last_strategy_trade[identifier]).total_seconds()
                return max(0, self.STRATEGY_COOLDOWN - elapsed)
            
            return 0.0

    
    def adjust_cooldown(self, level: str, new_value: int):
        """Dynamically adjust cooldown durations"""
        with self._state_lock:
            if level == 'global':
                self.GLOBAL_COOLDOWN = new_value
            elif level == 'symbol':
                self.SYMBOL_COOLDOWN = new_value
            elif level == 'pattern':
                self.PATTERN_COOLDOWN = new_value
            elif level == 'strategy':
                self.STRATEGY_COOLDOWN = new_value
    
    def adjust_for_regime(self, regime: str):
        """
        Ajustar cooldowns basado en régimen de mercado.
        RESCUE PROTOCOL: Ya no hacemos bypass a 1s.
        """
        with self._state_lock:
            # We don't bypass anymore. We keep the strict 300s cooldown.
            pass
    
    def get_statistics(self) -> dict:
        """Get cooldown blocking statistics"""
        with self._state_lock:
            total_blocked = sum(self.blocked_count.values())
            
            return {
                'total_blocked': total_blocked,
                'blocked_by_level': dict(self.blocked_count),
                'active_symbol_cooldowns': len(self.last_symbol_trade),
                'active_pattern_cooldowns': len(self.last_pattern_trade),
                'active_strategy_cooldowns': len(self.last_strategy_trade)
            }
    
    def clear_expired(self):
        """Clean up expired cooldowns (memory optimization)"""
        with self._state_lock:
            now = datetime.now(timezone.utc)
            
            # Clear expired symbol cooldowns
            expired_symbols = [
                sym for sym, last_time in self.last_symbol_trade.items()
                if (now - last_time).total_seconds() > self.SYMBOL_COOLDOWN * 2
            ]
            for sym in expired_symbols:
                del self.last_symbol_trade[sym]
            
            # Clear expired pattern cooldowns
            expired_patterns = [
                key for key, last_time in self.last_pattern_trade.items()
                if (now - last_time).total_seconds() > self.PATTERN_COOLDOWN * 2
            ]
            for key in expired_patterns:
                del self.last_pattern_trade[key]
            
            # Clear expired strategy cooldowns
            expired_strategies = [
                strat for strat, last_time in self.last_strategy_trade.items()
                if (now - last_time).total_seconds() > self.STRATEGY_COOLDOWN * 2
            ]
            for strat in expired_strategies:
                del self.last_strategy_trade[strat]
    
    def reset(self):
        """Reset all cooldown state (for testing or restart)"""
        with self._state_lock:
            self.last_global_trade = None
            self.last_symbol_trade.clear()
            self.last_pattern_trade.clear()
            self.last_strategy_trade.clear()
            self.blocked_count.clear()

    # Added to support custom cooldowns (e.g. data processing frequency)
    def check_custom_cooldown(self, key: str, duration_seconds: float) -> bool:
        """
        Check and update a custom cooldown key.
        Returns True if action is allowed (cooldown expired or new), False otherwise.
        Automatically updates the timestamp if allowed.
        """
        with self._state_lock:
            # We can use a separate dict or reuse one. Let's add a custom dict.
            if not hasattr(self, 'custom_cooldowns'):
                 self.custom_cooldowns = {}
            
            now = self._get_now()
            if key in self.custom_cooldowns:
                last_time = self.custom_cooldowns[key]
                elapsed = (now - last_time).total_seconds()
                if elapsed < duration_seconds:
                    return False
            
            self.custom_cooldowns[key] = now
            return True



# Singleton instance for global access
cooldown_manager = CooldownManager()
