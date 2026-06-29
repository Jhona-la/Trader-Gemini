"""
Telemetry Display - Real-Time Fleet Monitoring Console (Phase 99)
================================================================
QUÉ: Genera una tabla formateada para consola mostrando métricas por moneda.
POR QUÉ: Visibilidad instantánea del estado PnL/Gap de la flota completa.
PARA QUÉ: Permitir decisiones rápidas sobre cierres manuales o ajustes.
CÓMO: Lee Portfolio.positions y calcula Gap al TP, TTT estimado.
CUÁNDO: Se ejecuta cada 60s dentro de metrics_heartbeat_loop.
DÓNDE: utils/telemetry.py
QUIÉN: Invocado por main.py → metrics_heartbeat_loop.
"""

import time
from typing import Dict, Any, Optional
from utils.logger import logger


class TelemetryDisplay:
    """
    Renders a per-coin telemetry table.
    Thread-safe: Only reads from portfolio snapshot.
    """
    
    def __init__(self):
        # Track price velocity for TTT estimation
        self._price_history: Dict[str, list] = {}  # {symbol: [(timestamp, price), ...]}
        self._max_history = 10  # Keep last 10 samples for velocity calc
    
    def render(self, portfolio, data_provider=None) -> str:
        """
        Generates the telemetry table string.
        
        Args:
            portfolio: Portfolio instance (reads positions).
            data_provider: BinanceData instance (reads current price if needed).
        
        Returns:
            Formatted table string suitable for logger.info().
        """
        try:
            # Get atomic snapshot to avoid race conditions
            snapshot = portfolio.get_atomic_snapshot()
            positions = snapshot['positions']
            equity = snapshot['total_equity']
            cash = snapshot['cash']
            
            # Filter active positions only
            active = {}
            for symbol, pos in positions.items():
                qty = pos['quantity']
                if qty != 0:
                    active[symbol] = pos
            
            if not active:
                return self._render_idle(equity, cash, len(positions))
            
            return self._render_active(active, equity, cash)
            
        except Exception as e:
            logger.debug(f"Telemetry render error: {e}")
            return f"📡 Telemetry Error: {e}"
    
    def _render_idle(self, equity: float, cash: float, total_symbols: int) -> str:
        """Renders a compact status when no positions are open."""
        lines = [
            "",
            "📡 ═══ FLEET TELEMETRY ═══",
            f"   💰 Equity: ${equity:.2f} | Cash: ${cash:.2f}",
            f"   🔍 Monitoring {total_symbols} symbols | No active positions",
            "═" * 30,
        ]
        return "\n".join(lines)
    
    def _render_active(self, active: Dict[str, Any], equity: float, cash: float) -> str:
        """Renders the full telemetry table with active positions."""
        # Header
        lines = [
            "",
            "📡 ═══════════════════ FLEET TELEMETRY ═══════════════════",
            f"   💰 Equity: ${equity:.2f} | Cash: ${cash:.2f} | Open: {len(active)}",
            "┌──────────────┬──────────┬──────────┬──────────┬──────────┐",
            "│   Symbol     │  Side    │  PnL%    │  Gap TP% │ TTT Est. │",
            "├──────────────┼──────────┼──────────┼──────────┼──────────┤",
        ]
        
        for symbol, pos in sorted(active.items()):
            row = self._format_position_row(symbol, pos)
            lines.append(row)
        
        lines.append("└──────────────┴──────────┴──────────┴──────────┴──────────┘")
        
        return "\n".join(lines)
    
    def _format_position_row(self, symbol: str, pos: Dict[str, Any]) -> str:
        """Formats a single position row."""
        qty = pos['quantity']
        entry = pos['avg_price']
        current = pos['current_price']
        side = "LONG" if qty > 0 else "SHORT"
        
        # PnL%
        if entry > 0 and current > 0:
            if side == "LONG":
                pnl_pct = ((current - entry) / entry) * 100
            else:
                pnl_pct = ((entry - current) / entry) * 100
        else:
            pnl_pct = 0.0
        
        # Gap to TP
        tp_price = pos['tp_price']
        if tp_price > 0 and current > 0:
            if side == "LONG":
                gap_pct = ((tp_price - current) / current) * 100
            else:
                gap_pct = ((current - tp_price) / current) * 100
        else:
            gap_pct = None
        
        # TTT Estimation (Time-To-Target)
        ttt_str = self._estimate_ttt(symbol, current, tp_price, side)
        
        # Formatting
        sym_display = symbol[:12].ljust(12)
        side_display = side.ljust(8)
        
        # Color indicators via emoji
        if pnl_pct > 0:
            pnl_str = f"+{pnl_pct:.2f}%".rjust(8)
        else:
            pnl_str = f"{pnl_pct:.2f}%".rjust(8)
        
        gap_str = f"{gap_pct:.2f}%".rjust(8) if gap_pct is not None else "   —    "
        ttt_display = ttt_str.rjust(8)
        
        return f"│ {sym_display} │ {side_display} │ {pnl_str} │ {gap_str} │ {ttt_display} │"
    
    def _estimate_ttt(self, symbol: str, current_price: float, tp_price: float, side: str) -> str:
        """
        Estimates Time-To-Target based on recent price velocity.
        Uses a simple linear extrapolation from the last N price samples.
        """
        now = time.time()
        
        # Record current price
        if symbol not in self._price_history:
            self._price_history[symbol] = []
        
        history = self._price_history[symbol]
        history.append((now, current_price))
        
        # Trim to max_history
        if len(history) > self._max_history:
            self._price_history[symbol] = history[-self._max_history:]
            history = self._price_history[symbol]
        
        # Need at least 2 samples
        if len(history) < 2 or tp_price <= 0 or current_price <= 0:
            return "—"
        
        # Calculate velocity (price change per second)
        dt = history[-1][0] - history[0][0]
        if dt <= 0:
            return "—"
        
        dp = history[-1][1] - history[0][1]
        velocity = dp / dt  # $/sec
        
        # Estimate time to TP
        if side == "LONG":
            distance = tp_price - current_price
        else:
            distance = current_price - tp_price
        
        if distance <= 0:
            return "✅ HIT"
        
        if velocity <= 0:
            return "∞"
        
        seconds_remaining = distance / velocity
        
        if seconds_remaining > 3600:
            return f"{seconds_remaining/3600:.0f}h"
        elif seconds_remaining > 60:
            return f"{seconds_remaining/60:.0f}m"
        else:
            return f"{seconds_remaining:.0f}s"
    
    def clear_symbol(self, symbol: str):
        """Remove price history for a symbol (called on position close)."""
        self._price_history.pop(symbol, None)


# Module-level singleton for easy import
telemetry = TelemetryDisplay()
