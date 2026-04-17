"""
📢 ENHANCED NOTIFICATION ENGINE (Phase 4 → Phase 4.5)
======================================================

PROFESSOR METHOD:
- QUÉ: Centro de alertas multicanal enriquecido (Telegram/Email) con métricas
    detalladas, análisis de viabilidad y rate limiting.
- POR QUÉ: Las notificaciones básicas no proporcionan contexto suficiente para
    entender operaciones y estado del sistema en tiempo real.
- PARA QUÉ: Información accionable, detallada y visualmente atractiva que permite
    monitorear el sistema de trading y tomar decisiones informadas.
- CÓMO: ThreadPoolExecutor dedicado (no bloquea engine), rate limiting interno
    (30 msg/min), templates enriquecidos con emojis estratégicos.
- CUÁNDO: En cada trade open/close, alertas de riesgo, reportes diarios, updates.
- DÓNDE: utils/notifier.py — importado por portfolio.py, kill_switch.py,
    session_manager.py, system_monitor.py.
- QUIÉN: Clase Notifier (estática + singleton interno para ThreadPool/rate limit).
"""

import time
import threading
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from typing import Optional, Dict, Any, List

from config import Config
from utils.logger import logger


# ═══════════════════════════════════════════════════════════════════════════
# EMOJI MAPPING — Identificación Visual Rápida
# ═══════════════════════════════════════════════════════════════════════════
EMOJI_MAP = {
    # Directions
    "LONG": "🟢", "SHORT": "🔴", "BUY": "🟢", "SELL": "🔴",
    # Results
    "PROFIT": "💰", "LOSS": "💸", "BREAKEVEN": "🔶",
    # Alerts
    "WARNING": "⚠️", "CRITICAL": "🚨", "INFO": "ℹ️",
    "SUCCESS": "✅", "FAILURE": "❌",
    # Horizons
    "SCALPING": "⚡", "SWING": "📈",
    # System
    "KILL_SWITCH": "☠️", "CIRCUIT_BREAKER": "🛑",
    "HEARTBEAT": "💓", "SYSTEM": "🖥️",
    # Exit reasons
    "STOP_LOSS": "🛑", "TAKE_PROFIT": "💰", "TRAILING_STOP": "📉",
    "COGNITIVE_DECAY": "🧠", "TIMEOUT": "⏰", "MANUAL": "👤",
}


# ═══════════════════════════════════════════════════════════════════════════
# ENHANCED TRADE DATA — Cálculos Automáticos de Viabilidad
# ═══════════════════════════════════════════════════════════════════════════
class EnhancedTradeData:
    """
    PROFESSOR METHOD:
    - QUÉ: Wrapper que calcula automáticamente métricas enriquecidas de un trade.
    - POR QUÉ: Centralizar cálculos de fees, breakeven, viabilidad, R-multiple
        en un solo lugar para que todas las notificaciones sean consistentes.
    - PARA QUÉ: Proveer datos accionables (net PnL, viabilidad, eficiencia)
        en lugar de datos crudos.
    - CÓMO: Recibe dict con datos del trade, calcula derivados automáticamente.
    """

    # Fee rates (aligned with Config)
    FUTURES_FEE_RATE = 0.0006  # 0.06% (taker)
    SPOT_FEE_RATE = 0.001     # 0.10%

    def __init__(self, trade_info: Dict[str, Any]):
        # ── Basic Data ──
        self.symbol = trade_info.get('symbol', 'UNKNOWN')
        self.strategy = trade_info.get('strategy', 'Unknown')
        self.horizon = trade_info.get('horizon', 'SCALPING')
        self.direction = trade_info.get('direction', 'BUY')
        self.entry_price = float(trade_info.get('entry_price', 0.0))
        self.exit_price = float(trade_info.get('exit_price', 0.0))
        self.quantity = float(trade_info.get('quantity', 0.0))
        self.fill_price = float(trade_info.get('fill_price', 0.0))

        # ── Sizing ──
        price = self.fill_price or self.entry_price or self.exit_price
        self.size_usd = self.quantity * price if price > 0 else 0.0

        # ── Risk Params ──
        self.sl_price = float(trade_info.get('sl_price', 0.0))
        self.tp_price = float(trade_info.get('tp_price', 0.0))
        self.sl_pct = float(trade_info.get('sl_pct', 0.0))
        self.tp_pct = float(trade_info.get('tp_pct', 0.0))

        # ── PnL (pre-computed or calculated) ──
        self.pnl = float(trade_info.get('pnl', 0.0))
        self.commission = float(trade_info.get('commission', 0.0))

        # ── Estimate fees if not provided ──
        if self.commission == 0.0 and self.size_usd > 0:
            fee_rate = self.FUTURES_FEE_RATE if Config.BINANCE_USE_FUTURES else self.SPOT_FEE_RATE
            self.commission = self.size_usd * fee_rate * 2  # Entry + Exit

        # ── Net PnL ──
        self.net_pnl = self.pnl - self.commission
        self.net_pnl_pct = (self.net_pnl / self.size_usd * 100) if self.size_usd > 0 else 0.0

        # ── Viability Analysis ──
        self.breakeven_pct = self._calc_breakeven()
        self.min_viable_net = self._calc_min_viable()

        # ── Risk/Reward ──
        self.rr_ratio = self._calc_rr_ratio()

        # ── Management Metrics ──
        self.mfe_pct = float(trade_info.get('mfe_pct', 0.0))
        self.mae_pct = float(trade_info.get('mae_pct', 0.0))
        self.r_multiple = self._calc_r_multiple()
        self.duration = trade_info.get('duration', 'N/A')
        self.exit_reason = trade_info.get('exit_reason', 'Unknown')

        # ── Balance ──
        self.margin_used = float(trade_info.get('margin_used', 0.0))
        self.leverage = float(trade_info.get('leverage', 1.0))
        self.fee_tag = trade_info.get('fee_tag', 'Unknown')
        self.balance_before = float(trade_info.get('balance_before', 0.0))
        self.balance_after = float(trade_info.get('balance_after', 0.0))
        self.balance_change_pct = (
            ((self.balance_after - self.balance_before) / self.balance_before * 100)
            if self.balance_before > 0 else 0.0
        )
        
        # ── ML Telemetry ──
        self.ml_confidence = trade_info.get('ml_confidence', None)
        self.predicted_duration = trade_info.get('predicted_duration', None)

        # ── Market Context ──
        self.volatility = float(trade_info.get('volatility', 0.0))
        self.spread = float(trade_info.get('spread', 0.0))
        self.win_rate = float(trade_info.get('win_rate', 0.0))
        self.timestamp = trade_info.get('timestamp', datetime.now(timezone.utc).strftime('%H:%M:%S UTC'))

    def _calc_breakeven(self) -> float:
        """Breakeven % considering fees."""
        if self.size_usd <= 0:
            return 0.0
        return (self.commission / self.size_usd) * 100

    def _calc_min_viable(self) -> float:
        """Minimum viable net profit in USD."""
        return max(0.0, self.commission * 1.5)  # Need 1.5x fees to be meaningfully profitable

    def _calc_rr_ratio(self) -> float:
        """Risk/Reward ratio."""
        if self.sl_pct > 0 and self.tp_pct > 0:
            return self.tp_pct / self.sl_pct
        return 0.0

    def _calc_r_multiple(self) -> float:
        """R-multiple: PnL / Risk."""
        if self.sl_pct > 0 and self.entry_price > 0:
            risk_usd = self.size_usd * self.sl_pct
            return self.net_pnl / risk_usd if risk_usd > 0 else 0.0
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Export all computed fields as dict."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


# ═══════════════════════════════════════════════════════════════════════════
# RATE LIMITER — Protección Anti-Throttle Telegram
# ═══════════════════════════════════════════════════════════════════════════
class _RateLimiter:
    """
    Token bucket rate limiter for Telegram API.
    Max 30 messages per minute by default.
    """
    def __init__(self, max_per_minute: int = 30):
        self._max = max_per_minute
        self._timestamps: deque = deque(maxlen=max_per_minute)
        self._lock = threading.Lock()

    def allow(self) -> bool:
        now = time.monotonic()
        with self._lock:
            # Purge timestamps older than 60s
            while self._timestamps and (now - self._timestamps[0]) > 60.0:
                self._timestamps.popleft()
            if len(self._timestamps) < self._max:
                self._timestamps.append(now)
                return True
            return False

    @property
    def remaining(self) -> int:
        now = time.monotonic()
        with self._lock:
            while self._timestamps and (now - self._timestamps[0]) > 60.0:
                self._timestamps.popleft()
            return max(0, self._max - len(self._timestamps))


# ═══════════════════════════════════════════════════════════════════════════
# NOTIFIER — Motor de Notificaciones Enriquecido
# ═══════════════════════════════════════════════════════════════════════════
class Notifier:
    """
    📢 ENHANCED NOTIFICATION ENGINE (Phase 4.5)

    PROFESSOR METHOD:
    - QUÉ: Centro de alertas multicanal con métricas enriquecidas.
    - POR QUÉ: El trader no puede estar 24/7 pegado al monitor.
    - PARA QUÉ: Recibir avisos detallados de trades, riesgo y performance.
    - CÓMO: ThreadPoolExecutor (non-blocking) + Rate Limiter (anti-throttle).
    - CUÁNDO: En apertura/cierre de trades, alertas de riesgo, reportes.
    - DÓNDE: Importado por portfolio.py, kill_switch.py, session_manager.py.
    - QUIÉN: Clase Notifier con métodos estáticos.
    """

    # ── Singleton Infrastructure ──
    _executor: Optional[ThreadPoolExecutor] = None
    _rate_limiter = _RateLimiter(
        getattr(Config.Observability, 'NOTIFICATION_MAX_MESSAGES_PER_MIN', 30)
    )
    _lock = threading.Lock()

    @classmethod
    def _get_executor(cls) -> ThreadPoolExecutor:
        """Lazy-init ThreadPoolExecutor (max 2 workers for I/O)."""
        if cls._executor is None:
            with cls._lock:
                if cls._executor is None:
                    cls._executor = ThreadPoolExecutor(
                        max_workers=2,
                        thread_name_prefix="notifier"
                    )
        return cls._executor

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 0: LOW-LEVEL TRANSPORT (Backward Compatible)
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def send_telegram(message: str, priority: str = "INFO") -> None:
        """
        Envía alerta a Telegram (Rule 4.2) — BACKWARD COMPATIBLE.
        Now submits to ThreadPoolExecutor to avoid blocking the engine event loop.
        """
        if not Config.Observability.TELEGRAM_ENABLED:
            return

        # Submit to background thread
        Notifier._get_executor().submit(Notifier._do_send_telegram, message, priority)

    @staticmethod
    def _do_send_telegram(message: str, priority: str) -> None:
        """Actual Telegram send (runs in background thread)."""
        if not Notifier._rate_limiter.allow():
            logger.warning("📢 [Notifier] Rate limited — Telegram message dropped")
            return

        # Priority visual header
        header = "🤖 *TRADER GEMINI*"
        if priority == "CRITICAL":
            header = "🚨 *CRITICAL ALERT*"
        elif priority == "WARNING":
            header = "⚠️ *WARNING*"

        ts = datetime.now(timezone.utc).strftime('%H:%M:%S')
        full_message = f"{header}\n\n{message}\n\n🕒 {ts} UTC"

        # Telegram has 4096 char limit — truncate if needed
        if len(full_message) > 4000:
            full_message = full_message[:3990] + "\n\n⚠️ _(truncated)_"

        url = f"https://api.telegram.org/bot{Config.Observability.TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": Config.Observability.TELEGRAM_CHAT_ID,
            "text": full_message,
            "parse_mode": "Markdown"
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code != 200:
                logger.warning(f"Telegram failed: {response.text}")
        except Exception as e:
            logger.error(f"Error sending Telegram: {e}")

    @staticmethod
    def send_email(subject: str, body: str, is_html: bool = False) -> None:
        """Envía reporte o alerta por Email (Rule 4.2) — BACKWARD COMPATIBLE."""
        if not Config.Observability.EMAIL_ENABLED:
            return

        # Submit to background thread
        Notifier._get_executor().submit(Notifier._do_send_email, subject, body, is_html)

    @staticmethod
    def _do_send_email(subject: str, body: str, is_html: bool) -> None:
        """Actual Email send (runs in background thread)."""
        try:
            msg = MIMEMultipart()
            msg['From'] = Config.Observability.EMAIL_USER
            msg['To'] = Config.Observability.EMAIL_RECEIVER
            msg['Subject'] = f"Trader Gemini: {subject}"

            msg.attach(MIMEText(body, 'html' if is_html else 'plain'))

            with smtplib.SMTP(Config.Observability.SMTP_SERVER, Config.Observability.SMTP_PORT) as server:
                server.starttls()
                server.login(Config.Observability.EMAIL_USER, Config.Observability.EMAIL_PASS)
                server.send_message(msg)

            logger.info(f"📧 Email sent: {subject}")
        except Exception as e:
            logger.error(f"Error sending Email: {e}")

    @staticmethod
    def notify_trade(symbol, direction, price, qty, pnl=None, winrate=None):
        """
        Legacy trade notification — BACKWARD COMPATIBLE.
        Formats and sends basic trade notification.
        """
        type_str = "COMPRA (LONG)" if direction == "BUY" else "VENTA (SELL)"
        emoji = "🟢" if direction == "BUY" else "🔴"

        msg = f"{emoji} *Trade Executed*\n"
        msg += f"Symbol: `{symbol}`\n"
        msg += f"Action: {type_str}\n"
        msg += f"Price: `${price:,.4f}`\n"
        msg += f"Qty: `{qty}`"

        if pnl is not None:
            pnl_emoji = "💰" if pnl > 0 else "📉"
            msg += f"\n\n{pnl_emoji} *PnL Realized: ${pnl:,.2f}*"
            if winrate:
                msg += f"\n🏆 Win Rate: `{winrate:.1f}%`"

        Notifier.send_telegram(msg)

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 1: ENHANCED TRADE NOTIFICATIONS
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def send_trade_open(trade_data: Dict[str, Any]) -> None:
        """
        🎯 Enhanced Trade Open Notification.

        PROFESSOR METHOD:
        - QUÉ: Notificación enriquecida de apertura de trade con análisis de viabilidad.
        - POR QUÉ: Un trade abierto sin contexto de viabilidad (fees vs target) 
            es operar a ciegas.
        - PARA QUÉ: Saber inmediatamente si el trade tiene potencial de ser 
            rentable después de fees.
        - CÓMO: EnhancedTradeData calcula breakeven, fees estimados, net mínimo viable.
        - CUÁNDO: Cada vez que portfolio.py registra una apertura de posición.
        - DÓNDE: Enviado a Telegram (y email si habilitado).
        """
        if not getattr(Config.Observability, 'NOTIFICATION_TRADE_OPEN', True):
            return

        td = EnhancedTradeData(trade_data)
        dir_emoji = EMOJI_MAP.get(str(td.direction).upper(), "🔶")
        horizon_emoji = EMOJI_MAP.get(td.horizon, "📊")

        msg = f"🎯 *NUEVO TRADE INICIADO* 🎯\n\n"
        msg += f"*Estrategia:* {td.strategy} ({horizon_emoji} {td.horizon})\n"
        msg += f"*Par:* `{td.symbol}`\n"
        msg += f"*Dirección:* {dir_emoji} {td.direction}\n"
        msg += f"*Entrada:* `${td.fill_price:,.4f}`\n"
        msg += f"*Tamaño:* `{td.quantity}` (${td.size_usd:,.2f} USD)\n"

        if td.sl_pct > 0:
            msg += f"*Stop Loss:* `{td.sl_pct*100:,.2f}%`\n"
        if td.tp_pct > 0:
            msg += f"*Take Profit:* `{td.tp_pct*100:,.2f}%`\n"
        if td.rr_ratio > 0:
            msg += f"*Risk/Reward:* `1:{td.rr_ratio:,.2f}`\n"

        msg += f"\n*Análisis de Viabilidad:*\n"
        msg += f"⚠️ Fees estimados: `${td.commission:,.4f}`\n"
        msg += f"📊 Breakeven: `{td.breakeven_pct:,.3f}%`\n"
        msg += f"🎯 Neto mínimo viable: `${td.min_viable_net:,.4f}`\n"

        if td.volatility > 0:
            msg += f"\n*Condiciones de Mercado:*\n"
            msg += f"📈 Volatilidad: `{td.volatility:,.2f}%`\n"

        if td.spread > 0:
            msg += f"📊 Spread: `{td.spread:,.4f}%`\n"

        msg += f"\n🕒 `{td.timestamp}`"

        Notifier.send_telegram(msg)

        # Email (optional)
        if Config.Observability.EMAIL_ENABLED:
            try:
                from utils.email_templates import render_trade_open_email
                html = render_trade_open_email(td.to_dict())
                Notifier.send_email("Nuevo Trade Iniciado", html, is_html=True)
            except ImportError:
                Notifier.send_email("Nuevo Trade Iniciado", msg)

    @staticmethod
    def send_trade_close(trade_data: Dict[str, Any]) -> None:
        """
        📊 Enhanced Trade Close Notification.

        PROFESSOR METHOD:
        - QUÉ: Notificación de cierre con métricas completas de performance.
        - POR QUÉ: Sin post-mortem detallado de cada trade, no hay aprendizaje.
        - PARA QUÉ: Evaluar eficiencia real (R-multiple, MFE/MAE, net PnL).
        - CÓMO: EnhancedTradeData calcula PnL neto, R-multiple, balance delta.
        - CUÁNDO: Cierre de posición (SL/TP/Trailing/Manual).
        """
        if not getattr(Config.Observability, 'NOTIFICATION_TRADE_CLOSE', True):
            return

        td = EnhancedTradeData(trade_data)
        result_emoji = "🟢" if td.net_pnl > 0 else ("🔴" if td.net_pnl < 0 else "🔶")
        horizon_emoji = EMOJI_MAP.get(td.horizon, "📊")
        exit_emoji = EMOJI_MAP.get(td.exit_reason, "📋")

        msg = f"{result_emoji} *TRADE CERRADO* {result_emoji}\n\n"

        msg += f"*Resumen:*\n"
        msg += f"Estrategia: {td.strategy} ({horizon_emoji} {td.horizon})\n"
        msg += f"Par: `{td.symbol}`\n"
        msg += f"Dirección: {EMOJI_MAP.get(str(td.direction).upper(), '🔶')} {td.direction}\n"
        msg += f"Duración: `{td.duration}`\n"

        msg += f"\n*Resultados:*\n"
        msg += f"Entrada: `${td.entry_price:,.4f}`\n"
        msg += f"Salida: `${td.exit_price:,.4f}`\n"

        if td.entry_price > 0:
            price_change = ((td.exit_price - td.entry_price) / td.entry_price) * 100
            msg += f"Movimiento: `{price_change:+,.2f}%`\n"

        msg += f"\n*Métricas:*\n"
        msg += f"Nocional: `${td.size_usd:,.2f}` (`{td.leverage}x Lev`)\n"
        msg += f"Margen gastado: `${td.margin_used:,.2f}`\n"
        msg += f"PnL Bruto: `${td.pnl:,.4f}`\n"
        msg += f"Fees: `-${td.commission:,.4f}` (`{td.fee_tag}`)\n"
        pnl_sign = "+" if td.net_pnl >= 0 else ""
        msg += f"*PnL Neto: `{pnl_sign}${td.net_pnl:,.4f}`* ({td.net_pnl_pct:+,.2f}%)\n"

        if td.ml_confidence is not None:
            msg += f"\n*Predicción IA:*\n"
            msg += f"Confianza: `{td.ml_confidence*100:.1f}%`\n"
            if td.predicted_duration:
                msg += f"Horizonte Objetivo: `{td.predicted_duration} barras`\n"

        msg += f"\n*Gestión:*\n"
        msg += f"Razón: {exit_emoji} `{td.exit_reason}`\n"
        if td.mfe_pct != 0:
            msg += f"MFE: `{td.mfe_pct:,.2f}%` _(máx a favor)_\n"
        if td.mae_pct != 0:
            msg += f"MAE: `{td.mae_pct:,.2f}%` _(máx en contra)_\n"
        if td.r_multiple != 0:
            msg += f"R multiple: `{td.r_multiple:,.2f}`\n"

        if td.balance_before > 0:
            msg += f"\n*Balance:*\n"
            msg += f"Antes: `${td.balance_before:,.2f}`\n"
            msg += f"Después: `${td.balance_after:,.2f}`\n"
            msg += f"Cambio: `{td.balance_change_pct:+,.2f}%`\n"

        if td.win_rate > 0:
            msg += f"\n🏆 Win Rate: `{td.win_rate:.1f}%`"

        Notifier.send_telegram(msg)

        # Email (optional)
        if Config.Observability.EMAIL_ENABLED:
            try:
                from utils.email_templates import render_trade_close_email
                html = render_trade_close_email(td.to_dict())
                Notifier.send_email("Trade Cerrado", html, is_html=True)
            except ImportError:
                Notifier.send_email("Trade Cerrado", msg)

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 2: RISK ALERTS
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def send_risk_alert(alert_data: Dict[str, Any]) -> None:
        """
        🚨 Enhanced Risk Alert Notification.

        PROFESSOR METHOD:
        - QUÉ: Alerta de riesgo con niveles de urgencia y acción recomendada.
        - POR QUÉ: Drawdown, exposición excesiva y rachas perdedoras requieren 
            atención inmediata del trader.
        - PARA QUÉ: Prevenir pérdidas mayores con alertas tempranas.
        - CÓMO: Enviado con prioridad CRITICAL/WARNING según nivel.
        - CUÁNDO: Kill switch, drawdown threshold, loss streak, API errors.
        """
        if not getattr(Config.Observability, 'NOTIFICATION_RISK_ALERTS', True):
            return

        level = alert_data.get('level', 'warning').upper()
        urgency_emoji = "🚨" if level == "CRITICAL" else "⚠️"
        alert_type = alert_data.get('type', 'GENERAL')

        msg = f"{urgency_emoji} *ALERTA DE RIESGO* {urgency_emoji}\n\n"
        msg += f"*Tipo:* `{alert_type}`\n"
        msg += f"*Nivel:* `{level}`\n"
        msg += f"*Timestamp:* `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC`\n"

        detail_msg = alert_data.get('message', '')
        if detail_msg:
            msg += f"\n*Detalles:*\n{detail_msg}\n"

        # Metrics section
        drawdown = alert_data.get('drawdown', 0)
        exposure = alert_data.get('exposure', 0)
        balance = alert_data.get('balance', 0)

        if any([drawdown, exposure, balance]):
            msg += f"\n*Métricas Actuales:*\n"
            if drawdown:
                msg += f"Drawdown: `{drawdown:,.2f}%`\n"
            if exposure:
                msg += f"Exposición: `{exposure:,.2f}%`\n"
            if balance:
                msg += f"Balance: `${balance:,.2f}`\n"

        # Risk per trade
        rpt = alert_data.get('risk_per_trade', 0)
        if rpt:
            msg += f"Riesgo por trade: `{rpt:,.2f}%`\n"

        # Recommended action
        action = alert_data.get('recommended_action', '')
        if action:
            msg += f"\n*Acción Recomendada:*\n{action}\n"

        # System state
        open_pos = alert_data.get('open_positions', None)
        trades_today = alert_data.get('trades_today', None)
        win_rate = alert_data.get('win_rate', None)

        if any(v is not None for v in [open_pos, trades_today, win_rate]):
            msg += f"\n*Estado del Sistema:*\n"
            if open_pos is not None:
                msg += f"Posiciones abiertas: `{open_pos}`\n"
            if trades_today is not None:
                msg += f"Trades hoy: `{trades_today}`\n"
            if win_rate is not None:
                msg += f"Win rate: `{win_rate:.1f}%`\n"

        priority = "CRITICAL" if level == "CRITICAL" else "WARNING"
        Notifier.send_telegram(msg, priority=priority)

        # Always send critical alerts via email too
        if Config.Observability.EMAIL_ENABLED and level == "CRITICAL":
            Notifier.send_email(f"🚨 ALERTA CRÍTICA: {alert_type}", msg)

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 3: DAILY REPORTS
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def send_daily_report(report_data: Dict[str, Any]) -> None:
        """
        📊 Enhanced Daily Report.

        PROFESSOR METHOD:
        - QUÉ: Reporte diario con métricas de trading, riesgo y análisis por estrategia.
        - POR QUÉ: Sin resumen diario no hay feedback loop para el trader.
        - PARA QUÉ: Evaluar si el sistema está cumpliendo objetivos de rentabilidad.
        """
        if not getattr(Config.Observability, 'NOTIFICATION_DAILY_REPORT', True):
            return

        daily_pnl = report_data.get('daily_pnl', 0.0)
        daily_emoji = "📈" if daily_pnl > 0 else "📉"

        msg = f"{daily_emoji} *REPORTE DIARIO* {daily_emoji}\n\n"
        msg += f"*Resumen del Día:*\n"
        msg += f"Fecha: `{report_data.get('date', 'N/A')}`\n"
        msg += f"Balance inicial: `${report_data.get('start_balance', 0):,.2f}`\n"
        msg += f"Balance final: `${report_data.get('end_balance', 0):,.2f}`\n"
        pnl_sign = "+" if daily_pnl >= 0 else ""
        daily_pnl_pct = report_data.get('daily_pnl_pct', 0.0)
        msg += f"*PnL Diario: `{pnl_sign}${daily_pnl:,.2f}`* ({daily_pnl_pct:+,.2f}%)\n"

        # Trading metrics
        total_trades = report_data.get('total_trades', 0)
        winning = report_data.get('winning_trades', 0)
        losing = report_data.get('losing_trades', 0)
        win_rate = report_data.get('win_rate', 0.0)

        msg += f"\n*Métricas de Trading:*\n"
        msg += f"Total trades: `{total_trades}`\n"
        msg += f"Ganadores: `{winning}` ({win_rate:.1f}%)\n"
        msg += f"Perdedores: `{losing}`\n"

        wl_ratio = report_data.get('win_loss_ratio', 0.0)
        if wl_ratio > 0:
            msg += f"Ratio Win/Loss: `{wl_ratio:,.2f}`\n"

        expectancy = report_data.get('expectancy', 0.0)
        if expectancy != 0:
            msg += f"Expectancia: `${expectancy:,.4f}`\n"

        # Per-strategy breakdown
        strategies = report_data.get('strategies', [])
        if strategies:
            msg += f"\n*Análisis por Estrategia:*\n"
            for strat in strategies:
                name = strat.get('name', 'Unknown')
                msg += f"\n`{name}:`\n"
                msg += f"  Trades: `{strat.get('trades', 0)}`\n"
                msg += f"  Win Rate: `{strat.get('win_rate', 0):.1f}%`\n"
                msg += f"  PnL: `${strat.get('pnl', 0):,.2f}`\n"

        # Risk metrics
        msg += f"\n*Análisis de Riesgo:*\n"
        msg += f"Max Drawdown: `{report_data.get('max_drawdown', 0):,.2f}%`\n"
        msg += f"Max Exposición: `{report_data.get('max_exposure', 0):,.2f}%`\n"

        sharpe = report_data.get('sharpe_ratio', 0.0)
        sortino = report_data.get('sortino_ratio', 0.0)
        if sharpe != 0:
            msg += f"Sharpe Ratio: `{sharpe:,.2f}`\n"
        if sortino != 0:
            msg += f"Sortino Ratio: `{sortino:,.2f}`\n"

        Notifier.send_telegram(msg)

        # Send full daily report via email
        if Config.Observability.EMAIL_ENABLED:
            try:
                from utils.email_templates import render_daily_report_email
                html = render_daily_report_email(report_data)
                Notifier.send_email("Reporte Diario", html, is_html=True)
            except ImportError:
                Notifier.send_email("Reporte Diario", msg)

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 4: PERFORMANCE UPDATES
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def send_performance_update(update_data: Dict[str, Any]) -> None:
        """
        📊 Periodic Performance Update.

        PROFESSOR METHOD:
        - QUÉ: Update periódico de performance y estado del mercado.
        - POR QUÉ: Monitoreo pasivo sin necesidad de abrir el dashboard.
        - PARA QUÉ: Detectar anomalías y confirmar operación normal.
        """
        if not getattr(Config.Observability, 'NOTIFICATION_PERFORMANCE_UPDATE', True):
            return

        msg = f"📊 *UPDATE DE PERFORMANCE* 📊\n\n"

        msg += f"*Tiempo real:*\n"
        msg += f"Balance: `${update_data.get('balance', 0):,.2f}`\n"
        daily_pnl = update_data.get('daily_pnl', 0.0)
        daily_pnl_pct = update_data.get('daily_pnl_pct', 0.0)
        msg += f"PnL hoy: `${daily_pnl:+,.2f}` ({daily_pnl_pct:+,.2f}%)\n"
        msg += f"Drawdown: `{update_data.get('drawdown', 0):,.2f}%`\n"
        msg += f"Exposición: `{update_data.get('exposure', 0):,.2f}%`\n"

        msg += f"\n*Métricas de Trading:*\n"
        msg += f"Trades hoy: `{update_data.get('trades_today', 0)}`\n"
        msg += f"Win Rate: `{update_data.get('win_rate', 0):.1f}%`\n"

        expectancy = update_data.get('expectancy', 0.0)
        if expectancy != 0:
            msg += f"Expectancia: `${expectancy:,.4f}`\n"

        # Market conditions
        volatility = update_data.get('avg_volatility', 0)
        condition = update_data.get('market_condition', '')
        if volatility or condition:
            msg += f"\n*Análisis de Mercado:*\n"
            if volatility:
                msg += f"Volatilidad: `{volatility:,.2f}%`\n"
            if condition:
                msg += f"Condición: `{condition}`\n"

        # Active symbols
        active = update_data.get('active_symbols', [])
        if active:
            symbols_str = " ".join(f"`{s}`" for s in active[:10])
            msg += f"\n*Posiciones activas:*\n{symbols_str}\n"

        Notifier.send_telegram(msg)

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 5: SYSTEM ALERTS
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def send_system_alert(alert_type: str, message: str,
                          priority: str = "WARNING") -> None:
        """
        🖥️ System Alert Notification.

        For kill_switch activations, circuit breakers, API failures, etc.
        """
        type_emoji = EMOJI_MAP.get(alert_type, "🖥️")
        msg = f"{type_emoji} *ALERTA DE SISTEMA* {type_emoji}\n\n"
        msg += f"*Tipo:* `{alert_type}`\n"
        msg += f"*Timestamp:* `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC`\n\n"
        msg += message

        Notifier.send_telegram(msg, priority=priority)

        # Always email critical system alerts
        if Config.Observability.EMAIL_ENABLED and priority == "CRITICAL":
            Notifier.send_email(f"🚨 Sistema: {alert_type}", msg)

    # ══════════════════════════════════════════════════════════════════════
    # UTILITIES
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def shutdown():
        """Graceful shutdown of the ThreadPoolExecutor."""
        if Notifier._executor:
            try:
                Notifier._executor.shutdown(wait=False)
                logger.info("📢 Notifier: ThreadPool shutdown complete.")
            except Exception:
                pass

    @staticmethod
    def get_rate_limiter_status() -> Dict[str, int]:
        """Returns rate limiter status for monitoring."""
        return {
            "remaining_messages": Notifier._rate_limiter.remaining,
            "max_per_minute": Notifier._rate_limiter._max,
        }
