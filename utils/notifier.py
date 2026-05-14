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
from utils.fast_json import FastJson


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
        self.trade_id = trade_info.get('trade_id', 'UNKNOWN')
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
        
        # ── ML Telemetry & Forensic Data ──
        self.ml_confidence = trade_info.get('ml_confidence', None)
        self.predicted_duration = trade_info.get('predicted_duration', None)
        self.predicted_magnitude = trade_info.get('predicted_magnitude', None)
        
        # Parse metadata if exists
        self.metadata = trade_info.get('metadata', {})
        self.order_type = self.metadata.get('enriched_order_type', trade_info.get('order_type', 'UNKNOWN'))
        self.setup_type = self.metadata.get('setup_type', trade_info.get('setup_type', 'UNKNOWN'))
        self.neural_bias = self.metadata.get('neural_bias', None)
        self.rsi = self.metadata.get('rsi', None)
        self.adx = self.metadata.get('adx', None)
        self.confluence = self.metadata.get('multi_timeframe_score', None)
        self.raw_ml_confidence = self.metadata.get('raw_ml_confidence', None)
        self.smoothed_ml_confidence = self.metadata.get('smoothed_ml_confidence', None)
        
        # Phase & Concept from Sophia (Unified Oracle)
        self.concept = self.metadata.get('concept', None)
        self.phase = self.metadata.get('phase', None)

        # ── Market Context ──
        self.volatility = float(trade_info.get('volatility', 0.0))
        self.spread = float(trade_info.get('spread', 0.0))
        self.win_rate = float(trade_info.get('win_rate', 0.0))
        self.alltime_win_rate = float(trade_info.get('alltime_win_rate', 0.0))
        self.session_wins = int(trade_info.get('session_wins', 0))
        self.session_losses = int(trade_info.get('session_losses', 0))
        self.timestamp = trade_info.get('timestamp', datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'))

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
    def __init__(self, max_per_minute: int = 999999):
        self._max = max_per_minute
        self._timestamps: deque = deque(maxlen=max_per_minute)
        self._lock = threading.Lock()
        self._pause_until = 0.0

    def allow(self) -> bool:
        now = time.monotonic()
        with self._lock:
            if now < self._pause_until:
                return False
            # Purge timestamps older than 60s
            while self._timestamps and (now - self._timestamps[0]) > 60.0:
                self._timestamps.popleft()
            if len(self._timestamps) < self._max:
                self._timestamps.append(now)
                return True
            return False

    def pause(self, seconds: float):
        """Pause rate limiter completely for X seconds (e.g. after 429)."""
        now = time.monotonic()
        with self._lock:
            self._pause_until = max(self._pause_until, now + seconds)

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
        # ── FORENSIC-V42: TRANSPARENT BLACK BOX SINK ──
        # QUÉ: Guarda TODO el spam localmente antes del Rate Limiter.
        # POR QUÉ: El usuario requiere visibilidad absoluta para auditar.
        try:
            print(f"\n📢 [SPAM-{priority}]\n{message}\n")
            import os, json
            os.makedirs("dashboard/data", exist_ok=True)
            with open("dashboard/data/backtest_telemetry_spam.jsonl", "a", encoding="utf-8") as f:
                log_entry = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "priority": priority,
                    "message": message
                }
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

        # ═══════════════════════════════════════════════════════════════
        # CTOS OMNISCIENCE: BATCHED SPAM (Evade Rate Limits)
        # QUÉ: Si superamos el Rate Limit, no dropeamos, lo metemos al batch.
        # POR QUÉ: El usuario exige visibilidad absoluta de TODO.
        # PARA QUÉ: Evadir el 429 concatenando en bloques gigantes.
        # ═══════════════════════════════════════════════════════════════
        if not Notifier._rate_limiter.allow():
            logger.warning("📢 [Notifier] Rate limited — Queueing message for batching")
            with Notifier._lock:
                if not hasattr(Notifier, '_backtest_trade_batch'):
                    Notifier._backtest_trade_batch = []
                Notifier._backtest_trade_batch.append(f"[{priority}] {message}")
                
                # Auto-flush if batch gets too big (Telegram limit is 4096 chars, 
                # so let's flush every 5 messages or ~2500 chars)
                current_len = sum(len(m) for m in Notifier._backtest_trade_batch)
                if len(Notifier._backtest_trade_batch) >= 5 or current_len > 3000:
                    batch_msg = "==== 📊 SPAM BATCH ====\n\n" + "\n---\n".join(Notifier._backtest_trade_batch)
                    Notifier._backtest_trade_batch.clear()
                    # Resubmit the batch as a single message bypassing the rate limiter temporarily
                    # but sleeping to prevent an instant 429
                    time.sleep(1.0)
                    Notifier._get_executor().submit(Notifier._do_send_telegram_bypass, batch_msg, priority)
            return

        Notifier._do_send_telegram_bypass(message, priority)

    @staticmethod
    def _do_send_telegram_bypass(message: str, priority: str) -> None:
        """Sends the message directly without checking the rate limiter."""
        # Priority visual header
        header = "🤖 <b>TRADER GEMINI</b>"
        if priority == "CRITICAL":
            header = "🚨 <b>CRITICAL ALERT</b>"
        elif priority == "WARNING":
            header = "⚠️ <b>WARNING</b>"

        ts = datetime.now(timezone.utc).strftime('%H:%M:%S')
        full_message = f"{header}\n\n{message}\n\n🕒 {ts} UTC"

        # ── FORENSIC FIX: TELEGRAM FORMATTING CRASH ──
        # Telegram Markdown throws "Can't parse entities" if unescaped `_` (e.g. strategy_id) exists.
        # HTML mode safely ignores raw `_` and `[]` characters.
        import re
        html_message = re.sub(r'\*(.*?)\*', r'<b>\1</b>', full_message) # Bold
        html_message = re.sub(r'`(.*?)`', r'<code>\1</code>', html_message) # Monospace

        # Telegram has a 4096 char limit — truncate if needed
        if len(html_message) > 4000:
            html_message = html_message[:3990] + "\n\n⚠️ <i>(truncated)</i>"

        url = f"https://api.telegram.org/bot{Config.Observability.TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": Config.Observability.TELEGRAM_CHAT_ID,
            "text": html_message,
            "parse_mode": "HTML"
        }

        try:
            serialized_payload = FastJson.dumps(payload).encode('utf-8')
            response = requests.post(
                url, 
                data=serialized_payload, 
                headers={'Content-Type': 'application/json'}, 
                timeout=10
            )
            if response.status_code == 429:
                try:
                    data = response.json()
                    retry_after = data.get("parameters", {}).get("retry_after", 30)
                    logger.warning(f"Telegram API 429 Too Many Requests. Pausing notifications for {retry_after}s.")
                    Notifier._rate_limiter.pause(retry_after)
                except Exception:
                    logger.warning("Telegram API 429 Too Many Requests. Pausing notifications for 30s.")
                    Notifier._rate_limiter.pause(30)
            elif response.status_code != 200:
                logger.warning(f"Telegram failed: {response.text}")
        except Exception as e:
            logger.error(f"Error sending Telegram: {e}")

    @staticmethod
    def flush_backtest_trades() -> None:
        """Flushes any remaining batched trades during a backtest."""
        with Notifier._lock:
            if hasattr(Notifier, '_backtest_trade_batch') and len(Notifier._backtest_trade_batch) > 0:
                batch_msg = "==== 📊 BATCH DE TRADES FINAL (BACKTEST) ====\n\n" + "\n-------------------\n".join(Notifier._backtest_trade_batch)
                Notifier._backtest_trade_batch.clear()
                Notifier._do_send_telegram(batch_msg, "CRITICAL")


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
        dir_str = str(td.direction).upper()
        dir_label = "📈 LONG" if dir_str in ("BUY", "LONG") else "📉 SHORT" if dir_str in ("SELL", "SHORT") else f"🔶 {dir_str}"
        horizon_emoji = "⚡" if td.horizon == "SCALPING" else "🌊" if td.horizon == "SWING" else "📊"

        # Safe parsing for visual formatting
        visual_tid = td.trade_id if td.trade_id and td.trade_id not in ("None", "UNKNOWN") else "Pendiente_Asignación"
        visual_setup = td.setup_type if td.setup_type and td.setup_type not in ("None", "UNKNOWN") else "AUTOMATICO"

        msg = f"🎯 *NUEVO TRADE INICIADO* 🎯\n"
        msg += f"ID: `{visual_tid}`\n"
        if td.metadata.get('thought_id'):
            msg += f"Cortex ID: `{td.metadata.get('thought_id')}`\n\n"
        else:
            msg += "\n"
        msg += f"*Estrategia:* {td.strategy} ({horizon_emoji} {td.horizon})\n"
        
        # Avoid printing -100% win rate for uninitialized / open signals
        if td.win_rate >= 0:
            msg += f"*Win Rate Estrategia:* `{td.win_rate*100:.1f}%` (Sesión: {td.session_wins}W/{td.session_losses}L)\n"
            
        msg += f"*Par:* `{td.symbol}`\n"
        msg += f"*Setup:* `{visual_setup}`\n"
        msg += f"*Dirección:* {dir_label}\n"
        msg += f"*Tipo Orden:* `{td.order_type}`\n"
        msg += f"*Entrada:* `${td.fill_price:,.4f}`\n"
        msg += f"*Tamaño:* `{td.quantity}` (${td.size_usd:,.2f} USD)\n"

        if td.concept or td.phase:
            msg += f"\n*🧠 Decisión Oráculo (Sophia):*\n"
            if td.phase: msg += f"Fase Mercado: `{td.phase}`\n"
            if td.concept: msg += f"Concepto: {td.concept}\n"

        if td.sl_pct > 0 or td.tp_pct > 0:
            msg += f"\n*Niveles (Precios):*\n"
            direction_mult = 1 if str(td.direction).upper() == "LONG" else -1
            
            if td.tp_pct > 0:
                tp_price = td.fill_price * (1 + (td.tp_pct * direction_mult))
                expected_profit = td.size_usd * td.tp_pct
                msg += f"🎯 *Target Limit (TP):* `${tp_price:,.4f}` (`+{td.tp_pct*100:,.2f}%`)\n"
                msg += f"   💰 *Crecimiento Esperado:* `${expected_profit:,.2f}`\n"
                
            if td.sl_pct > 0:
                sl_price = td.fill_price * (1 - (td.sl_pct * direction_mult))
                msg += f"🛡️ *Stop Loss:* `${sl_price:,.4f}` (`-{td.sl_pct*100:,.2f}%`)\n"
                
        if td.rr_ratio > 0:
            msg += f"*Risk/Reward:* `1:{td.rr_ratio:,.2f}`\n"

        msg += f"\n*Decisión Forense:*\n"
        if td.confluence is not None:
            msg += f"🎯 Confluencia: `{td.confluence:.2f}`\n"
        if td.neural_bias is not None:
            msg += f"🧠 Neural Bias: `{td.neural_bias:.2f}`\n"
        
        sophia_prob = td.metadata.get('sophia_prob', None)
        if sophia_prob is not None:
            msg += f"🔮 Sophia Prob: `{sophia_prob*100:.1f}%`\n"
            
        if td.raw_ml_confidence is not None:
            msg += f"🤖 ML Raw: `{td.raw_ml_confidence*100:.1f}%` | Smoothed: `{td.smoothed_ml_confidence*100:.1f}%`\n"
        if td.rsi is not None and td.adx is not None:
            msg += f"📉 RSI: `{td.rsi:.1f}` | ADX: `{td.adx:.1f}`\n"

        if td.ml_confidence is not None or td.predicted_magnitude is not None:
            msg += f"\n*Predicción Cuantitativa IA:*\n"
            if td.predicted_magnitude:
                msg += f"📏 Magnitud Proyectada: `+{td.predicted_magnitude*100:.2f}%`\n"
            if td.predicted_duration:
                msg += f"⏱️ Tiempo Estimado: `{td.predicted_duration} barras`\n"

        msg += f"\n*Análisis de Viabilidad:*\n"
        msg += f"⚠️ Fees estimados: `${td.commission:,.4f}`\n"
        msg += f"📊 Breakeven: `{td.breakeven_pct:,.3f}%`\n"
        msg += f"🎯 Neto mínimo viable: `${td.min_viable_net:,.4f}`\n"
        
        msg += f"\n*Features Activos:*\n"
        if td.sl_pct >= 0.0075:
            msg += f"🛡️ Defensa: `Anti-Barrido NY` (SL Holgado)\n"
        if td.confluence is not None and td.confluence > 0:
            msg += f"🌌 Engine: `Sophia Quantum Veto Activo`\n"

        if td.volatility > 0:
            msg += f"\n*Condiciones de Mercado:*\n"
            msg += f"📈 Volatilidad: `{td.volatility:,.2f}%`\n"

        if td.spread > 0:
            msg += f"📊 Spread: `{td.spread:,.4f}%`\n"

        # ═══════════════════════════════════════════════════════════════
        # CTOS PHASE 3: PREDICTION DETAILS & SIZE TRACKING
        # ═══════════════════════════════════════════════════════════════
        _p_audit = trade_data.get('prediction_audit', {})
        _p_mag = _p_audit.get('predicted_magnitude') or trade_data.get('predicted_magnitude')
        _p_dur = _p_audit.get('predicted_duration_bars') or trade_data.get('predicted_duration')
        _p_target = _p_audit.get('predicted_target_price')
        _p_conf = _p_audit.get('confidence') or td.ml_confidence

        if _p_mag or _p_dur or _p_target:
            msg += f"\n📏 *Predicción de Estrategia:*\n"
            if _p_mag:
                msg += f"   Magnitud: `+{float(_p_mag)*100:.2f}%`"
                if _p_target:
                    msg += f" → `${float(_p_target):,.2f}`"
                msg += "\n"
            if _p_dur:
                msg += f"   Tiempo: `~{_p_dur} barras`\n"
            if _p_conf:
                msg += f"   Confianza: `{float(_p_conf)*100:.1f}%`\n"

        _open_size = trade_data.get('open_size_usd', 0.0)
        if _open_size > 0:
            msg += f"\n📦 *Tamaño de Apertura:*\n"
            msg += f"   Qty: `{td.quantity}` (`${_open_size:,.2f}` USD)\n"
            _margin = trade_data.get('margin_used', 0.0)
            if _margin > 0:
                msg += f"   Margen: `${_margin:,.2f}` (`{td.leverage}x` Lev)\n"

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
        horizon_emoji = "⚡" if td.horizon == "SCALPING" else "🌊" if td.horizon == "SWING" else "📊"
        exit_emoji = EMOJI_MAP.get(td.exit_reason, "📋")
        
        # Direction with clearer labeling
        dir_str = str(td.direction).upper()
        dir_label = "📈 LONG" if dir_str in ("BUY", "LONG") else "📉 SHORT" if dir_str in ("SELL", "SHORT") else f"🔶 {dir_str}"

        msg = f"{result_emoji} *TRADE CERRADO* {result_emoji}\n"
        msg += f"ID: `{td.trade_id}`\n"
        thought_id = trade_data.get('thought_id', td.metadata.get('thought_id', 'N/A'))
        if thought_id != 'N/A':
            msg += f"Cortex ID: `{thought_id}`\n\n"
        else:
            msg += "\n"

        if td.exit_reason == "TIME_STOP_ZOMBIE":
            msg += f"🧟 *ZOMBIE CATCHER TRIGGERED*\n"
            msg += f"_{td.duration} de inmovilización en mercado sin tendencia_\n\n"
        elif td.exit_reason == "TURBO_BE":
            msg += f"⚡ *TURBO-BREAKEVEN PROTEGIDO*\n"
            msg += f"_Peak PnL alcanzado y retrocedido — capital protegido_\n\n"
        elif td.exit_reason == "FLIP_EXIT":
            msg += f"🔄 *FLIP EXIT — Cambio de dirección detectado*\n\n"
        elif td.exit_reason == "HARD_SL":
            msg += f"🛑 *HARD STOP LOSS — Pérdida cortada*\n\n"

        msg += f"*Resumen:*\n"
        msg += f"Estrategia: {td.strategy} ({horizon_emoji} {td.horizon})\n"
        msg += f"Par: `{td.symbol}`\n"
        msg += f"Razón de Cierre: {exit_emoji} `{td.exit_reason}`\n"
        msg += f"Setup: `{td.setup_type}`\n"
        msg += f"Dirección: {dir_label}\n"
        msg += f"Tipo Orden: `{td.order_type}`\n"
        msg += f"Duración: `{td.duration}`\n"
        
        # Market Regime Context
        market_regime = td.metadata.get('market_regime', trade_data.get('market_regime', None))
        if market_regime:
            msg += f"Régimen: `{market_regime}`\n"
        
        # Peak PnL for exit context (especially useful for TURBO_BE and trailing)
        peak_pnl = td.metadata.get('peak_pnl_pct', trade_data.get('peak_pnl_pct', None))
        if peak_pnl is not None:
            msg += f"Peak PnL: `+{peak_pnl:.2f}%`\n"

        msg += f"\n*Resultados (Precios):*\n"
        msg += f"Entrada: `${td.entry_price:,.4f}`\n"
        msg += f"Salida: `${td.exit_price:,.4f}`\n"

        if td.entry_price > 0:
            price_change = ((td.exit_price - td.entry_price) / td.entry_price) * 100
            msg += f"Movimiento: `{price_change:+,.2f}%`\n"

        msg += f"\n💰 *TRADE PNL (Aislado):*\n"
        msg += f"Nocional: `${td.size_usd:,.2f}` (`{td.leverage}x Lev`)\n"
        msg += f"Margen gastado: `${td.margin_used:,.2f}`\n"
        msg += f"PnL Bruto: `${td.pnl:,.4f}`\n"
        msg += f"Fees: `-${td.commission:,.4f}` (`{td.fee_tag}`)\n"
        pnl_sign = "+" if td.net_pnl >= 0 else ""
        msg += f"*PnL Neto: `{pnl_sign}${td.net_pnl:,.4f}`* ({td.net_pnl_pct:+,.2f}%)\n"

        if td.confluence is not None or td.neural_bias is not None or td.raw_ml_confidence is not None:
            msg += f"\n*Decisión Forense:*\n"
            if td.confluence is not None:
                msg += f"🎯 Confluencia: `{td.confluence:.2f}`\n"
            if td.neural_bias is not None:
                msg += f"🧠 Neural Bias: `{td.neural_bias:.2f}`\n"
            if td.raw_ml_confidence is not None:
                msg += f"🤖 ML Raw: `{td.raw_ml_confidence*100:.1f}%` | Smoothed: `{td.smoothed_ml_confidence*100:.1f}%`\n"
            if td.rsi is not None and td.adx is not None:
                msg += f"📉 RSI: `{td.rsi:.1f}` | ADX: `{td.adx:.1f}`\n"

        if td.ml_confidence is not None or td.predicted_magnitude is not None:
            msg += f"\n*Auditoría de Predicción IA:*\n"
            if td.ml_confidence is not None:
                msg += f"🧠 Confianza Inicial: `{td.ml_confidence*100:.1f}%`\n"
            if td.predicted_magnitude:
                msg += f"🎯 Proyectado: `+{td.predicted_magnitude*100:.2f}%` | 📊 Realidad (MFE): `+{td.mfe_pct:.2f}%`\n"
            if td.predicted_duration:
                msg += f"⏱️ Estimado: `{td.predicted_duration} barras` | ⏳ Real: `{td.duration}`\n"

        msg += f"\n*Gestión:*\n"
        msg += f"Razón: {exit_emoji} `{td.exit_reason}`\n"
        if td.mfe_pct != 0:
            msg += f"MFE: `{td.mfe_pct:,.2f}%` _(máx a favor)_\n"
        if td.mae_pct != 0:
            msg += f"MAE: `{td.mae_pct:,.2f}%` _(máx en contra)_\n"
        if td.r_multiple != 0:
            msg += f"R multiple: `{td.r_multiple:,.2f}`\n"

        if td.balance_before > 0:
            balance_change = td.balance_after - td.balance_before
            balance_change_pct = (balance_change / td.balance_before) * 100 if td.balance_before > 0 else 0.0
            
            msg += f"\n🏦 *CTOS OMNISCIENT BALANCE:*\n"
            msg += f"├ Antes de iniciar la sesión: `${trade_data.get('session_start_equity', 0):,.4f}`\n"
            msg += f"├ Valor antes del trade: `${td.balance_before:,.4f}`\n"
            msg += f"├ Crecimiento neto de aporte: `${balance_change:+,.4f}` (`{balance_change_pct:+,.2f}%`)\n"
            msg += f"├ Crecimiento acumulado sesión: `${trade_data.get('session_net_pnl', 0):+,.4f}`\n"
            msg += f"└ Balance Total Actual: `${td.balance_after:,.4f}`\n"

        # ═══════════════════════════════════════════════════════════════
        # CTOS PHASE 5: EXPONENTIAL COMPOUNDING ROADMAP
        # ═══════════════════════════════════════════════════════════════
        roadmap = trade_data.get('growth_roadmap')
        if roadmap:
            msg += f"\n🚀 *ROADMAP CRECIMIENTO EXPONENCIAL (100% en 15 días):*\n"
            msg += f"Meta Diaria: `+${roadmap.get('daily_target_usd', 0.0):.4f}` (`{roadmap.get('daily_target_pct', 0.0):.2f}%`)\n"
            msg += f"Progreso Hoy: `${roadmap.get('usd_progress_today', 0.0):+.4f}`\n"
            msg += f"Trades Ganadores Faltantes Hoy: `{roadmap.get('trades_needed_today', 0)}`\n"
            if not roadmap.get('on_track', False) and roadmap.get('trades_needed_today', 0) > 0:
                msg += f"⚠️ *ALERTA:* Velocidad baja. Necesitamos `{roadmap.get('trades_needed_today', 0)}` aciertos de `~${roadmap.get('avg_win_usd', 0.0):.2f}`.\n"

        # ═══════════════════════════════════════════════════════════════
        # CTOS PHASE 3: FORENSIC ENRICHMENT SECTIONS
        # ═══════════════════════════════════════════════════════════════
        
        # A) Prediction Audit: What was predicted vs reality
        _p_audit = trade_data.get('prediction_audit', {})
        _p_mag = _p_audit.get('predicted_magnitude')
        _p_dur = _p_audit.get('predicted_duration_bars')
        _p_target = _p_audit.get('predicted_target_price')
        _optimal_exit = _p_audit.get('optimal_exit_price')
        _missed_profit = _p_audit.get('missed_profit_pct')
        _was_correct = _p_audit.get('was_correct')
        
        _open_size = trade_data.get('open_size_usd', 0.0)
        _close_size = trade_data.get('close_size_usd', 0.0)

        if _p_mag or _p_dur:
            pred_icon = "✅" if _was_correct else "❌"
            msg += f"🧠 *Predicción de Estrategia:* {pred_icon}\n"
            if _p_mag:
                msg += f"   Se predijo magnitud: `+{float(_p_mag)*100:.2f}%`"
                if _p_target:
                    msg += f" → `${float(_p_target):,.2f}`"
                msg += "\n"
                msg += f"   Realidad lograda: `{td.net_pnl_pct:+,.2f}%`\n"
            if _p_dur:
                msg += f"   Se predijo tiempo: `{_p_dur} barras`\n"
                msg += f"   Realidad tiempo: `{td.duration}`\n"
            msg += f"   {pred_icon} Predicción {'ACERTADA' if _was_correct else 'FALLIDA'}\n"
            if _optimal_exit:
                msg += f"   💡 Punto óptimo (MFE): `${float(_optimal_exit):,.4f}`\n"
            if _missed_profit and float(_missed_profit) > 0:
                msg += f"   🕳️ Ganancia perdida: `{float(_missed_profit)*100:.2f}%`\n"

        # B) Size Tracking: Open → Close
        _open_sz = trade_data.get('open_size_usd', 0.0)
        _close_sz = trade_data.get('close_size_usd', 0.0)
        if _open_sz > 0 and _close_sz > 0:
            _delta_sz = _close_sz - _open_sz
            msg += f"\n📦 *Tamaños de Posición (Apertura vs Cierre):*\n"
            msg += f"   Se abrió con:  `${_open_sz:,.2f}` USD\n"
            msg += f"   Se cerró con: `${_close_sz:,.2f}` USD\n"
            msg += f"   Diferencia (PnL Nocional): `${_delta_sz:+,.2f}` USD\n"

        # C) Strategy Attribution
        _opener = trade_data.get('opener_strategy')
        _closer = trade_data.get('closer_strategy')
        if _opener or _closer:
            msg += f"\n🔄 *Atribución de Estrategia:*\n"
            if _opener: msg += f"   Abrió: `{_opener}`\n"
            if _closer: msg += f"   Cerró: `{_closer}` ({exit_emoji} `{td.exit_reason}`)\n"

        # D) Session Growth Progress
        _session_start = trade_data.get('session_start_equity', 0.0)
        _session_growth = trade_data.get('session_growth_pct', 0.0)
        _daily_target = trade_data.get('daily_target_pct', 4.73)
        _growth_progress = trade_data.get('growth_progress', 0.0)
        if _session_start > 0 and _daily_target > 0:
            # Progress bar: 10 blocks
            filled = int(_growth_progress * 10)
            bar = '▓' * filled + '░' * (10 - filled)
            msg += f"\n📈 *Meta Diaria ({_daily_target:.2f}%):*\n"
            msg += f"   `{bar}` `{abs(_session_growth):.2f}%`\n"
            msg += f"   Sesión inicio: `${_session_start:,.2f}`\n"
            _session_net = trade_data.get('session_net_pnl', 0.0)
            msg += f"   Acumulado sesión: `${_session_net:+,.4f}`\n"

        # ═══════════════════════════════════════════════════════════════
        # SOPHIA-GLOBAL FIX: WR display for CLOSE notifications only
        # QUÉ: Solo muestra WR en trades cerrados (win_rate >= 0).
        # POR QUÉ: Entries envían win_rate=-1 como sentinel.
        # PARA QUÉ: Evitar mostrar "WR: 100%" en un trade que aún no cerró.
        # ═══════════════════════════════════════════════════════════════
        if td.win_rate >= 0 and (td.session_wins > 0 or td.session_losses > 0):
            session_total = td.session_wins + td.session_losses
            msg += f"\n🏆 WR Global: `{td.win_rate:.1f}%` ({td.session_wins}W/{td.session_losses}L de {session_total})"
            
            # FORENSIC-V15: Strategy Specific WR
            strat_wr = trade_data.get('strat_win_rate', -1.0)
            if strat_wr >= 0:
                strat_w = trade_data.get('strat_wins', 0)
                strat_l = trade_data.get('strat_losses', 0)
                strat_tot = strat_w + strat_l
                if strat_tot > 0:
                    msg += f"\n🎯 WR {td.strategy}: `{strat_wr:.1f}%` ({strat_w}W/{strat_l}L de {strat_tot})"

        # ═══════════════════════════════════════════════════════════════
        # XAI AUTOPSY DISPLAY (Phase Omega)
        # ═══════════════════════════════════════════════════════════════
        xai_autopsy = trade_data.get('xai_autopsy')
        if xai_autopsy:
            msg += f"\n\n🧠 *Autopsia Sophia (XAI):*\n{xai_autopsy}"
        sophia_narrative = trade_data.get('sophia_narrative')
        if sophia_narrative:
            msg += f"\n💬 _{sophia_narrative}_"

        # ═══════════════════════════════════════════════════════════════
        # FORENSIC FIX #2: EXIT BALLOT DISPLAY
        # QUÉ: Muestra qué estrategias de cierre votaron EXIT vs HOLD.
        # POR QUÉ: Sin esto, no sabemos por qué se cerró o por qué NO
        #   se cerró a tiempo. Es la pieza clave para diagnosticar pérdidas.
        # PARA QUÉ: El usuario ve exactamente quién mandó cerrar.
        # ═══════════════════════════════════════════════════════════════
        _exit_ballot = trade_data.get('exit_ballot')
        if _exit_ballot and isinstance(_exit_ballot, dict):
            _exit_v = _exit_ballot.get('exit_voters', [])
            _hold_v = _exit_ballot.get('hold_voters', [])
            if _exit_v or _hold_v:
                msg += f"\n\n🗳️ *Votación de Cierre:*\n"
                for voter in _exit_v:
                    msg += f"   🔴 EXIT: `{voter}`\n"
                for voter in _hold_v:
                    msg += f"   🟢 HOLD: `{voter}`\n"

        # ═══════════════════════════════════════════════════════════════
        # FORENSIC FIX #2: DIAGNOSTIC STATS
        # QUÉ: Estadísticas diagnósticas para entender salud del sistema.
        # POR QUÉ: Si avg_loss > avg_win, necesitas WR > 57% para ser rentable.
        # PARA QUÉ: Detectar R:R invertido y ajustar estrategias.
        # ═══════════════════════════════════════════════════════════════
        _diag = trade_data.get('diagnostic_stats')
        if _diag and isinstance(_diag, dict):
            avg_w = _diag.get('avg_win_pnl', 0.0)
            avg_l = _diag.get('avg_loss_pnl', 0.0)
            pf = _diag.get('profit_factor', 0.0)
            msg += f"\n📊 *Diagnóstico Estadístico:*\n"
            msg += f"   Avg Win: `${avg_w:,.4f}` | Avg Loss: `${avg_l:,.4f}`\n"
            msg += f"   Profit Factor: `{pf:,.2f}`\n"
            if avg_l > avg_w and avg_l > 0:
                min_wr = avg_l / (avg_w + avg_l) * 100 if (avg_w + avg_l) > 0 else 50
                msg += f"   ⚠️ R:R Invertido. WR mínimo para profit: `{min_wr:.1f}%`\n"

        # E) Auditoría Inteligente: Sugerencias basadas en el resultado
        if td.net_pnl < 0:
            msg += f"\n\n💡 *Sugerencia Forense:* El trade se cerró en pérdida. El Oráculo reporta: `{td.exit_reason}`. Revisa si el `Alpha Decay` intervino demasiado temprano o si la estrategia `{_opener}` no capturó el momentum correcto."
        elif _missed_profit and float(_missed_profit) > 0.005: # Si se perdió más de 0.5%
            msg += f"\n\n💡 *Sugerencia Forense:* Cerramos en ganancia, pero perdimos un movimiento de `{float(_missed_profit)*100:.2f}%`. Revisa los parámetros de trailing stop de la estrategia `{_closer}`."

        if trade_data.get('skip_telegram'):
            # Just log to telemetry file and return
            import os, json
            os.makedirs("dashboard/data", exist_ok=True)
            try:
                with open("dashboard/data/backtest_telemetry_spam.jsonl", "a", encoding="utf-8") as f:
                    log_entry = {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "priority": "INFO",
                        "message": msg
                    }
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            except Exception: pass
            return

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
    # LAYER 6: SYSTEM STARTUP IDENTIFICATION
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def send_system_startup(mode: str, context: Dict[str, Any]) -> None:
        """
        🚀 System Startup Notification — Identifies WHAT is running.

        PROFESSOR METHOD:
        - QUÉ: Mensaje de inicio que identifica claramente si es PRODUCCIÓN o BACKTEST.
        - POR QUÉ: El usuario necesita saber inmediatamente qué sistema arrancó.
        - PARA QUÉ: Distinguir entre sesiones y evitar confusión en Telegram.
        - CÓMO: Envía un mensaje rico con todos los parámetros de la sesión.
        - CUÁNDO: Al inicio de main.py (producción) o run_god_mode_backtest.py.
        - DÓNDE: Primer mensaje que llega a Telegram en cualquier sesión.
        - QUIÉN: Notifier (invocado por main.py o backtest script).
        """
        ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

        if mode == "PRODUCTION":
            emoji = "🚀"
            title = "TRADER GEMINI — PRODUCCIÓN LIVE"
            mode_detail = context.get('trading_mode', 'FUTURES').upper()
        elif mode == "BACKTEST":
            emoji = "🧪"
            title = "TRADER GEMINI — BACKTEST GOD MODE"
            mode_detail = f"{context.get('days', '?')} días"
        elif mode == "PAPER":
            emoji = "📝"
            title = "TRADER GEMINI — PAPER TRADING"
            mode_detail = context.get('trading_mode', 'DEMO').upper()
        else:
            emoji = "⚙️"
            title = f"TRADER GEMINI — {mode.upper()}"
            mode_detail = mode

        msg = f"{emoji} *{title}* {emoji}\n\n"
        msg += f"*Modo:* `{mode_detail}`\n"
        msg += f"*Capital:* `${context.get('capital', 0):,.2f}`\n"
        msg += f"*Leverage:* `{context.get('leverage', 1)}x`\n"
        msg += f"*Símbolos:* `{context.get('symbols_count', 0)}` activos\n"
        msg += f"*Estrategias:* `{context.get('strategies_count', 0)}` registradas\n"

        if mode == "BACKTEST":
            msg += f"\n*Configuración Backtest:*\n"
            msg += f"Periodo: `{context.get('days', '?')} días`\n"
            msg += f"Capital inicial: `${context.get('capital', 0):,.2f}`\n"
            seed = context.get('seed', 42)
            msg += f"Seed: `{seed}` (determinístico)\n"
            msg += f"Epochs: `{context.get('total_epochs', '?'):,}`\n"
        else:
            msg += f"\n*Conexión:*\n"
            testnet = context.get('testnet', False)
            demo = context.get('demo', False)
            if testnet:
                msg += f"Exchange: `Binance TESTNET`\n"
            elif demo:
                msg += f"Exchange: `Binance DEMO`\n"
            else:
                msg += f"Exchange: `Binance MAINNET` ⚠️\n"

        # Risk params
        msg += f"\n*Parámetros de Riesgo:*\n"
        msg += f"Max Drawdown: `{context.get('max_drawdown', 0):.1f}%`\n"
        msg += f"TP Scalping: `{context.get('tp_scalp', 0)*100:.2f}%`\n"
        msg += f"SL Scalping: `{context.get('sl_scalp', 0)*100:.2f}%`\n"
        msg += f"Kill Switch: `Activo`\n"

        # Symbols list (abbreviated)
        symbols = context.get('symbols_list', [])
        if symbols:
            symbols_display = symbols[:10]
            symbols_str = ", ".join(f"`{s}`" for s in symbols_display)
            if len(symbols) > 10:
                symbols_str += f" +{len(symbols) - 10} más"
            msg += f"\n*Símbolos:*\n{symbols_str}\n"

        msg += f"\n🕒 `{ts}`"

        Notifier.send_telegram(msg, priority="CRITICAL")

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 7: STRATEGY PULSE — IDLE MARKET INTELLIGENCE
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def send_strategy_pulse(pulse_data: Dict[str, Any]) -> None:
        """
        🫀 Strategy Pulse — Reports what strategies and market are doing.

        PROFESSOR METHOD:
        - QUÉ: Reporte periódico del estado de estrategias cuando NO hay trades.
        - POR QUÉ: Sin este pulso, el usuario no sabe si el sistema está vivo
            o muerto cuando no genera señales.
        - PARA QUÉ: Visibilidad total — saber POR QUÉ no se opera.
        - CÓMO: Cada 15 min (configurable), si no hubo trades recientes,
            envía un resumen de lo que cada estrategia ve en el mercado.
        - CUÁNDO: Invocado por metrics_heartbeat_loop() en main.py.
        - DÓNDE: Telegram (canal principal).
        - QUIÉN: Notifier + Engine metrics + Portfolio stats.
        """
        ts = datetime.now(timezone.utc).strftime('%H:%M:%S')

        msg = f"🫀 *PULSO DEL SISTEMA* 🫀\n\n"

        # System vitals
        equity = pulse_data.get('equity', 0)
        initial = pulse_data.get('initial_capital', 13.0)
        growth = ((equity - initial) / initial * 100) if initial > 0 else 0
        msg += f"*Estado:* ✅ Operativo\n"
        msg += f"*Equity:* `${equity:,.2f}` ({growth:+,.2f}%)\n"

        # Positions
        open_positions = pulse_data.get('open_positions', 0)
        open_symbols = pulse_data.get('open_symbols', [])
        msg += f"*Posiciones abiertas:* `{open_positions}`\n"
        if open_symbols:
            msg += f"  → {', '.join(f'`{s}`' for s in open_symbols[:8])}\n"

        # Engine metrics
        events_processed = pulse_data.get('events_processed', 0)
        signals_generated = pulse_data.get('signals_generated', 0)
        signals_rejected = pulse_data.get('signals_rejected', 0)
        avg_latency = pulse_data.get('avg_latency_ms', 0)
        msg += f"\n*Motor (Engine):*\n"
        msg += f"Eventos procesados: `{events_processed:,}`\n"
        msg += f"Señales generadas: `{signals_generated:,}`\n"
        msg += f"Señales rechazadas: `{signals_rejected:,}`\n"
        msg += f"Latencia promedio: `{avg_latency:.2f}ms`\n"

        # Market regime
        regime = pulse_data.get('market_regime', 'UNKNOWN')
        regime_emojis = {
            'TRENDING_BULL': '📈', 'TRENDING_BEAR': '📉',
            'RANGING': '↔️', 'HIGH_VOLATILITY': '🌪️',
            'CHOPPY': '🔀', 'UNKNOWN': '❓'
        }
        regime_emoji = regime_emojis.get(regime, '❓')
        msg += f"\n*Mercado:*\n"
        msg += f"Régimen: {regime_emoji} `{regime}`\n"

        btc_price = pulse_data.get('btc_price', 0)
        if btc_price > 0:
            msg += f"BTC: `${btc_price:,.2f}`\n"

        # Why no trades?
        rejection_reasons = pulse_data.get('rejection_reasons', {})
        if rejection_reasons:
            msg += f"\n*¿Por qué no se opera?*\n"
            # Sort by count descending, show top 5
            sorted_reasons = sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)[:5]
            for reason, count in sorted_reasons:
                msg += f"  🚫 `{reason}`: {count}x\n"

        # Strategies status
        strategies_status = pulse_data.get('strategies_status', [])
        if strategies_status:
            msg += f"\n*Estrategias activas:*\n"
            for strat in strategies_status[:8]:
                name = strat.get('name', 'Unknown')
                horizon = strat.get('horizon', '?')
                signals = strat.get('signals_emitted', 0)
                h_emoji = "⚡" if horizon == "SCALPING" else "🌊"
                msg += f"  {h_emoji} `{name}`: {signals} señales\n"

        # Session stats
        session_trades = pulse_data.get('session_trades', 0)
        session_wins = pulse_data.get('session_wins', 0)
        session_losses = pulse_data.get('session_losses', 0)
        if session_trades > 0:
            wr = (session_wins / session_trades * 100) if session_trades > 0 else 0
            msg += f"\n*Sesión:*\n"
            msg += f"Trades: `{session_trades}` | WR: `{wr:.1f}%` ({session_wins}W/{session_losses}L)\n"
        else:
            last_trade_ago = pulse_data.get('minutes_since_last_trade', None)
            if last_trade_ago is not None:
                msg += f"\n⏳ Sin trades en esta sesión ({last_trade_ago:.0f} min)\n"
            else:
                msg += f"\n⏳ Sin trades en esta sesión\n"

        msg += f"\n🕒 `{ts} UTC`"

        Notifier.send_telegram(msg, priority="INFO")

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 8: BACKTEST PROGRESS & COMPLETION
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def send_backtest_progress(progress_data: Dict[str, Any]) -> None:
        """
        📊 Backtest Progress Notification — Sent at key milestones.
        """
        pct = progress_data.get('progress_pct', 0)
        equity = progress_data.get('equity', 0)
        trades = progress_data.get('trades', 0)
        elapsed = progress_data.get('elapsed_seconds', 0)
        open_pos = progress_data.get('open_positions', 0)
        epoch = progress_data.get('epoch', 0)
        total = progress_data.get('total_epochs', 1)

        bar_fill = int(pct / 10)
        bar = "█" * bar_fill + "░" * (10 - bar_fill)

        msg = f"🧪 *BACKTEST PROGRESO* 🧪\n\n"
        msg += f"[{bar}] `{pct}%`\n"
        msg += f"Epoch: `{epoch:,}/{total:,}`\n"
        msg += f"Equity: `${equity:,.2f}`\n"
        msg += f"Trades ejecutados: `{trades}`\n"
        msg += f"Posiciones abiertas: `{open_pos}`\n"
        msg += f"Tiempo: `{elapsed:.0f}s`\n"

        Notifier.send_telegram(msg, priority="INFO")

    @staticmethod
    def send_backtest_complete(results: Dict[str, Any]) -> None:
        """
        🏁 Backtest Completion — Final results summary.
        """
        Notifier.flush_backtest_trades()
        
        msg = f"🏁 *BACKTEST COMPLETADO* 🏁\n\n"

        config = results.get('config', {})
        metrics = results.get('metrics', {})

        initial = config.get('initial_capital', results.get('initial_capital', 13.0))
        final = metrics.get('final_capital', results.get('final_equity', 0.0))
        pnl = final - initial
        pnl_pct = (pnl / initial * 100) if initial > 0 else 0
        result_emoji = "🟢" if pnl > 0 else "🔴"

        msg += f"*Resultado:* {result_emoji}\n"
        msg += f"Capital inicial: `${initial:,.2f}`\n"
        msg += f"Capital final: `${final:,.2f}`\n"
        msg += f"PnL Neto: `${pnl:+,.4f}` ({pnl_pct:+,.2f}%)\n\n"

        total = metrics.get('total_trades', results.get('total_trades', 0))
        wins = metrics.get('wins', results.get('wins', 0))
        losses = metrics.get('losses', results.get('losses', 0))
        wr = metrics.get('win_rate', results.get('win_rate', 0.0))
        
        msg += f"*Trades:*\n"
        msg += f"Total: `{total}` | Wins: `{wins}` | Losses: `{losses}`\n"
        msg += f"Win Rate: `{wr:.1f}%`\n"

        sharpe = metrics.get('sharpe_ratio', results.get('sharpe', 0.0))
        max_dd = metrics.get('max_drawdown_pct', results.get('max_drawdown', 0.0))
        if sharpe != 0:
            msg += f"Sharpe: `{sharpe:.2f}`\n"
        msg += f"Max Drawdown: `{max_dd:.2f}%`\n"

        elapsed = metrics.get('elapsed_seconds', results.get('elapsed_seconds', 0))
        msg += f"\n⏱️ Duración: `{elapsed:.0f}s`\n"
        msg += f"Días simulados: `{config.get('days', results.get('days', 0))}`\n"
        msg += f"Símbolos: `{config.get('num_symbols', results.get('symbols_count', 0))}`\n"

        Notifier.send_telegram(msg, priority="CRITICAL")

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 9: ML TRAINING & STRATEGY LEADERBOARD
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def send_ml_training_update(symbol: str, horizon: str, status: str, details: Dict[str, Any] = None) -> None:
        """
        🧠 Notificación de Progreso de Entrenamiento ML.
        Estados: STARTING, SUCCESS, FAILED, REJECTED
        """
        if details is None:
            details = {}

        # Mapeo de emojis y descripciones por estado
        status_map = {
            "STARTING": ("🔄", "Iniciando Entrenamiento", "INFO"),
            "SUCCESS": ("✨", "Entrenamiento Exitoso", "INFO"),
            "FAILED": ("⚠️", "Error en Entrenamiento", "WARNING"),
            "REJECTED": ("🛡️", "Modelo Rechazado (Quality Guard)", "WARNING")
        }
        
        emoji, status_desc, priority = status_map.get(status, ("❓", f"Estado: {status}", "INFO"))
        
        msg = f"{emoji} *ML TRAINING: {symbol}* {emoji}\n\n"
        msg += f"*Horizonte:* `{horizon}`\n"
        msg += f"*Estado:* `{status_desc}`\n"

        if status == "SUCCESS":
            score = details.get("score", 0)
            features = details.get("features", 0)
            duration = details.get("duration", 0)
            msg += f"\n*Métricas de Éxito:*\n"
            msg += f"Puntuación (Score): `{score:.3f}`\n"
            msg += f"Features Activos: `{features}`\n"
            msg += f"Tiempo: `{duration:.1f}s`\n"
        elif status == "REJECTED":
            score = details.get("score", 0)
            min_acc = details.get("min_acc", 0)
            msg += f"\n*Motivo de Rechazo:*\n"
            msg += f"Puntuación (`{score:.3f}`) < Mínimo Requerido (`{min_acc:.3f}`)\n"
            msg += f"_El modelo es peor que una predicción aleatoria._\n"
        elif status == "FAILED":
            error = details.get("error", "Unknown")
            msg += f"\n*Error Reportado:*\n`{error}`\n"
            
        Notifier.send_telegram(msg, priority=priority)

    @staticmethod
    def send_strategy_leaderboard(strategy_performance: Dict[str, Any], title_prefix: str = "") -> None:
        """
        🏆 Strategy Leaderboard: Top 5 Mejores y Peores Estrategias.
        """
        if not strategy_performance:
            return

        # Filtrar solo estrategias con al menos 1 trade
        active_strats = [
            (sid, perf) for sid, perf in strategy_performance.items() 
            if (perf.get("wins", 0) + perf.get("losses", 0)) > 0
        ]
        
        if not active_strats:
            return

        # Ordenar por PnL descendente
        sorted_strats = sorted(active_strats, key=lambda x: x[1].get("pnl", 0), reverse=True)
        
        top_5 = sorted_strats[:5]
        bottom_5 = sorted_strats[-5:] if len(sorted_strats) > 5 else []
        
        prefix = f"{title_prefix} " if title_prefix else ""
        msg = f"🏆 *{prefix}STRATEGY LEADERBOARD* 🏆\n\n"
        msg += f"Total de estrategias con actividad: `{len(active_strats)}`\n\n"

        msg += f"🌟 *TOP 5 GANADORAS:*\n"
        for i, (sid, perf) in enumerate(top_5, 1):
            pnl = perf.get('pnl', 0)
            wins = perf.get('wins', 0)
            losses = perf.get('losses', 0)
            total = wins + losses
            wr = (wins / total * 100) if total > 0 else 0
            msg += f"{i}. `{sid}`\n   💰 `${pnl:+.4f}` | WR: `{wr:.0f}%` ({wins}W/{losses}L)\n"

        if bottom_5:
            msg += f"\n📉 *TOP 5 PERDEDORAS:*\n"
            # Invertimos para mostrar la peor al final
            for i, (sid, perf) in enumerate(reversed(bottom_5), 1):
                pnl = perf.get('pnl', 0)
                wins = perf.get('wins', 0)
                losses = perf.get('losses', 0)
                total = wins + losses
                wr = (wins / total * 100) if total > 0 else 0
                msg += f"{i}. `{sid}`\n   🩸 `${pnl:+.4f}` | WR: `{wr:.0f}%` ({wins}W/{losses}L)\n"

        msg += f"\n🕒 `{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}`"
        
        Notifier.send_telegram(msg, priority="INFO")

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
