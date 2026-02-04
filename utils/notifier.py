import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone
from config import Config
from utils.logger import logger

class Notifier:
    """
    📢 MOTOR DE NOTIFICACIONES (Phase 4)
    
    PROFESSOR METHOD:
    - QUÉ: Centro de alertas multicanal (Telegram/Email).
    - POR QUÉ: El trader no puede estar 24/7 pegado al monitor.
    - PARA QUÉ: Recibir avisos inmediatos de trades y fallos críticos.
    - CÓMO: REST API para Telegram y protocolo SMTP para Email.
    """
    
    @staticmethod
    def send_telegram(message, priority="INFO"):
        """Envía alerta a Telegram (Rule 4.2)"""
        if not Config.Observability.TELEGRAM_ENABLED:
            return
            
        # Prioridad visual
        header = "🤖 **TRADER GEMINI**"
        if priority == "CRITICAL": header = "🚨 **CRITICAL ALERT**"
        elif priority == "WARNING": header = "⚠️ **WARNING**"
        
        full_message = f"{header}\n\n{message}\n\n🕒 {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC"
        
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
    def send_email(subject, body, is_html=False):
        """Envía reporte o alerta por Email (Rule 4.2)"""
        if not Config.Observability.EMAIL_ENABLED:
            return
            
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
        """Formatea y envía notificación de operación"""
        type_str = "COMPRA (LONG)" if direction == "BUY" else "VENTA (SELL)"
        emoji = "🟢" if direction == "BUY" else "🔴"
        
        msg = f"{emoji} **Trade Executed**\n"
        msg += f"Symbol: `{symbol}`\n"
        msg += f"Action: {type_str}\n"
        msg += f"Price: `${price:,.4f}`\n"
        msg += f"Qty: `{qty}`"
        
        if pnl is not None:
            pnl_emoji = "💰" if pnl > 0 else "📉"
            msg += f"\n\n{pnl_emoji} **PnL Realized: ${pnl:,.2f}**"
            if winrate:
                msg += f"\n🏆 Win Rate: `{winrate:.1f}%`"
                
        Notifier.send_telegram(msg)
